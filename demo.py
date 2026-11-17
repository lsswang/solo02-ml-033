"""End-to-end demo of additive dictionary learning and one-to-one retrieval.

Run ``python demo.py``.  It trains the additive dictionary, encodes the
external library into a self-contained catalog artifact, runs a query, and
writes the three deliverables to ``artifacts/``:

* ``model.json``   - final dictionary, per-round objectives, final codes,
                     per-image errors and the complete objective;
* ``catalog.json`` - the compressed query-side product: shared codebook plus
                     code tuples only;
* ``results.json`` - structured one-to-one retrieval output.

The same JSON object is printed to stdout.  A consumer can later verify the
artifact with ``python inspect_artifact.py`` without any training data.
"""
from __future__ import annotations

import json
from pathlib import Path

from dictionary import fit
from retrieval import build_catalog, search


def main() -> dict:
    root = Path(__file__).parent
    data = json.loads((root / "fixtures/visual.json").read_text())
    model = fit(
        data["training_images"],
        data["initial"],
        data["iterations"],
        data["ridge"],
    )
    catalog = build_catalog(data["catalog"], model)
    results = search(data["query"], catalog, top_k=data["top_k"])
    # The ordinary single-region query case, kept as a baseline.
    single_results = search(
        data["single_query"], catalog, top_k=data["top_k"]
    )

    # Model-change linkage: same query, same raw external library, same top_k,
    # but a different legal training set + initial dictionary.  The new
    # dictionary and codes must genuinely feed the same search.
    alt_model = fit(
        data["alternate_training_images"],
        data["alternate_initial"],
        data["iterations"],
        data["ridge"],
    )
    alt_catalog = build_catalog(data["catalog"], alt_model)
    alt_results = search(data["query"], alt_catalog, top_k=data["top_k"])

    payload = {
        "model": model,
        "catalog": catalog,
        "results": results,
        "single_region_query": {
            "query": data["single_query"],
            "results": single_results,
        },
        "model_change": {
            "query": data["query"],
            "top_k": data["top_k"],
            "model": alt_model,
            "catalog": alt_catalog,
            "results": alt_results,
        },
    }

    def ranking(view: dict) -> list[dict]:
        return [
            {
                "id": r["id"],
                "cost": r["cost"],
                "region_indices": r["region_indices"],
                "pair_costs": r["pair_costs"],
            }
            for r in view["results"]
        ]

    payload["summary"] = {
        "query": data["query"],
        "ranking": ranking(results),
        "insufficient": results["insufficient"],
        "single_region": {
            "query": data["single_query"],
            "ranking": ranking(single_results),
        },
        "model_change": {
            "base_codebook_digest": catalog["codebook_digest"],
            "alternate_codebook_digest": alt_catalog["codebook_digest"],
            "catalog_codes_changed": catalog["images"] != alt_catalog["images"],
            "rankings_changed": ranking(results) != ranking(alt_results),
            "base_ranking": ranking(results),
            "alternate_ranking": ranking(alt_results),
        },
        "training_objective": model["objective"],
        "rounds": [
            {
                "round": r["round"],
                "before": r["objective_before_update"],
                "after": r["objective_after_update"],
            }
            for r in model["rounds"]
        ],
    }

    out_dir = root / "artifacts"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n")
    (out_dir / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "model_change.json").write_text(
        json.dumps(payload["model_change"], indent=2) + "\n"
    )
    (out_dir / "single_region_query.json").write_text(
        json.dumps(payload["single_region_query"], indent=2) + "\n"
    )
    (out_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2) + "\n"
    )
    return payload


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
