"""Inspect a delivered catalog artifact without any training state.

Usage::

    python inspect_artifact.py [artifacts/catalog.json]

It decodes every code tuple against the shared dictionary embedded in the
artifact, recomputes one-to-one matching for the demo query, and prints the
result.  It does not import anything from training and will work on any legal
``additive_visual_catalog_v1`` artifact.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np


def reconstruct(code, codebooks):
    out = np.zeros(codebooks.shape[2], dtype=np.float64)
    for layer, word in enumerate(code):
        out += codebooks[layer, word]
    return out


def best_injection(cost):
    r, m = cost.shape
    best = None
    for perm in itertools.permutations(range(m), r):
        total = sum(cost[j, perm[j]] for j in range(r))
        if best is None or total < best[0]:
            best = (float(total), perm)
    return best


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/catalog.json")
    catalog = json.loads(path.read_text())
    codebooks = np.asarray(catalog["codebooks"], dtype=np.float64)

    fixture = json.loads(Path("fixtures/visual.json").read_text())
    query = np.asarray(fixture["query"], dtype=np.float64)
    r = query.shape[0]

    audit = {"codebook_digest": catalog.get("codebook_digest"), "images": {}}
    for image_id in sorted(catalog["images"]):
        codes = catalog["images"][image_id]
        decoded = np.asarray(
            [reconstruct(code, codebooks) for code in codes], dtype=np.float64
        ).reshape(len(codes), codebooks.shape[2])
        audit["images"][image_id] = {
            "codes": codes,
            "decoded": decoded.tolist(),
            "regions": len(codes),
        }
        status = "ok"
        detail = None
        if len(codes) < r:
            status = "insufficient_regions"
            detail = f"{len(codes)} regions < {r} query regions"
        else:
            cost = np.sum((query[:, None, :] - decoded[None, :, :]) ** 2, axis=2)
            total, perm = best_injection(cost)
            status = "ok"
            detail = {"cost": total / r, "assignment": list(perm)}
        audit["images"][image_id]["status"] = status
        audit["images"][image_id]["match"] = detail

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
