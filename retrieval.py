"""Distinct-region one-to-one retrieval over an additive dictionary catalog.

The query side receives only the compressed catalog (shared codebook plus
per-image code tuples) and its own query vectors.  Reconstructed region
features are produced by :func:`dictionary.decode` from the delivered
codebook; original library vectors and training data are never consulted.

For a query with ``R`` regions and a library image with ``M`` reconstructed
regions:

* if ``M < R`` the image cannot provide a one-to-one cover.  It is reported as
  ``insufficient_regions`` and is *not* ranked;
* otherwise the cost is

      D = min over injections pi:{0..R-1} -> {0..M-1}
          ( sum_j ||q_j - decoded[pi(j)]||^2 ) / R

  The minimum is over *injections*, so no reconstructed region may be used by
  two query regions.  The search is exhaustive (``R <= 4``, ``M <= 6``, at
  most 360 ordered selections), so the returned assignment is the global
  optimum.

Ranking is ascending by ``D``; exact ties break on ``image_id`` Unicode code
point order.  At most ``top_k`` feasible images are returned, and the reason
for every excluded insufficient image is preserved.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping

import numpy as np

from dictionary import (
    MAX_CATALOG_IMAGES,
    MAX_CATALOG_REGIONS,
    MAX_EXTERNAL_CODEBOOK_ABS,
    MAX_QUERY_REGIONS,
    MAX_TOP_K,
    MAX_INPUT_VECTOR_ABS,
    MIN_QUERY_REGIONS,
    _best_codes,
    _reconstruct,
    _validate_codes,
    _validate_vectors,
    codebook_digest,
    extract_codebooks,
)


def _validate_catalog_id(image_id: Any) -> str:
    if not isinstance(image_id, str):
        raise ValueError("catalog image_id values must be strings")
    if image_id == "":
        raise ValueError("catalog image_id values must be non-empty strings")
    return image_id


def _validate_catalog(
    catalog: Any, *, model: Any = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ``(codebooks, {image_id: codes_array})``.

    Two forms are accepted:

    * a self-contained artifact mapping with ``"codebooks"`` and ``"images"``
      (this is what :func:`build_catalog` emits);
    * a plain ``{image_id: codes}`` mapping together with a legal ``model``
      supplying the dictionary.
    """
    if not isinstance(catalog, Mapping):
        raise ValueError("catalog must be a mapping")

    images: Any
    if "codebooks" in catalog:
        # An artifact emitted by build_catalog carries a dictionary that may
        # have been produced by training, so it is not subject to the external
        # magnitude bound.  A plain external mapping still is.
        artifact_tagged = str(catalog.get("format", "")).startswith(
            "additive_visual_catalog_"
        )
        bound = float("inf") if artifact_tagged else MAX_EXTERNAL_CODEBOOK_ABS
        cbs = extract_codebooks(
            catalog["codebooks"], name="catalog codebooks", max_abs=bound
        )
        if "images" in catalog:
            images = catalog["images"]
        elif "codes" in catalog:
            images = catalog["codes"]
        else:
            raise ValueError("catalog artifact must contain an 'images' mapping")
        if model is not None:
            supplied = extract_codebooks(model, name="model codebooks")
            if codebook_digest(supplied) != codebook_digest(cbs):
                raise ValueError(
                    "model codebooks differ from the catalog codebooks; refusing "
                    "to decode the catalog with a stale dictionary"
                )
    else:
        if model is None:
            raise ValueError(
                "catalog without embedded 'codebooks' requires a legal model"
            )
        cbs = extract_codebooks(model, name="model codebooks")
        images = catalog

    if not isinstance(images, Mapping):
        raise ValueError("catalog images must be a mapping of image_id -> codes")
    if len(images) > MAX_CATALOG_IMAGES:
        raise ValueError(
            f"catalog may contain at most {MAX_CATALOG_IMAGES} images"
        )

    L, K, _ = cbs.shape
    decoded_codes: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for key, codes in images.items():
        image_id = _validate_catalog_id(key)
        if image_id in seen:  # pragma: no cover - mapping keys are unique
            raise ValueError(f"duplicate catalog image_id {image_id!r}")
        seen.add(image_id)
        arr = _validate_codes(
            codes, L, K, name=f"catalog[{image_id!r}]"
        )
        if arr.shape[0] > MAX_CATALOG_REGIONS:
            raise ValueError(
                f"catalog[{image_id!r}] has {arr.shape[0]} regions; "
                f"at most {MAX_CATALOG_REGIONS} are allowed"
            )
        decoded_codes[image_id] = arr
    return cbs, decoded_codes


def build_catalog(images: Any, model: Any) -> dict[str, Any]:
    """Encode external library regions with the *actual final* dictionary.

    Parameters
    ----------
    images:
        Mapping ``image_id -> sequence of region vectors``.  Every key must be
        a non-empty unique string; up to 8 images with 0..6 regions each.
    model:
        Trained model dict, artifact dict, or a raw ``(L, K, d)`` codebook.

    Returns
    -------
    dict
        Self-contained artifact with the shared ``codebooks`` and per-image
        ``images`` codes, plus linking metadata.
    """
    cbs = extract_codebooks(model, name="model")
    L, K, d = cbs.shape

    if not isinstance(images, Mapping):
        raise ValueError("catalog images must be a mapping of image_id -> regions")
    if len(images) > MAX_CATALOG_IMAGES:
        raise ValueError(f"catalog may contain at most {MAX_CATALOG_IMAGES} images")

    encoded: dict[str, list[list[int]]] = {}
    region_counts: dict[str, int] = {}
    for key, regions in images.items():
        image_id = _validate_catalog_id(key)
        if image_id in encoded:  # pragma: no cover
            raise ValueError(f"duplicate catalog image_id {image_id!r}")
        arr = _validate_vectors(
            regions,
            d,
            name=f"catalog[{image_id!r}]",
            max_abs=MAX_INPUT_VECTOR_ABS,
            min_count=0,
            max_count=MAX_CATALOG_REGIONS,
        )
        if arr.shape[0]:
            codes_arr, _ = _best_codes(arr, cbs)
        else:
            codes_arr = np.zeros((0, L), dtype=np.int64)
        encoded[image_id] = [
            [int(v) for v in row] for row in codes_arr
        ]
        region_counts[image_id] = int(arr.shape[0])

    return {
        "format": "additive_visual_catalog_v1",
        "codebooks": cbs.tolist(),
        "codebook_digest": codebook_digest(cbs),
        "layers": int(L),
        "codebook_size": int(K),
        "dim": int(d),
        "images": encoded,
        "region_counts": region_counts,
    }


def _best_injection(cost: np.ndarray) -> tuple[int, tuple[int, ...], list[float]]:
    """Global minimum over injections for an ``R x M`` cost matrix.

    Returns ``(best_total_squared, assignment, pair_costs)``.  Ties resolve to
    the lexicographically first injection in ``itertools.permutations`` order.
    """
    R, M = cost.shape
    best_total: float | None = None
    best_perm: tuple[int, ...] | None = None
    for perm in itertools.permutations(range(M), R):
        total = 0.0
        for j, col in enumerate(perm):
            total += cost[j, col]
        if best_total is None or total < best_total:
            best_total = total
            best_perm = perm
    assert best_total is not None and best_perm is not None
    pair_costs = [float(cost[j, best_perm[j]]) for j in range(R)]
    return best_total, best_perm, pair_costs


def search(
    query: Any,
    catalog: Any,
    model: Any = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """One-to-one region retrieval over the compressed catalog.

    ``query`` has ``R`` regions (``1 <= R <= 4``).  ``catalog`` is either the
    artifact returned by :func:`build_catalog` or a plain code mapping plus a
    legal ``model`` dictionary.  ``top_k`` is ``0..8``.

    Returns feasible ``results`` ranked by average one-to-one cost (ties by
    ``image_id``), each with its injection and per-pair costs, and an
    ``insufficient`` list that keeps the reason for every image with fewer
    reconstructed regions than the query.
    """
    if isinstance(top_k, (bool, np.bool_)) or not isinstance(
        top_k, (int, np.integer)
    ):
        raise ValueError(f"top_k must be an integer in [0, {MAX_TOP_K}]")
    top_k = int(top_k)
    if not (0 <= top_k <= MAX_TOP_K):
        raise ValueError(f"top_k must be in [0, {MAX_TOP_K}]")

    cbs, image_codes = _validate_catalog(catalog, model=model)
    L, K, d = cbs.shape

    q = _validate_vectors(
        query,
        d,
        name="query",
        max_abs=MAX_INPUT_VECTOR_ABS,
        min_count=MIN_QUERY_REGIONS,
        max_count=MAX_QUERY_REGIONS,
    )
    R = q.shape[0]

    feasible: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []

    # Decode once per image from the delivered codebook only.
    for image_id in sorted(image_codes):
        codes = image_codes[image_id]
        M = codes.shape[0]
        if M < R:
            insufficient.append(
                {
                    "id": image_id,
                    "status": "insufficient_regions",
                    "reason": "fewer reconstructed regions than query regions",
                    "regions": int(M),
                    "required": int(R),
                }
            )
            continue

        decoded = _reconstruct(codes, cbs)  # (M, d); already validated above
        diff = q[:, None, :] - decoded[None, :, :]
        sq = np.sum(diff * diff, axis=2)  # (R, M)

        total_sq, assignment, pair_costs = _best_injection(sq)
        cost = float(total_sq) / R

        feasible.append(
            {
                "id": image_id,
                "status": "ok",
                "cost": cost,
                "mean_squared_cost": cost,
                "total_squared_cost": float(total_sq),
                "assignment": [
                    {"query_index": j, "region_index": int(assignment[j]),
                     "cost": pair_costs[j]}
                    for j in range(R)
                ],
                "region_indices": [int(v) for v in assignment],
                "pair_costs": pair_costs,
                "regions": int(M),
                "query_regions": int(R),
            }
        )

    feasible.sort(key=lambda item: (item["cost"], item["id"]))
    selected = feasible[:top_k]

    return {
        "format": "additive_visual_search_v1",
        "query_regions": int(R),
        "top_k": int(top_k),
        "feasible_count": len(feasible),
        "insufficient_count": len(insufficient),
        "ranking_basis": "additive dictionary reconstruction",
        "codebook_digest": codebook_digest(cbs),
        "results": selected,
        "all_feasible": feasible,
        "insufficient": insufficient,
    }


def audit_catalog(catalog: Any, model: Any = None) -> dict[str, Any]:
    """Independently re-decode a catalog artifact; no trainer state required.

    Returns the delivered codebook digest, per-image delivered codes, decoded
    region coordinates and basic integrity checks (index ranges and the
    digest recorded inside the artifact).  This is what a consumer can run to
    confirm that the codes decode against the shared dictionary without any
    original library vectors.
    """
    cbs, image_codes = _validate_catalog(catalog, model=model)
    L, K, d = cbs.shape

    recorded = None
    if isinstance(catalog, Mapping) and "codebook_digest" in catalog:
        recorded = catalog["codebook_digest"]
    actual = codebook_digest(cbs)
    digest_ok = recorded is None or recorded == actual

    decoded: dict[str, list[list[float]]] = {}
    for image_id in sorted(image_codes):
        codes = image_codes[image_id]
        rec = _reconstruct(codes, cbs) if codes.shape[0] else np.zeros((0, d))
        decoded[image_id] = [[float(v) for v in row] for row in rec]

    return {
        "format": "additive_visual_catalog_audit_v1",
        "codebook_digest": actual,
        "recorded_digest_matches": bool(digest_ok),
        "layers": int(L),
        "codebook_size": int(K),
        "dim": int(d),
        "codes": {
            image_id: [[int(v) for v in row] for row in image_codes[image_id]]
            for image_id in sorted(image_codes)
        },
        "decoded": decoded,
        "region_counts": {
            image_id: int(image_codes[image_id].shape[0])
            for image_id in sorted(image_codes)
        },
    }
