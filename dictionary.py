"""Additive visual dictionary learning.

A region ``x`` is compressed to a code ``z = (z_1, ..., z_L)`` and is
reconstructed from the *sum* of one word per layer::

    x_hat(z) = sum_l C[l, z_l]

Layers are additive, not concatenated and not disjoint coordinate subspaces.
Encoding enumerates all ``K ** L`` codes and returns a code with globally
minimal squared reconstruction error (ties may return any optimal code).

Training performs exactly ``iterations`` alternating rounds:

1. with the current ``C``, choose the globally best code for every training
   region (full ``K ** L`` enumeration);
2. with those codes fixed, solve the *joint* weighted ridge least-squares
   problem for the whole dictionary.  The solve is done over all ``L * K``
   words at once, so the cross terms between layers that co-explain a region
   are respected.  ``lambda > 0`` makes the solution unique; unused words are
   pulled to zero by the regulariser instead of keeping an old centre.

The objective is::

    J(C, z) = sum_regions w_i ||x_i - x_hat(z_i)||^2
              + lambda * sum_{l,k} ||C[l,k]||^2

with ``w_i = 1 / (n_images * n_regions_in_i's_image)``, so every training
image carries the same total weight.

All numerical state is float64.  Public functions validate their inputs and
raise ``ValueError`` with a specific message for malformed shapes, illegal
code indices, non-finite values, empty training images or out-of-range
parameters.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Any, Mapping

import numpy as np

# Input contract bounds -----------------------------------------------------
MAX_INPUT_VECTOR_ABS = 10.0      # training / initial / catalog / query vectors
MAX_EXTERNAL_CODEBOOK_ABS = 1000.0
MIN_LAYERS, MAX_LAYERS = 1, 3
MIN_WORDS, MAX_WORDS = 2, 4
MIN_DIM, MAX_DIM = 1, 6
MIN_ITERATIONS, MAX_ITERATIONS = 1, 5
MIN_RIDGE, MAX_RIDGE = 0.001, 1.0
MAX_TRAIN_IMAGES, MAX_TRAIN_REGIONS = 12, 6
MAX_CATALOG_IMAGES, MAX_CATALOG_REGIONS = 8, 6
MIN_QUERY_REGIONS, MAX_QUERY_REGIONS = 1, 4
MAX_TOP_K = 8

_CODE_CACHE: dict[tuple[int, int], np.ndarray] = {}


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _finite_abs_check(arr: np.ndarray, *, name: str, max_abs: float) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if arr.size and float(np.max(np.abs(arr))) > max_abs:
        raise ValueError(
            f"{name} entries must have absolute value <= {max_abs:g}"
        )


def _as_float_matrix(value: Any, *, name: str) -> np.ndarray:
    """Convert a sequence of vectors to a 2-D float64 array.

    A bare empty sequence is represented as shape ``(0, d)`` by callers that
    already know ``d``; here a non-empty 1-D input is rejected because region
    vectors have to be supplied as a sequence of rows.
    """
    if value is None:
        raise ValueError(f"{name} must not be None")
    try:
        seq = list(value)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a sequence of region vectors") from exc

    if len(seq) == 0:
        raise ValueError(f"{name} must not be empty")

    rows: list[np.ndarray] = []
    for i, row in enumerate(seq):
        try:
            arr = np.asarray(row, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{i}] must be a numeric vector") from exc
        if arr.ndim != 1:
            raise ValueError(f"{name}[{i}] must be a 1-D vector")
        if arr.size == 0:
            raise ValueError(f"{name}[{i}] must not be empty")
        rows.append(arr)
    shape = rows[0].shape
    for i, arr in enumerate(rows):
        if arr.shape != shape:
            raise ValueError(f"{name}[{i}] has shape {arr.shape}, expected {shape}")
    return np.vstack(rows)


def _validate_vectors(
    vectors: Any,
    dim: int,
    *,
    name: str,
    max_abs: float = MAX_INPUT_VECTOR_ABS,
    min_count: int = 0,
    max_count: int | None = None,
) -> np.ndarray:
    """Validate a sequence of ``dim``-dimensional region vectors.

    Empty sequences are accepted only when ``min_count == 0``.
    """
    try:
        seq = list(vectors) if not isinstance(vectors, np.ndarray) else list(vectors)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of region vectors") from exc

    if len(seq) == 0:
        if min_count > 0:
            raise ValueError(f"{name} must contain at least {min_count} vector(s)")
        return np.zeros((0, dim), dtype=np.float64)

    arr = _as_float_matrix(seq, name=name)
    if arr.shape[1] != dim:
        raise ValueError(
            f"{name} vectors have dimension {arr.shape[1]}, expected {dim}"
        )
    if min_count and arr.shape[0] < min_count:
        raise ValueError(f"{name} must contain at least {min_count} vector(s)")
    if max_count is not None and arr.shape[0] > max_count:
        raise ValueError(f"{name} must contain at most {max_count} vector(s)")
    _finite_abs_check(arr, name=name, max_abs=max_abs)
    return arr


def _validate_codebooks(
    codebooks: Any,
    *,
    name: str = "codebooks",
    max_abs: float = MAX_EXTERNAL_CODEBOOK_ABS,
) -> np.ndarray:
    """Validate and return an ``(L, K, d)`` float64 dictionary."""
    try:
        arr = np.asarray(codebooks, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric (L, K, d) array") from exc
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3-D (L, K, d) array")
    L, K, d = arr.shape
    if not (MIN_LAYERS <= L <= MAX_LAYERS):
        raise ValueError(f"{name} layer count must be in [{MIN_LAYERS}, {MAX_LAYERS}]")
    if not (MIN_WORDS <= K <= MAX_WORDS):
        raise ValueError(f"{name} codebook size must be in [{MIN_WORDS}, {MAX_WORDS}]")
    if not (MIN_DIM <= d <= MAX_DIM):
        raise ValueError(f"{name} dimension must be in [{MIN_DIM}, {MAX_DIM}]")
    _finite_abs_check(arr, name=name, max_abs=max_abs)
    return np.ascontiguousarray(arr, dtype=np.float64)


def extract_codebooks(
    model: Any,
    *,
    name: str = "codebooks",
    max_abs: float | None = None,
) -> np.ndarray:
    """Extract and validate a dictionary from a model dict, artifact or array.

    Dictionaries produced by :func:`fit` (tagged ``additive_visual_*``) are not
    clipped or magnitude-limited, because training updates may legally push
    words past the external-input bound.  Raw external arrays and untagged
    mappings are held to ``MAX_EXTERNAL_CODEBOOK_ABS``.
    """
    tagged = False
    if isinstance(model, Mapping):
        if "codebooks" not in model:
            raise ValueError(f"{name} mapping must contain a 'codebooks' key")
        value = model["codebooks"]
        tagged = str(model.get("format", "")).startswith("additive_visual_")
    else:
        value = model
    if max_abs is None:
        max_abs = float("inf") if tagged else MAX_EXTERNAL_CODEBOOK_ABS
    return _validate_codebooks(value, name=name, max_abs=max_abs)


def _validate_codes(
    codes: Any,
    layers: int,
    words: int,
    *,
    name: str = "codes",
) -> np.ndarray:
    """Validate region codes, returning an ``(M, L)`` int64 array."""
    if codes is None:
        raise ValueError(f"{name} must not be None")
    try:
        seq = list(codes)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of codes") from exc
    if len(seq) == 0:
        return np.zeros((0, layers), dtype=np.int64)

    rows: list[list[int]] = []
    for i, code in enumerate(seq):
        if isinstance(code, (str, bytes)):
            raise ValueError(f"{name}[{i}] must be a sequence of integers")
        try:
            values = list(code)
        except TypeError as exc:
            raise ValueError(f"{name}[{i}] must be a sequence of integers") from exc
        if len(values) != layers:
            raise ValueError(
                f"{name}[{i}] must contain exactly {layers} layer index/indices"
            )
        row: list[int] = []
        for l, value in enumerate(values):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name}[{i}][{l}] must be an integer, not bool")
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name}[{i}][{l}] must be an integer")
            iv = int(value)
            if not (0 <= iv < words):
                raise ValueError(
                    f"{name}[{i}][{l}]={iv} is outside [0, {words - 1}]"
                )
            row.append(iv)
        rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def _validate_int(value: Any, *, name: str, lo: int, hi: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
    iv = int(value)
    if not (lo <= iv <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return iv


def _validate_ridge(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("ridge (lambda) must be a real number")
    rv = float(value)
    if not np.isfinite(rv) or not (MIN_RIDGE <= rv <= MAX_RIDGE):
        raise ValueError(f"ridge (lambda) must be finite and in [{MIN_RIDGE}, {MAX_RIDGE}]")
    return rv


# ---------------------------------------------------------------------------
# code enumeration / reconstruction
# ---------------------------------------------------------------------------
def all_codes(layers: int, words: int) -> np.ndarray:
    """Return the lexicographically ordered list of all ``K ** L`` codes."""
    key = (layers, words)
    cached = _CODE_CACHE.get(key)
    if cached is None:
        cached = np.asarray(
            list(itertools.product(range(words), repeat=layers)), dtype=np.int64
        )
        _CODE_CACHE[key] = cached
    return cached


def _reconstruct(codes: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Reconstruct rows of codes.  ``codes`` is (N, L), result is (N, d)."""
    codes = np.asarray(codes, dtype=np.int64)
    if codes.size == 0:
        return np.zeros((0, codebooks.shape[2]), dtype=np.float64)
    layer_index = np.arange(codebooks.shape[0], dtype=np.int64)[None, :]
    return np.sum(codebooks[layer_index, codes, :], axis=1)


def _best_codes(vectors: np.ndarray, codebooks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustive global code search for validated arrays.

    Returns ``(codes, squared_errors)`` where ``codes`` is ``(N, L)``.

    Squared errors are computed directly as ``||x - sum_l C[l, z_l]||^2``
    (not by an expanded inner-product shortcut), so the quantity compared
    against independent brute force is exactly the reconstruction error.
    Ties resolve to the first lexicographic code, which is an optimal code.
    """
    L, K, _ = codebooks.shape
    if vectors.shape[0] == 0:
        return np.zeros((0, L), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    codes = all_codes(L, K)
    recon = _reconstruct(codes, codebooks)                 # (C, d)
    diff = vectors[:, None, :] - recon[None, :, :]          # (N, C, d)
    dist = np.sum(diff * diff, axis=2)                      # (N, C)
    best = np.argmin(dist, axis=1)
    chosen = codes[best]
    err = dist[np.arange(vectors.shape[0]), best]
    return chosen, err


def encode(vectors: Any, codebooks: Any) -> list[list[int]]:
    """Globally optimal additive encoding of region vectors.

    Every region is assigned the code minimising
    ``||x - sum_l C[l, z_l]||^2`` over **all** ``K ** L`` codes.  This is not
    greedy residual quantisation.
    """
    C = extract_codebooks(codebooks)
    X = _validate_vectors(
        vectors, C.shape[2], name="vectors", min_count=0,
        max_abs=MAX_INPUT_VECTOR_ABS,
    )
    codes, _ = _best_codes(X, C)
    return [[int(v) for v in row] for row in codes]


def decode(codes: Any, codebooks: Any) -> np.ndarray:
    """Reconstruct regions from codes and the supplied dictionary.

    ``decode`` is stateless: it works with any legal external dictionary and
    does not consult training state.
    """
    C = extract_codebooks(codebooks)
    L, K, d = C.shape
    z = _validate_codes(codes, L, K, name="codes")
    return _reconstruct(z, C)


# ---------------------------------------------------------------------------
# weighted joint ridge dictionary update
# ---------------------------------------------------------------------------
def _solve_dictionary(
    vectors: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    ridge: float,
    layers: int,
    words: int,
) -> np.ndarray:
    """Global minimiser of ``J`` with codes fixed.

    Minising

        sum_i w_i ||x_i - sum_l C[l, z_il]||^2
        + ridge * sum_{l,k} ||C[l,k]||^2

    gives, for each word ``(l, k)``::

        (sum_{i: z_il=k} w_i + ridge) C[l,k]
        + sum_{l' != l} sum_{i: z_il=k} w_i C[l', z_il']
        = sum_{i: z_il=k} w_i x_i

    The same ``(L*K) x (L*K)`` Gram matrix applies to every coordinate, so we
    build it once and solve for all ``d`` right-hand sides together.  A row
    with no selected regions has RHS zero and diagonal ``ridge``, so the
    corresponding (empty) word is exactly zero.

    This is the joint solve; it is *not* a per-layer mean of residuals.
    """
    n_vars = layers * words
    A = np.zeros((n_vars, n_vars), dtype=np.float64)
    B = np.zeros((n_vars, vectors.shape[1]), dtype=np.float64)

    for i in range(vectors.shape[0]):
        wi = float(weights[i])
        selected = [l * words + int(codes[i, l]) for l in range(layers)]
        rhs = wi * vectors[i]
        for a in selected:
            B[a] += rhs
            for b in selected:
                A[a, b] += wi
    A[np.diag_indices(n_vars)] += ridge

    solution = np.linalg.solve(A, B)
    return solution.reshape(layers, words, vectors.shape[1])


def _objective_terms(
    vectors: np.ndarray,
    weights: np.ndarray,
    codes: np.ndarray,
    codebooks: np.ndarray,
    ridge: float,
) -> tuple[float, float, float, np.ndarray]:
    """Return ``(data, ridge_term, total, squared_errors)``."""
    recon = _reconstruct(codes, codebooks)
    sq = np.sum((vectors - recon) ** 2, axis=1)
    data = float(np.sum(weights * sq))
    reg = float(ridge * np.sum(codebooks * codebooks))
    return data, reg, data + reg, sq


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _validate_training_images(images: Any, dim: int) -> list[np.ndarray]:
    if isinstance(images, np.ndarray):
        if images.ndim != 1:
            raise ValueError(
                "training_images must be a sequence (one entry per image) of region vectors"
            )
        image_list = list(images)
    elif isinstance(images, (list, tuple)):
        image_list = list(images)
    else:
        raise ValueError("training_images must be a sequence of per-image region lists")

    if not (1 <= len(image_list) <= MAX_TRAIN_IMAGES):
        raise ValueError(
            f"training_images must contain between 1 and {MAX_TRAIN_IMAGES} images"
        )

    validated: list[np.ndarray] = []
    for i, image in enumerate(image_list):
        arr = _validate_vectors(
            image,
            dim,
            name=f"training_images[{i}]",
            min_count=1,
            max_count=MAX_TRAIN_REGIONS,
            max_abs=MAX_INPUT_VECTOR_ABS,
        )
        validated.append(arr)
    return validated


def fit(
    images: Any,
    initial: Any,
    iterations: int = 2,
    ridge: float = 0.1,
) -> dict[str, Any]:
    """Alternating additive dictionary learning.

    Runs exactly ``iterations`` rounds of (global code assignment, then joint
    weighted ridge dictionary solve).  Afterwards the final dictionary is used
    once more to encode the training regions.  That final re-encoding does not
    perform another dictionary update.
    """
    iterations = _validate_int(
        iterations, name="iterations", lo=MIN_ITERATIONS, hi=MAX_ITERATIONS
    )
    ridge = _validate_ridge(ridge)

    C = _validate_codebooks(
        initial, name="initial", max_abs=MAX_INPUT_VECTOR_ABS
    ).copy()
    L, K, d = C.shape
    image_arrays = _validate_training_images(images, d)
    n_images = len(image_arrays)

    # Region-level training data and image-balanced weights.
    training_vectors: list[np.ndarray] = []
    region_weights: list[float] = []
    region_image: list[int] = []
    for image_index, arr in enumerate(image_arrays):
        m = arr.shape[0]
        # 1 / (number of images * number of regions in this image)
        w = 1.0 / (n_images * m)
        for region_index in range(m):
            training_vectors.append(arr[region_index])
            region_weights.append(w)
            region_image.append(image_index)
    X = np.vstack(training_vectors)
    w = np.asarray(region_weights, dtype=np.float64)

    rounds: list[dict[str, Any]] = []
    for round_index in range(iterations):
        # 1) globally best codes for the current dictionary
        z, _ = _best_codes(X, C)

        # objective with the current dictionary and these codes
        data_before, reg_before, obj_before, _ = _objective_terms(X, w, z, C, ridge)

        # 2) joint weighted ridge solve with those codes fixed
        C = _solve_dictionary(X, w, z, ridge, L, K)

        data_after, reg_after, obj_after, _ = _objective_terms(X, w, z, C, ridge)
        rounds.append(
            {
                "round": round_index + 1,
                "data_before_update": data_before,
                "ridge_before_update": reg_before,
                "objective_before_update": obj_before,
                "data_after_update": data_after,
                "ridge_after_update": reg_after,
                "objective_after_update": obj_after,
            }
        )

    # Final re-encoding with the final dictionary.  No further update.
    final_codes_arr, _ = _best_codes(X, C)
    data_final, reg_final, total_final, sq_final = _objective_terms(
        X, w, final_codes_arr, C, ridge
    )

    # Per-image report and per-image final codes.
    image_errors: list[dict[str, Any]] = []
    final_codes: list[list[list[int]]] = []
    start = 0
    for image_index, arr in enumerate(image_arrays):
        m = arr.shape[0]
        stop = start + m
        w_i = 1.0 / (n_images * m)
        weighted = float(np.sum(w[start:stop] * sq_final[start:stop]))
        mean_sq = float(np.mean(sq_final[start:stop]))
        codes_i = [[int(v) for v in row] for row in final_codes_arr[start:stop]]
        final_codes.append(codes_i)
        image_errors.append(
            {
                "image_index": image_index,
                "regions": int(m),
                "weight_per_region": float(w_i),
                "weighted_squared_error": weighted,
                "mean_squared_error": mean_sq,
                "codes": codes_i,
            }
        )
        start = stop

    return {
        "format": "additive_visual_model_v1",
        "codebooks": C.tolist(),
        "layers": int(L),
        "codebook_size": int(K),
        "dim": int(d),
        "iterations": int(iterations),
        "ridge": float(ridge),
        "rounds": rounds,
        "final_codes": final_codes,
        "image_errors": image_errors,
        "training_image_count": int(n_images),
        "training_region_counts": [int(arr.shape[0]) for arr in image_arrays],
        "objective": {
            "total": float(total_final),
            "data": float(data_final),
            "ridge": float(reg_final),
            "lambda": float(ridge),
        },
        "code_count": int(K ** L),
    }


# ---------------------------------------------------------------------------
# small public utilities
# ---------------------------------------------------------------------------
def codebook_digest(codebooks: Any) -> str:
    """Stable SHA-256 of the float64 dictionary, used for linking checks.

    Magnitude is not constrained here: the digest is used to compare a
    delivered/trained dictionary with itself, and trained words may legally
    exceed the external input envelope.
    """
    C = extract_codebooks(codebooks, max_abs=float("inf"))
    contiguous = np.ascontiguousarray(C, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def training_objective(
    images: Any,
    codebooks: Any,
    codes: Any,
    ridge: float,
) -> dict[str, float]:
    """Recompute the weighted training objective for supplied codes.

    ``codes`` must be a per-image sequence of per-region codes.  This helper
    makes the artifact independently checkable without trusting trainer state.
    """
    C = extract_codebooks(codebooks)
    L, K, d = C.shape
    ridge = _validate_ridge(ridge)
    image_arrays = _validate_training_images(images, d)
    n_images = len(image_arrays)

    if not isinstance(codes, (list, tuple)) or len(codes) != n_images:
        raise ValueError("codes must provide one code list per training image")

    vectors: list[np.ndarray] = []
    weights: list[float] = []
    flat_codes: list[list[int]] = []
    for image_index, arr in enumerate(image_arrays):
        image_codes = _validate_codes(
            codes[image_index], L, K, name=f"codes[{image_index}]"
        )
        if image_codes.shape[0] != arr.shape[0]:
            raise ValueError(
                f"codes[{image_index}] has {image_codes.shape[0]} regions, "
                f"expected {arr.shape[0]}"
            )
        w = 1.0 / (n_images * arr.shape[0])
        for region_index in range(arr.shape[0]):
            vectors.append(arr[region_index])
            weights.append(w)
            flat_codes.append([int(v) for v in image_codes[region_index]])

    X = np.vstack(vectors)
    w = np.asarray(weights, dtype=np.float64)
    z = np.asarray(flat_codes, dtype=np.int64)
    data, reg, total, _ = _objective_terms(X, w, z, C, ridge)
    return {"data": data, "ridge": reg, "total": total, "lambda": ridge}
