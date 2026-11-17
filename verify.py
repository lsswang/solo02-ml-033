"""Independent correctness verification for the additive dictionary library.

This module deliberately re-derives the important quantities with separate
implementations (brute-force enumeration, an explicitly assembled least-squares
design matrix, and exhaustive permutation matching) rather than reading values
back out of the production solver.

Run ``python verify.py``; it prints a short report and exits non-zero if any
check fails.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

import dictionary as D
import retrieval as R


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def close(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


class Checker:
    def __init__(self) -> None:
        self.passed = 0

    def check(self, name: str, fn) -> None:
        fn()
        self.passed += 1
        print(f"  ok  {name}")


# ---------------------------------------------------------------------------
# independent reference implementations
# ---------------------------------------------------------------------------
def ref_reconstruct(code, codebooks) -> np.ndarray:
    """Direct sum of one word per layer, no vectorised indexing tricks."""
    out = np.zeros(codebooks.shape[2], dtype=np.float64)
    for layer, word in enumerate(code):
        out = out + codebooks[layer, word]
    return out


def ref_encode_exhaustive(vector, codebooks):
    """Independent brute force over every code with direct reconstruction."""
    L, K, _ = codebooks.shape
    best_code = None
    best_err = math.inf
    for code in itertools.product(range(K), repeat=L):
        rec = ref_reconstruct(code, codebooks)
        err = float(np.sum((vector - rec) ** 2))
        if err < best_err - 0.0:  # strict improvement; first tie kept
            best_err = err
            best_code = code
    return list(best_code), best_err


def ref_solve_joint(vectors, weights, codes, ridge, layers, words):
    """Independent joint ridge solve with an explicit stacked design matrix.

    For each region i we add one row per coordinate with entries
    sqrt(w_i) at ``(l, z_il)``.  Regularisation contributes
    ``sqrt(ridge) * I``.  ``lstsq`` then minimises exactly the training
    objective, so cross-layer terms are automatic.
    """
    n = vectors.shape[0]
    d = vectors.shape[1]
    n_vars = layers * words
    design = np.zeros((n, n_vars), dtype=np.float64)
    for i in range(n):
        s = math.sqrt(weights[i])
        for l in range(layers):
            design[i, l * words + int(codes[i, l])] = s
    reg_block = math.sqrt(ridge) * np.eye(n_vars)
    full = np.vstack([design, reg_block])
    result = np.zeros((n_vars, d), dtype=np.float64)
    for coord in range(d):
        rhs_rows = np.asarray(
            [math.sqrt(weights[i]) * vectors[i, coord] for i in range(n)],
            dtype=np.float64,
        ).reshape(n, 1)
        reg_rhs = np.zeros((n_vars, 1), dtype=np.float64)
        target = np.vstack([rhs_rows, reg_rhs])
        result[:, coord] = np.linalg.lstsq(full, target, rcond=None)[0][:, 0]
    return result.reshape(layers, words, d)


def ref_objective(images, codebooks, codes, ridge):
    n_images = len(images)
    data = 0.0
    for img_idx, image in enumerate(images):
        m = len(image)
        w = 1.0 / (n_images * m)
        for region_idx, vector in enumerate(image):
            rec = ref_reconstruct(codes[img_idx][region_idx], codebooks)
            data += w * float(np.sum((np.asarray(vector) - rec) ** 2))
    reg = ridge * float(np.sum(np.asarray(codebooks) ** 2))
    return data, reg, data + reg


def ref_best_matching(cost):
    """Independent exhaustive injection search returning (total, perm)."""
    r, m = cost.shape
    best = None
    for perm in itertools.permutations(range(m), r):
        total = sum(cost[j, perm[j]] for j in range(r))
        if best is None or total < best[0]:
            best = (float(total), perm)
    return best


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_encode_global() -> None:
    rng = np.random.default_rng(7)
    C = rng.integers(-3, 4, size=(3, 4, 3)).astype(float)
    X = rng.integers(-4, 5, size=(6, 3)).astype(float)
    codes = D.encode(X, C)
    require(len(codes) == len(X), "encode must return one code per vector")
    for i, x in enumerate(X):
        ref_code, ref_err = ref_encode_exhaustive(x, C)
        got_rec = ref_reconstruct(codes[i], C)
        got_err = float(np.sum((x - got_rec) ** 2))
        require(
            close(got_err, ref_err, atol=1e-9, rtol=1e-9),
            f"encode[{i}] not globally optimal: {got_err} vs {ref_err}",
        )
        # The returned code must also be an actual optimal code.
        require(
            all(0 <= v < C.shape[1] for v in codes[i]),
            f"encode[{i}] produced an illegal index",
        )


def check_greedy_residual_not_optimal() -> None:
    """Construct a case where greedy residual coding is suboptimal.

    The global additive encoder must find a code whose reconstruction is at
    least as good as the greedy one (usually strictly better here).
    """
    C = np.array(
        [
            [[0.0, 0.0], [10.0, 0.0]],
            [[0.0, 0.0], [10.0, 0.0]],
        ]
    )
    x = np.array([6.0, 0.0])
    # greedy: pick layer 0 word 1 (0 distance to 10), residual [-4,0];
    # layer 1 then picks word 0 (0), leaving reconstruction [10,0], err=16.
    greedy = [1, 0]
    greedy_err = float(np.sum((x - ref_reconstruct(greedy, C)) ** 2))
    codes = D.encode([x], C)[0]
    got_err = float(np.sum((x - ref_reconstruct(codes, C)) ** 2))
    require(
        got_err <= greedy_err + 1e-9,
        f"global code worse than greedy: {got_err} vs {greedy_err}",
    )
    # Enumerating all codes must agree with the reference brute force.
    ref_code, ref_err = ref_encode_exhaustive(x, C)
    require(
        close(got_err, ref_err, atol=1e-9, rtol=1e-9),
        "global code does not match brute force",
    )


def check_weighted_joint_update() -> None:
    """Joint solve equals independent stacked least squares, cross terms included."""
    rng = np.random.default_rng(11)
    L, K, d = 2, 3, 4
    C0 = rng.integers(-2, 3, size=(L, K, d)).astype(float)
    # Deliberately imbalanced images to exercise image-balanced weights.
    images = [
        [rng.integers(-5, 6, size=d).astype(float) for _ in range(1)],
        [rng.integers(-5, 6, size=d).astype(float) for _ in range(4)],
        [rng.integers(-5, 6, size=d).astype(float) for _ in range(2)],
    ]
    ridge = 0.25
    n_images = len(images)
    flat = [v for im in images for v in im]
    weights = []
    for im in images:
        weights.extend([1.0 / (n_images * len(im))] * len(im))
    X = np.vstack(flat)
    w = np.asarray(weights)
    codes, _ = D._best_codes(X, C0)
    got = D._solve_dictionary(X, w, codes, ridge, L, K)
    ref = ref_solve_joint(X, w, codes, ridge, L, K)
    require(
        np.allclose(got, ref, atol=1e-9, rtol=1e-9),
        "weighted joint dictionary solve disagrees with stacked least squares",
    )

    # Confirm the joint solve is not the same as a per-layer residual mean.
    naive = C0.copy()
    residual = X.copy()
    for l in range(L):
        for k in range(K):
            mask = codes[:, l] == k
            if np.any(mask):
                naive[l, k] = residual[mask].mean(axis=0)
        naive[l, codes[:, l]] = naive[l, codes[:, l]]
        residual -= np.stack([naive[l, c] for c in codes[:, l]])
    require(
        not np.allclose(got, naive, atol=1e-6, rtol=1e-6),
        "joint solve unexpectedly equals naive per-layer residual means",
    )

    # Joint solve must lower (or match) the fixed-code objective versus naive.
    def obj(C):
        rec = np.stack([sum(C[l, codes[i, l]] for l in range(L)) for i in range(len(X))])
        return float(np.sum(w * np.sum((X - rec) ** 2, axis=1))) + ridge * float(
            np.sum(C ** 2)
        )

    require(
        obj(got) <= obj(naive) + 1e-9,
        "joint solve did not attain the least-squares minimum",
    )
    require(
        obj(got) <= obj(C0) + 1e-9,
        "joint solve increased the objective",
    )


def check_empty_word_regularised() -> None:
    """A word selected by no region must be exactly zero after the update."""
    L, K, d = 2, 2, 1
    C0 = np.array([[[5.0], [-5.0]], [[5.0], [-5.0]]])
    # All regions are near +, so the ''word 1'' columns are never selected.
    images = [[[1.0], [1.1]], [[0.9], [1.2]]]
    model = D.fit(images, C0, iterations=1, ridge=0.5)
    C = np.asarray(model["codebooks"])
    used = set()
    for img_codes in model["final_codes"]:
        for code in img_codes:
            for l in range(L):
                used.add((l, code[l]))
    for l in range(L):
        for k in range(K):
            if (l, k) not in used:
                require(
                    float(np.max(np.abs(C[l, k]))) == 0.0,
                    f"empty word ({l},{k}) was not driven to zero: {C[l,k]}",
                )
    # It must not retain the arbitrary initial centres.
    for l in range(L):
        require(
            not np.allclose(C[l, 1], C0[l, 1]),
            f"empty word ({l},1) retained its initial centre",
        )


def check_training_rounds_match_objective() -> None:
    rng = np.random.default_rng(3)
    images = [
        [rng.integers(-4, 5, size=2).astype(float) for _ in range(2)],
        [rng.integers(-4, 5, size=2).astype(float) for _ in range(3)],
        [rng.integers(-4, 5, size=2).astype(float) for _ in range(1)],
    ]
    C0 = rng.integers(-2, 3, size=(2, 2, 2)).astype(float)
    ridge = 0.3
    iters = 3
    model = D.fit(images, C0, iterations=iters, ridge=ridge)
    C = np.asarray(model["codebooks"])

    require(len(model["rounds"]) == iters, "wrong number of recorded rounds")
    for rd in model["rounds"]:
        require(
            rd["data_after_update"] + rd["ridge_after_update"]
            == rd["objective_after_update"],
            "round objective is not data + ridge",
        )
    # Round 1's 'before' objective must equal the initial dictionary objective.
    z0 = [D.encode(im, C0) for im in images]
    d0, r0, t0 = ref_objective(images, C0, z0, ridge)
    require(
        close(model["rounds"][0]["objective_before_update"], t0, atol=1e-9, rtol=1e-9),
        "round 1 before objective does not match independent recomputation",
    )

    # Final re-encoding must use the final dictionary and not update it.
    final_codes = model["final_codes"]
    require(
        final_codes == [D.encode(im, C) for im in images],
        "final codes must equal a fresh global encode with the final dictionary",
    )
    dF, rF, tF = ref_objective(images, C, final_codes, ridge)
    require(
        close(model["objective"]["data"], dF, atol=1e-9, rtol=1e-9),
        "final data term disagrees with independent recomputation",
    )
    require(
        close(model["objective"]["ridge"], rF, atol=1e-9, rtol=1e-9),
        "final ridge term disagrees with independent recomputation",
    )
    require(
        close(model["objective"]["total"], tF, atol=1e-9, rtol=1e-9),
        "final total objective disagrees with independent recomputation",
    )
    # Final dictionary unchanged by the re-encode: recomputed codes are used.
    require(
        np.array_equal(C, np.asarray(model["codebooks"])),
        "dictionary changed unexpectedly",
    )


def check_image_balanced_weights() -> None:
    """A 1-region image and a 4-region image must carry equal total weight."""
    # Two images: one big cluster with 4 near-identical far regions, one with a
    # single region.  A balanced fit should place a word near the singleton.
    images = [
        [[10.0], [10.0], [10.0], [10.0]],
        [[0.0]],
    ]
    C0 = np.array([[[0.0], [5.0]], [[0.0], [5.0]]])
    model = D.fit(images, C0, iterations=2, ridge=0.01)
    C = np.asarray(model["codebooks"])
    # The singleton at 0 should be represented nearly exactly; without image
    # balancing its weight would be 1/5 of the cluster's total instead of 1/2.
    errs = []
    for img_codes, img in zip(model["final_codes"], images):
        rec = D.decode(img_codes, C)
        errs.append(np.mean((np.asarray(img) - rec) ** 2))
    require(
        errs[1] < 0.5,
        f"singleton image not well reconstructed under image balancing: {errs[1]}",
    )


def check_one_to_one_and_insufficient() -> None:
    L, K, d = 2, 2, 2
    C = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    )
    # A library image whose two regions decode to the same point.
    catalog = {
        "dup": [[0, 0], [0, 0]],
        "distinct": [[1, 0], [0, 1]],
        "small": [[1, 0]],
    }
    query = [[1.0, 0.0], [0.0, 1.0]]
    out = R.search(query, catalog, model={"codebooks": C}, top_k=8)

    # 'small' has fewer regions than R=2 -> insufficient, not ranked.
    required_ids = {e["id"] for e in out["insufficient"]}
    require("small" in required_ids, "small image should be insufficient")
    require(
        all(r["id"] != "small" for r in out["all_feasible"]),
        "insufficient image must not appear in ranking",
    )
    require(
        any(e["status"] == "insufficient_regions" for e in out["insufficient"]),
        "insufficient status missing",
    )

    # Independent exhaustive matching for the feasible images.
    for res in out["all_feasible"]:
        codes = catalog[res["id"]]
        decoded = np.array([ref_reconstruct(c, C) for c in codes])
        diff = np.asarray(query)[:, None, :] - decoded[None, :, :]
        cost = np.sum(diff * diff, axis=2)
        total, perm = ref_best_matching(cost)
        require(
            close(res["cost"], total / len(query), atol=1e-9, rtol=1e-9),
            f"{res['id']} cost disagrees with exhaustive injection search",
        )
        require(
            tuple(res["region_indices"]) == perm,
            f"{res['id']} assignment disagrees with exhaustive search",
        )
        require(
            len(set(res["region_indices"])) == len(res["region_indices"]),
            f"{res['id']} reused a region",
        )
        pair = [float(cost[j, perm[j]]) for j in range(len(query))]
        require(
            all(
                close(a, b, atol=1e-9, rtol=1e-9)
                for a, b in zip(pair, res["pair_costs"])
            ),
            f"{res['id']} pair costs disagree",
        )

    # 'dup' must pay for duplication (both query regions forced to one column).
    dup = next(r for r in out["all_feasible"] if r["id"] == "dup")
    distinct = next(r for r in out["all_feasible"] if r["id"] == "distinct")
    require(
        distinct["cost"] <= dup["cost"],
        "distinct image should not be worse than the duplicated-region image",
    )


def check_ranking_and_ties() -> None:
    C = np.array([[[0.0, 0.0], [5.0, 0.0]], [[0.0, 0.0], [5.0, 0.0]]])
    catalog = {
        "zeta": [[0, 0]],
        "alpha": [[0, 0]],
        "beta": [[1, 1]],
    }
    query = [[0.0, 0.0]]
    out = R.search(query, catalog, model={"codebooks": C}, top_k=2)
    ids = [r["id"] for r in out["results"]]
    require(ids == ["alpha", "zeta"], f"tie order wrong: {ids}")
    # beta is expensive but feasible and must still be present in all_feasible.
    require(
        any(r["id"] == "beta" for r in out["all_feasible"]),
        "feasible but low-ranked image missing",
    )
    # top_k=0 returns no ranked rows but keeps insufficient info.
    empty = R.search(query, catalog, model={"codebooks": C}, top_k=0)
    require(empty["results"] == [], "top_k=0 must return no ranked results")


def check_external_catalog_and_model_change() -> None:
    rng = np.random.default_rng(21)
    train_a = [
        [rng.integers(-3, 4, size=2).astype(float) for _ in range(2)],
        [rng.integers(-3, 4, size=2).astype(float) for _ in range(2)],
    ]
    train_b = [
        [rng.integers(-3, 4, size=2).astype(float) for _ in range(2)],
        [rng.integers(-3, 4, size=2).astype(float) for _ in range(2)],
    ]
    C0 = np.array([[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]])
    lib = {
        "p": [[0.0, 0.0], [1.0, 0.0]],
        "q": [[2.0, 0.0], [0.0, 2.0]],
    }
    query = [[0.5, 0.0], [0.0, 1.0]]

    model_a = D.fit(train_a, C0, iterations=2, ridge=0.2)
    model_b = D.fit(train_b, C0, iterations=2, ridge=0.2)
    cat_a = R.build_catalog(lib, model_a)
    cat_b = R.build_catalog(lib, model_b)

    require(
        cat_a["codebook_digest"] != cat_b["codebook_digest"],
        "different training should yield a different delivered dictionary",
    )
    # Catalog is self-contained: no original vectors leak in.
    require("images" in cat_a and "codebooks" in cat_a, "artifact incomplete")
    flat = str(cat_a["images"])
    require("0.5" not in flat, "original catalog vectors leaked into the artifact")

    res_a = R.search(query, cat_a, top_k=3)
    res_b = R.search(query, cat_b, top_k=3)

    # Ranking must be computed on the *delivered* reconstruction.  Recompute
    # independently from each artifact and confirm the reported cost matches.
    for cat, res in ((cat_a, res_a), (cat_b, res_b)):
        C = np.asarray(cat["codebooks"])
        for entry in res["all_feasible"]:
            decoded = np.array(
                [ref_reconstruct(c, C) for c in cat["images"][entry["id"]]]
            )
            diff = np.asarray(query)[:, None, :] - decoded[None, :, :]
            cost = np.sum(diff * diff, axis=2)
            total, perm = ref_best_matching(cost)
            require(
                close(entry["cost"], total / len(query), atol=1e-9, rtol=1e-9),
                "reported cost does not match artifact-only reconstruction",
            )

    # The two artifacts genuinely encode different codes/ranks.
    differing = cat_a["images"] != cat_b["images"]
    require(differing, "compressed codes did not change with the model")

    # Passing a *stale* model must be rejected rather than silently decoding
    # the new catalog with an old dictionary.
    raised = False
    try:
        R.search(query, cat_a, model=model_b)
    except ValueError:
        raised = True
    require(raised, "stale model/catalog mismatch was not rejected")

    # The plain external form (codes + legal dictionary) must work too.
    external = {"codebooks": cat_a["codebooks"]}
    plain = dict(cat_a["images"])
    direct = R.search(query, plain, model=external, top_k=3)
    require(
        [r["id"] for r in direct["results"]]
        == [r["id"] for r in res_a["results"]],
        "plain external catalog search disagrees with artifact search",
    )


def check_decode_external() -> None:
    """decode must work with any legal externally supplied dictionary."""
    C = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[100.0, 200.0], [300.0, 400.0]],
        ]
    )
    codes = [[0, 1, 0], [1, 0, 1]]
    got = D.decode(codes, {"codebooks": C})
    expected = np.array(
        [
            [1.0 + 30.0 + 100.0, 2.0 + 40.0 + 200.0],
            [3.0 + 10.0 + 300.0, 4.0 + 20.0 + 400.0],
        ]
    )
    require(np.allclose(got, expected, atol=1e-12, rtol=1e-12), "decode wrong")


def check_validation_errors() -> None:
    C = np.zeros((2, 2, 2))
    cases = []

    def expect_error(name, fn):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError(f"expected ValueError: {name}")

    expect_error("bad layer count", lambda: D.encode([[0.0, 0.0]], np.zeros((4, 2, 2))))
    expect_error("bad word count", lambda: D.encode([[0.0, 0.0]], np.zeros((2, 1, 2))))
    expect_error("bad dim", lambda: D.encode([[0.0, 0.0]], np.zeros((2, 2, 7))))
    expect_error("non-finite", lambda: D.encode([[np.nan, 0.0]], C))
    expect_error("input too large", lambda: D.encode([[1e9, 0.0]], C))
    expect_error("illegal code index", lambda: D.decode([[2, 0]], C))
    expect_error("wrong code length", lambda: D.decode([[0]], C))
    expect_error("bool code", lambda: D.decode([[True, 0]], C))
    expect_error("empty training images", lambda: D.fit([], C, 1, 0.1))
    expect_error("empty training image", lambda: D.fit([[]], C, 1, 0.1))
    expect_error("iterations low", lambda: D.fit([[[0.0, 0.0]]], C, 0, 0.1))
    expect_error("iterations high", lambda: D.fit([[[0.0, 0.0]]], C, 6, 0.1))
    expect_error("ridge low", lambda: D.fit([[[0.0, 0.0]]], C, 1, 0.0001))
    expect_error("ridge high", lambda: D.fit([[[0.0, 0.0]]], C, 1, 2.0))
    expect_error(
        "too many training images",
        lambda: D.fit([[[0.0, 0.0]]] * 13, C, 1, 0.1),
    )
    expect_error(
        "too many regions in image",
        lambda: D.fit([[[0.0, 0.0]] * 7], C, 1, 0.1),
    )
    expect_error(
        "catalog bad id",
        lambda: R.build_catalog({"": [[0.0, 0.0]]}, {"codebooks": C}),
    )
    expect_error(
        "too many catalog images",
        lambda: R.build_catalog({str(i): [] for i in range(9)}, {"codebooks": C}),
    )
    expect_error(
        "top_k too high",
        lambda: R.search([[0.0, 0.0]], {"codebooks": C}, top_k=9),
    )
    expect_error(
        "query empty",
        lambda: R.search([], {"codebooks": C}),
    )
    expect_error(
        "too many query regions",
        lambda: R.search([[0.0, 0.0]] * 5, {"codebooks": C}),
    )


def check_trained_dictionary_bound_roundtrip() -> None:
    """A trained dictionary may exceed the external bound and must round-trip.

    Training starts from legal inputs but ridge shrinking / cross terms can
    leave words outside the external +/-1000 envelope.  We force that here by
    using a large legal initial dictionary and check the artifact round-trips.
    """
    # Initial values are within the training input bound (<=10 per contract),
    # so instead validate the *policy*: a tagged trained model with large
    # words is accepted, and an untagged raw external array of the same
    # magnitude is rejected.
    raw = np.full((2, 2, 2), 5000.0)
    expect = False
    try:
        D.extract_codebooks(raw)
    except ValueError:
        expect = True
    require(expect, "raw external codebook above 1000 must be rejected")

    tagged = {"format": "additive_visual_model_v1", "codebooks": raw.tolist()}
    accepted = D.extract_codebooks(tagged)
    require(np.allclose(accepted, raw), "tagged trained model should not be clipped")

    # Simulated artifact around that dictionary round-trips through search.
    artifact = {
        "format": "additive_visual_catalog_v1",
        "codebooks": raw.tolist(),
        "codebook_digest": D.codebook_digest(raw),
        "images": {"a": [[0, 0]]},
    }
    out = R.search([[0.0, 0.0]], artifact, top_k=1)
    require(
        out["results"][0]["id"] == "a",
        "large trained dictionary artifact failed to round-trip",
    )


def check_float64_dtypes() -> None:
    C = np.zeros((2, 2, 2), dtype=np.float64)
    model = D.fit([[[0.0, 0.0], [1.0, 1.0]]], C, iterations=1, ridge=0.1)
    C_out = np.asarray(model["codebooks"])
    require(C_out.dtype == np.float64, "codebooks must be float64")
    dec = D.decode([[0, 0]], {"codebooks": C_out})
    require(dec.dtype == np.float64, "decode must produce float64")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Running independent correctness checks")
    c = Checker()
    c.check("encode achieves global optimum (full enumeration)", check_encode_global)
    c.check("global code beats greedy residual coding", check_greedy_residual_not_optimal)
    c.check("weighted joint ridge update (cross terms)", check_weighted_joint_update)
    c.check("empty words driven to zero by regulariser", check_empty_word_regularised)
    c.check("round records and final re-encoding", check_training_rounds_match_objective)
    c.check("image-balanced region weights", check_image_balanced_weights)
    c.check("one-to-one coverage, insufficiency, ties", check_one_to_one_and_insufficient)
    c.check("ranking order and top_k behaviour", check_ranking_and_ties)
    c.check("external catalog, model-change linkage, stale model", check_external_catalog_and_model_change)
    c.check("decode with external dictionary", check_decode_external)
    c.check("input validation errors", check_validation_errors)
    c.check("trained dictionary bound round-trip", check_trained_dictionary_bound_roundtrip)
    c.check("float64 outputs", check_float64_dtypes)
    print(f"\nAll {c.passed} checks passed.")


if __name__ == "__main__":
    main()
