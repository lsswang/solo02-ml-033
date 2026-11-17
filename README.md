# Compressed local visual representations

Additive multi-layer visual dictionary learning and one-to-one region
retrieval. Python 3.11 + NumPy (SciPy available but not required at runtime).
No image model is downloaded; inputs are small already-extracted local
visual vectors.

Run:

```bash
python demo.py              # train, encode catalog, query; writes artifacts/
python verify.py            # independent correctness suite (13 checks)
python inspect_artifact.py artifacts/catalog.json   # artifact-only re-decode
```

Demo output is a single JSON object; the same content is split into
`artifacts/model.json`, `artifacts/catalog.json`, `artifacts/results.json`,
`artifacts/model_change.json` and `artifacts/single_region_query.json`.

---

## 1. Additive dictionary

A dictionary `C` has shape `(L, K, d)` — `L` layers, `K` words per layer.
A region `x` has code `z = (z_1, …, z_L)` and is reconstructed as the **sum**
of one word per layer:

```
x_hat(z) = Σ_l C[l, z_l]
```

Layers are additive. They are neither concatenated nor disjoint coordinate
subspaces: every layer contributes in the full `d`-dimensional space, and two
layers can co-explain the same region.

### Encoding (global optimum)

`encode` enumerates all `K**L` codes (at most `4**3 = 64`) and returns a code
with globally minimal `||x − x_hat(z)||²`. It does **not** run per-layer
nearest-neighbour then subtract the residual — that greedy construction is
generally suboptimal, and `verify.py` includes a case that demonstrates it.
The returned error is computed directly as `||x − Σ_l C[l,z_l]||²`, so it is
the same quantity an independent brute force checks. Exact ties may return any
optimal code; we return the lexicographically first.

### Decoding (stateless)

`decode(codes, codebooks)` only sums the words named by the supplied code under
the supplied dictionary. It accepts any legal external `(L,K,d)` dictionary
(entries `|·| ≤ 1000`) and does not consult training state. Trained
dictionaries are tagged (`format: additive_visual_model_v1`) and are not
magnitude-limited, since a training update may legally push words outside the
external input envelope.

---

## 2. Training

Training data: `1..12` images, each `1..6` regions (empty images are an
error). Image-balanced weights:

```
w_i = 1 / (n_images * regions_in_image_i)
```

so every image has total weight `1 / n_images`. Objective:

```
J(C, z) = Σ_regions w_i ||x_i − Σ_l C[l, z_il]||²
          + λ Σ_{l,k} ||C[l,k]||²
```

with `λ ∈ [0.001, 1]`.

`fit` runs **exactly `iterations` rounds** (1..5) of alternating:

1. **Global code assignment.** For the current `C`, every training region gets
   its globally best code by full `K**L` enumeration.
2. **Joint weighted ridge dictionary update.** With codes fixed, solve for the
   whole dictionary at once.

### Why the update is joint

With codes fixed, minimising `J` over `C` gives, for each word `(l,k)`:

```
(Σ_{i: z_il = k} w_i + λ) C[l,k]
  + Σ_{l' ≠ l} Σ_{i: z_il = k} w_i C[l', z_il']
  = Σ_{i: z_il = k} w_i x_i
```

The cross terms between layers that share a region are part of the normal
equations. We assemble the shared `(L·K) × (L·K)` Gram matrix once and solve
all `d` right-hand sides with one `np.linalg.solve`. This is *not* a per-layer
mean of residuals; `verify.py` compares it against an independently assembled
stacked least-squares design matrix (`np.linalg.lstsq`) and checks it is not
equal to the naive per-layer residual mean.

Because `λ > 0`, the system is strictly positive definite, so the update is
unique. A word selected by no region has zero right-hand side and diagonal
`λ`, so it is driven **exactly to zero** — it does not keep an arbitrary old
centre. The objective with the new `C` is at most the objective before the
update (up to floating point).

### Round records and final re-encoding

Each round records `objective_before_update` (data + ridge with the current
dictionary) and `objective_after_update` (after the joint solve). After the
last round the final dictionary is used **once more** to globally encode all
training regions; this final re-encoding performs **no further dictionary
update**. Outputs:

- `codebooks` — final dictionary;
- `rounds` — per-round data / ridge / total objective (before and after);
- `final_codes` — final per-image codes;
- `image_errors` — per-image region count, weight, weighted and mean squared
  error, and final codes;
- `objective` — complete final `J` split into `data`, `ridge`, `total`, `λ`.

Since the joint problem is non-convex, no claim is made that different legal
initialisations converge to the same model.

---

## 3. Catalog and one-to-one retrieval

`build_catalog(images, model)` encodes external library regions with the
**actual final dictionary** and returns a self-contained artifact:

```json
{
  "format": "additive_visual_catalog_v1",
  "codebooks": [[[...]]],            // full shared dictionary
  "codebook_digest": "sha256…",
  "layers": L, "codebook_size": K, "dim": d,
  "images": {"image_id": [[z1, …, zL], …]},
  "region_counts": {"image_id": M}
}
```

It contains only the shared dictionary and code tuples — no original library
vectors, training data or image files. An external caller may also pass a
plain `{image_id: codes}` mapping together with a legal dictionary/model
instead of calling the trainer.

`search(query, catalog, model=None, top_k=3)` receives only this compressed
product and the query regions. It decodes each image's regions from the
delivered dictionary (`Σ_l C[l,z_l]`), never from original vectors.

For a query with `R` regions and an image with `M` reconstructed regions:

- **`M < R`** → insufficient cover. The image is reported with
  `status: "insufficient_regions"`, `reason`, `regions` and `required`, and is
  **excluded from the ranking**.
- **`M ≥ R`** → minimise over **injections** `π: {1..R} → {1..M}`:

  ```
  D = min_π ( Σ_j ||q_j − decoded[π(j)]||² ) / R
  ```

  The minimum is over injective assignments, so one reconstructed region cannot
  serve two query regions. With `R ≤ 4`, `M ≤ 6` the exhaustive search is at
  most `6P4 = 360` ordered selections, so the result is the global optimum. We
  do not take each row's nearest column independently.

Results are ranked by ascending `D`; exact ties break on `image_id` Unicode
code point order. At most `top_k ∈ [0,8]` feasible images are returned in
`results`. The full feasible ranking is in `all_feasible`, and every
insufficient image remains in `insufficient`, so the reason is never lost.

Each result contains the image cost, the actual `region_indices` injection,
and per-pair costs (`assignment` / `pair_costs`). The ranking is defined on the
delivered dictionary reconstruction; it is not claimed to equal an
uncompressed-feature ranking, and quantisation error is not hidden in a
numerical tolerance.

---

## 4. Model-change linkage

`demo.py` and `verify.py` both fit a second legal model from different training
images and a different initial dictionary while keeping the query, the raw
external library (same `image_id`s and original regions) and `top_k` fixed. The
new dictionary genuinely re-encodes the library: the catalog digest, the code
tuples and the ranking all change, and search recomputes costs from the new
decoded features. Passing a stale model whose dictionary differs from the
catalog's is rejected rather than silently decoding the new catalog with an old
dictionary (`verify.py`, "stale model" check).

---

## 5. Independent verification

`verify.py` re-derives the key quantities with separate implementations:

| check | independent method |
|---|---|
| global code optimality | brute force over all `K**L` codes with direct summation |
| greedy suboptimality | a hand-built case where residual greedy is worse |
| weighted joint update | explicitly stacked `sqrt(w)`/`sqrt(λ)` design matrix via `lstsq`, plus a check that it differs from naive residual means |
| empty-word regularisation | word used by no region is exactly 0 and not the old centre |
| rounds / final re-encode | recompute every round's objective and confirm final codes equal a fresh encode with the final `C` |
| image balancing | singleton image reconstructed well despite a large image |
| one-to-one / insufficiency | exhaustive permutation matching, duplicate-column penalty, no reuse |
| ranking / ties | code-point tie order and `top_k` behaviour |
| external artifact | recompute costs from artifact-only decoding, reject stale model, plain external form agrees |
| decode | direct sum with an unrelated external dictionary |
| validation | shapes, illegal indices, non-finite, bounds, empty images |

It also runs randomized stress tests across the full parameter ranges during
development. All 13 checks pass.

---

## 6. Computational / storage trade-offs

- **Code search:** full `K**L` enumeration. Exhaustive, trivially correct and
  small (≤ 64 codes); costs `O(N · K**L · d)` time and `O(K**L · d)` for cached
  reconstructions. No approximation, so no optimality claim depends on a
  heuristic.
- **Dictionary solve:** one `(L·K) × (L·K)` normal-equation solve for all `d`
  coordinates. Intermediate storage is `O((L·K)² + L·K·d)`, tiny for these
  bounds; it captures all cross-layer terms.
- **Region assignment:** exhaustive injection enumeration, `≤ 360` per image,
  `O(R!/(M−R)!)` work and `O(M·d)` decoded storage per image. Exact, with no
  reuse. Materialising all decoded library vectors is allowed and does not
  change the contract.
- **Catalog:** stores only the shared dictionary and codes (plus metadata), not
  original vectors. Full decoding and full scanning are intended.

No large dataset, visual backbone training, online approximate index, database
or network service is needed.
