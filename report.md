# Technical Reference Report: Anatomy Posets

This document covers every mechanism in the repository from end to end, in the order data flows through the system: GUI → relation matrix → multi-annotator aggregation → empirical poset → centre-of-mass atlases → artifact simulation → CM3 cleaning → evaluation.

---

## 1. Poset Construction GUI

### Overview

The GUI (`src/anatomy_poset/gui/`) is a PySide6 desktop application for eliciting anatomical spatial ordering from a human annotator. Launch it with:

```bash
anatomy-poset
# or
python run.py
```

The annotator either opens an existing session JSON or starts a new one. On start-up the application loads a list of anatomical structures (each with a name and three centre-of-mass coordinates), instantiates a `MatrixBuilder`, and begins presenting pairs.

### Workflow

Pairs are shown one at a time. The annotator answers whether structure A is **strictly above** structure B along the active axis (vertical by default). Three answers are available:

| Key | Meaning | Stored value |
|-----|---------|-------------|
| `F` | Yes — A is strictly above B | `+1` |
| `S` | No — A is not strictly above B | `−1` |
| `D` | Unsure | `0` |

Responses are auto-saved after every answer. The Hasse diagram (derived from all `+1` edges) updates live.

### Session JSON format

```json
{
  "structures": [
    { "name": "brain", "com_vertical": 115.3, "com_lateral": 50.0, "com_anteroposterior": 50.0 }
  ],
  "matrix_vertical": [[...], ...],
  "matrix_mediolateral": [[...], ...],
  "matrix_anteroposterior": [[...], ...]
}
```

CoM values for the vertical axis use the vertebrae-span normalisation: 0 = L5 inferior, 100 = C1 superior (structures above C1, such as the brain, exceed 100; structures below L5 are negative).

### Merging sessions

The **Merge JSON files...** dialog in the poset viewer combines multiple annotator sessions into a probability poset. Each cell in the output stores `P(yes)` — the fraction of annotators who answered `+1` for that directed pair. The merge logic is described in Section 4 (Multi-annotator aggregation).

---

## 2. Tri-valued Relation Matrix

### Cell semantics

The `n×n` matrix `M` stores one of four values per directed cell `M[i][j]` ("is structure i strictly above structure j?"):

| Value | Meaning |
|-------|---------|
| `+1` | i is strictly above j (annotator said Yes) |
| `−1` | i is not strictly above j (annotator said No, or implied by prior) |
| `0` | Queried but annotator was unsure |
| `None` | Not yet asked |

`None` serialises as JSON `null`. This vocabulary is richer than a simple yes/no matrix: the `None` entries distinguish "unanswered" from "answered No", which is important for aggregation and for knowing which pairs the annotator still needs to see.

### Initial state

When a `MatrixBuilder` is constructed, `initial_tri_valued_relation_matrix(n)` fills the matrix as follows:

- **Diagonal** (`M[i][i]`): `−1` — a structure cannot be strictly above itself.
- **Strict lower triangle** (`M[i][j]` where `i > j`): `−1` — structures are sorted by CoM descending, so by construction `CoM[i] < CoM[j]` for `i > j`, meaning i cannot be strictly above j under the CoM prior.
- **Strict upper triangle** (`M[i][j]` where `i < j`): `None` — these are the cells the annotator needs to fill.

After this initialisation, `_apply_com_not_above_prior()` also sets `−1` for any off-diagonal pair where the two structures have the same CoM value (neither can be strictly above the other).

---

## 3. Gap-Based Query Algorithm

### Pair ordering

The annotator is never asked to compare all n(n−1)/2 upper-triangle pairs; instead `MatrixBuilder` uses a **gap-based CoM strategy**. After sorting structures by descending CoM, the index gap between two structures is a proxy for how far apart they are anatomically. The query order is:

1. gap = 1: ask adjacent-CoM pairs (i, i+1) for i = 0 … n−2
2. gap = 2: ask (i, i+2) for i = 0 … n−3
3. … and so on up to gap = n−1

Within each gap level, the iterator sweeps i from 0 to n−1−gap. The rationale is that nearby structures have the most uncertain ordering (the annotator is most useful there), while transitivity propagation often resolves distant pairs automatically after nearby ones are answered.

### Skipping already-decided pairs

`next_pair()` skips any `(i, j)` where:

1. `M[i][j]` is not `None` — already answered (directly or by propagation).
2. `path_exists_matrix(i, j)` returns True — there is a `+1` chain from i to j through intermediate structures, so i being above j is already implied by transitivity (see Section 3.2).
3. Both `path_exists_matrix(i, j)` and `path_exists_matrix(j, i)` are True simultaneously — this would be a cycle in the `+1` graph; the cell is marked `0` (ambiguous) and skipped.
4. The pair is a bilateral Left/Right symmetric partner on the vertical axis (kidneys, femurs, etc.) — same-side anatomical cores cannot be strictly ordered on the vertical axis, so these cells are pre-set to `−1`.
5. `query_allowed_indices` is provided and either endpoint falls outside the allowed subset.

### Estimating remaining questions

`estimate_remaining_questions()` traverses the same gap loop from the current position and counts pairs that would not be skipped, giving the annotator a progress estimate.

---

## 4. Transitivity and Bilateral Symmetry

### Transitive closure (`_propagate`)

After every answer, `_propagate()` runs a fixed-point loop:

> If `M[i][j] == +1` and `M[j][k] == +1`, infer `M[i][k] = +1` (transitivity of strict ordering).

For each new `+1` entry:
- The inverse cell is set: `M[k][i] = −1` (strict ordering is asymmetric).
- Inference is blocked if `M[i][k]` is already `−1` (an explicit contradiction from the annotator is not overridden).
- Inference is blocked if `CoM[i] ≤ CoM[k]` (the CoM prior: we never infer that a lower-CoM structure is above a higher-CoM one purely by transitivity).
- Inference is blocked for Left/Right same-core pairs on the vertical axis.

After the `+1` fixed-point, `_close_transitive_unknowns()` sweeps remaining `None` cells: if a directed path exists from i to j (or j to i) in the `+1` graph, the corresponding cell is sealed with `+1` (or `−1`) so the annotator is not asked about an already-determined pair.

### Bilateral symmetry enforcement

For the vertical axis, Left and Right organs of the same anatomical core (e.g. `kidney_left` and `kidney_right`) must behave identically in all comparisons with third structures — there is no anatomical basis for "the left kidney is above the right lung but the right kidney is not". `_sync_vertical_bilateral_mirrors()` ensures:

- For every column k: `M[left][k] == M[right][k]` (if one is answered and the other is not, the answered value is copied; if they disagree, the smaller-index entry wins as a tie-break).
- For every row k: `M[k][left] == M[k][right]` (same).

The same-core pair itself is always `−1` in both directions (neither side is above the other on the vertical axis).

---

## 5. Multi-Annotator Aggregation

### CellAggregate

`aggregate_matrices_with_counts()` (`src/anatomy_poset/core/matrix_aggregation.py`) combines an arbitrary number of session files (tri-valued or probability) into a per-cell summary. Each cell accumulates:

- **mean** (µ) — weighted average of answers in [−1, +1]; tri-valued answers map as `+1 → +1`, `−1 → −1`, `0 → 0`. Probability files first convert `P` to µ via `µ = 2P − 1`.
- **n_answered** — total answer-weight contributed by answering files.
- **n_notasked** — number of files where the cell was `None`.
- **counts** — raw vote histogram {−1, 0, +1} for tri-valued sources.

### Weighted merging

When merging probability files that carry a `matrix_*_n_answered` sidecar (saved from a previous merge), each file contributes weight equal to its stored `n_answered` count for that cell (minimum 1 when missing). Raw tri-valued session files contribute weight 1 per cell. This lets a merged summary from 10 annotators count correctly against a single fresh annotator in a subsequent merge.

### Projecting to P(yes)

`aggregate_to_p_yes_matrix()` converts the per-cell mean µ ∈ [−1, +1] to a probability:

```
P(yes) = (µ + 1) / 2
```

`P(yes) = 1.0` means every annotator said Yes; `P(yes) = 0.0` means every annotator said No; `P(yes) = 0.5` means the annotators were split 50/50 or uniformly unsure. Cells where no file contributed an answer remain `null` in the output.

### Structure alignment (permutation)

Different annotation sessions may have structures listed in different orders (because different CoM JSON files were used as input). `align_matrix_lists_to_reference()` aligns all input files to a single reference structure ordering before aggregation. `find_alignment_permutation()` finds the bijective mapping from each file's structure list to the reference list by matching on (name, CoM) with a small floating-point tolerance, then `permute_relation_matrix()` reindexes rows and columns accordingly (`out[i][j] = M[perm[i]][perm[j]]`).

---

## 6. Empirical Poset Extraction

### Purpose

Instead of asking human annotators, we can derive a spatial ordering automatically from ground-truth segmentations across a large cohort. The empirical poset captures how consistently one structure is located strictly superior to another across real subjects.

### Script: `compute_empirical_poset.py`

For every ordered pair `(i, j)` and every anatomical axis:

```
P(i strictly above j) = count(subjects where bbox(i) is entirely above bbox(j))
                        / count(subjects where both i and j are present)
```

"Strictly above" on the vertical axis means the anatomical **inferior boundary** of structure i (the lowest non-zero voxel projected to the S-I axis) lies above the anatomical **superior boundary** of structure j (the highest non-zero voxel). No overlap is allowed — if even a single voxel of i's extent overlaps j's extent on the S-I axis, the pair does not count as strictly above for that subject.

The script:
1. Discovers all subject directories and optionally excludes a held-out test set (`--exclude`).
2. For each subject, loads every `segmentations/*.nii.gz`, converts bounding box extents to anatomical fractions via the affine's axis codes (handling both S-axis and I-axis orientations).
3. Accumulates `strictly_above[ax][i,j]` and `both_present[i,j]` counts.
4. Computes the ratio; cells with fewer than `--min_subjects` (default 5) co-occurrences are left as `null`.
5. Writes a JSON in the same format as a merged annotator session, loadable directly by the GUI and cleaning scripts.

### Primary poset: `totalseg_mri_empirical_poset.json`

- **50 structures** from TotalSegmentator MRI v200
- **606 subjects** (616 total minus 10 held-out test subjects)
- Test subjects excluded: `s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250`
- **303 pairs** have P ≥ 0.99 on the vertical axis
- Structure ordering fixed by the MRI CoM atlas (`--com_json`)

---

## 7. Centre-of-Mass Atlases

Two CoM atlases are provided. Neither is required by CM3 at inference time; they serve as structure-ordering priors for the GUI and as references for any future atlas-based approach.

### MRI atlas: `totalseg_mri_com_landmark.json`

**Normalization:** vertebrae-landmark normalisation. For each subject, the combined vertebrae mask (`vertebrae.nii.gz`) provides two landmarks:

- `vert_sup` — S-I voxel index of the vertebrae's superior edge (≈ C1 top)
- `vert_inf` — S-I voxel index of the vertebrae's inferior edge (≈ L5 bottom)

Each structure's centroid along S-I is then:

```
com_vertical = 100 × (centroid_si − vert_inf) / (vert_sup − vert_inf)
```

This is scan-extent independent: two subjects with different fields of view but identical anatomy produce the same `com_vertical` for every organ, because the centroid and the landmark are expressed in the same subject-local frame. Structures below L5 (femur, hip, gluteus) have `com_vertical < 0`; structures above C1 (brain) have `com_vertical > 100`.

Lateral (`com_lateral`) and AP (`com_anteroposterior`) coordinates use image-extent normalisation (fraction 0–100 of the scan axis), which is less consistent across subjects but sufficient for the GUI's bilateral-symmetry pairing.

**Coverage:**
- 50 structures (the full TotalSegmentator MRI label set)
- 606 subjects (10 test subjects excluded to prevent data leakage)
- Structures seen in fewer than 5 subjects are dropped

### CT atlas: `totalseg_v2_com.json`

**Normalization:** image-extent normalisation (each centroid expressed as a fraction of the scan's axis length). This is less anatomically consistent than the landmark-normalised version because scans with different fields of view produce different fractions for the same organ.

**Coverage:**
- 117 structures (the full TotalSegmentator v2.01 CT label set)
- Includes individual vertebrae levels (C1–L5) not available in the MRI label set
- Computed from the TotalSegmentator v2.01 CT benchmark dataset

The CT atlas is kept as a legacy reference for CT-specific work or when the larger structure set is needed.

---

## 8. Wrap-Around Artifact Simulation (WM3)

### Physical model

MRI wrap-around (aliasing) occurs when anatomy extends beyond the scanner's field of view (FOV). Signal from outside the FOV folds back into the image. In the superior–inferior direction, anatomy just above the superior FOV boundary wraps to appear at the inferior end of the image, and anatomy just below the inferior FOV boundary wraps to appear at the superior end.

WM3 simulates this by taking a cropped anatomical window (the simulated FOV) and adding ghost layers from just outside each boundary:

```
extended window = N + 2d slices   (d above FOV + N FOV + d below FOV)

  d slices above FOV  →  wrap to INFERIOR end of FOV image
  ──────────────────────────────────────────────────────
       N slices = FOV image  (what TotalSegmentator sees)
  ──────────────────────────────────────────────────────
  d slices below FOV  →  wrap to SUPERIOR end of FOV image
```

### Parameters

| Parameter | Symbol | Range used | Meaning |
|-----------|--------|------------|---------|
| Shift fraction | `d` | 0.05–0.50 | Fraction of FOV height that wraps in from each edge |
| Ghost intensity | `r` | 0.25–1.00 | Scaling factor applied to the ghost signal |

`d = 0.10` means the 10% of the full volume just outside each FOV boundary wraps in. `r = 1.0` means the ghost has the same intensity as the original signal.

### Implementation

For each anatomical crop window and each (d, r) combination:

1. Load the full MRI volume and GT segmentations to determine the crop extent.
2. Extend the window by d above and below to build `I_hat` (ghost layer):
   ```python
   I_hat = np.zeros_like(I, dtype=np.float32)
   # For S-axis (si_sign=+1, high index = superior):
   I_hat[si_slice(0, d)] = I[si_slice(M - d, M)] * r   # top wraps to bottom
   I_hat[si_slice(N - d, N)] = I[si_slice(M, M + d)] * r   # bottom wraps to top
   ```
3. Add ghost directly: `I_s = I + I_hat`. No body masking, no brightness normalisation — real MRI aliasing adds the wrapped signal everywhere (air, background, tissue). `r` alone controls ghost intensity.
4. Crop the result to `[crop_lo, crop_hi]` along the S-I axis; update the affine origin if the crop removes slices from the inferior end.

### Why no body masking and no brightness normalisation

The paper applies a foreground mask `F = (I > 0)` and a brightness normalisation factor `I2/I1`. Both are omitted in WM3 for physical correctness:

- **No body masking:** Real MRI aliasing is a Fourier phenomenon — the aliased signal adds everywhere in the image, including in background air. Masking the ghost to `I > 0` produces a jagged artifact at the body boundary (because MRI background is not exactly zero — thermal noise gives small non-zero values) rather than the clean ghost a scanner would produce.
- **No brightness normalisation:** The `I2/I1` factor is designed for large horizontal wraps where the wrapped area is a substantial fraction of the image. For vertical S-I wrap with small `d`, the wrapped strip is a thin sliver, so `I2/I1 ≪ 1` and the ghost would appear nearly black — not physically realistic. `r` alone controls ghost intensity and directly matches the real parameter (ghost signal = r × wrapped anatomy signal).

### Crop windows

Three anatomical windows are defined by the S-I extent of GT anchor structures (±5 voxel margin):

| Crop | Superior anchor | Inferior anchor | Available subjects |
|------|----------------|-----------------|-------------------|
| `brain_to_heart` | brain | heart | s0175, s0236 only |
| `heart_to_kidney` | heart | kidney_left | all 10 |
| `kidney_to_hip` | kidney_left | hip_left | all 10 |

### Experiment scale

880 conditions: 10 subjects × (2 or 3 crops) × 10 d values × 4 r values. TotalSegmentator (`--fast` mode) is run on each `mri_artifact.nii.gz` to produce per-structure binary masks.

---

## 9. Cleaning Method CM3

### Overview

CM3 (**middle-out + constraint-consistency**) is a purely subtractive post-processing step that removes anatomically impossible connected components from TotalSegmentator predictions. It requires only:

- The predicted binary masks (one per structure)
- The empirical poset (a JSON with P(i above j) for each pair)
- The image affine (to determine the S-I axis orientation)

No atlas is consulted at inference time. The pair ordering in the poset, derived offline from the CoM atlas, is sufficient to identify ghosts.

### Connected component analysis

For each predicted mask, `get_components()` uses `scipy.ndimage.label` to find all connected components. The **largest connected component (LCC)** by voxel count is taken as the primary candidate for the real structure. `axis_extent(mask, si_ax)` returns `(min_voxel_index, max_voxel_index)` along the S-I axis for any binary mask.

### Active constraint pairs

`_get_pairs(poset, threshold)` extracts all pairs `(name_i, name_j)` from the poset where `P(i above j) ≥ threshold`. Three thresholds are evaluated:

| Threshold | Behaviour |
|-----------|-----------|
| 0.95 | Fires when ≥95% of training subjects show this ordering |
| 0.99 | Fires when ≥99% show this ordering |
| 1.00 | Fires only for pairs never violated in any training subject |

### Middle-out ordering

Pairs are processed in order of how central the predicted LCCs are to the image. For each pair `(i, j)`, compute the midpoint between i's LCC centre and j's LCC centre along S-I, then sort ascending by distance of that midpoint from the image centre `N/2`. Pairs involving aorta, spine, and other central structures are processed first; pairs involving brain, femur, and other peripheral structures are processed later. This means the peripheral cleanings are informed by already-trustworthy central anchors.

### Constraint-consistency guard

For pair `(i above j)` — meaning the empirical poset says i is consistently superior to j — the guard checks whether i's LCC is **entirely below** j's LCC:

```python
def _is_entirely_below(ext_a, ext_b) -> bool:
    if si_sign == +1:
        return ext_a[1] < ext_b[0]   # a's superior edge < b's inferior edge
    else:
        return ext_a[0] > ext_b[1]
```

If this condition holds, i is displaced from its expected position. Because the pair ordering encodes the CoM prior (i should be superior to j), there is only one explanation: i's LCC is a wrap-around ghost that appeared at the wrong end of the image. The guard triggers:

```python
if _is_entirely_below(ext_i, ext_j):
    _remove_violated_components(..., below_limit=ext_j[0], protect_anchor=False)
    anchor_i, ext_i = None, None   # skip normal cleaning for this pair
    anchor_j, ext_j = None, None
```

`protect_anchor=False` is critical: it sets `anchor_label = -1`, meaning **no component is protected** from removal, including the LCC itself. All i-components below j's inferior boundary are removed. Any real fragment of i that already sits above j is left intact; on the next access to the CC cache, it will be found as the new LCC and used as the anchor for subsequent pairs.

### Normal symmetric cleaning

After the guard (or if it did not trigger), normal cleaning proceeds:

- Remove any non-anchor component of i that lies entirely below j's inferior boundary (i.e. a stray fragment of i that is too far inferior to be real given j's position).
- Remove any non-anchor component of j that lies entirely above i's superior boundary (i.e. a stray fragment of j that is too far superior given i's position).

The anchor for i is its current LCC; the anchor is never removed in this step.

### Cache invalidation

`_remove_violated_components()` updates `cleaned[name]` in place and calls `cc_cache.pop(name, None)` whenever any voxels are removed. The next access to this structure's CC cache re-runs `get_components()` on the updated mask, ensuring that downstream pairs see the new (post-removal) LCC, not the stale one.

### `_remove_violated_components` in full

```python
def _remove_violated_components(
    cleaned, removed, name, si_ax, si_sign,
    below_limit, above_limit,
    cc_cache,
    anchor_override=None,
    protect_anchor=True,
) -> bool:
    mask = cleaned[name]
    labeled, n, sizes, lcc_label = get_or_compute_cc(mask, cc_cache, name)

    if not protect_anchor:
        anchor_label = -1           # protect nothing
    elif anchor_override is not None:
        # anchor = component with most overlap with anchor_override mask
        anchor_label = component_with_max_overlap(labeled, anchor_override)
    else:
        anchor_label = lcc_label    # default: protect the LCC

    for comp_label in range(1, n + 1):
        if comp_label == anchor_label:
            continue
        ext = axis_extent(labeled == comp_label, si_ax)
        violated = (below_limit is not None and entirely_below(ext, below_limit, si_sign))
                or (above_limit is not None and entirely_above(ext, above_limit, si_sign))
        if violated:
            cleaned[name][labeled == comp_label] = False
            removed[name] += component_size

    if any voxels were removed:
        cc_cache.pop(name, None)
    return changed
```

---

## 10. Evaluation

### Why precision, and why we keep Dice

CM3 is **purely subtractive** — it can only remove voxels, never add them. Therefore:

- **Recall** = TP / (TP + FN) is mathematically invariant to cleaning (FN never changes).
- **Precision** = TP / (TP + FP) is the primary metric: it directly measures FP reduction, which is the only thing CM3 does.
- **Dice** is reported for comparability with published TotalSegmentator benchmarks. Because CM3 can remove a small number of TPs when it aggressively cleans, Dice can degrade slightly even when precision improves.

### Evaluation loop

For each (subject, crop, d, r):
1. Load all predicted masks from `segmentations/`.
2. Determine crop window from GT anchor structures.
3. Apply CM3 in memory (no disk write).
4. Compute Dice and Precision before/after for every structure with a GT mask in the crop.
5. Write a row to `results.csv`.

After all conditions, `make_plots()` generates heatmaps, bar charts, stacked count plots, a per-structure diverging bar, and a ΔDice vs ΔPrecision scatter. `make_report()` writes `report.md` with statistical tables.

### Full results — threshold = 0.95

Dataset: 10 subjects × 3 crops × 40 artifact conditions (10 d × 4 r) = 1 200 (subject, crop, condition) tuples.
Wilcoxon signed-rank test on non-zero differences; per-structure p-values corrected with Benjamini–Hochberg FDR (α = 0.05).

#### Overall

| Metric | Mean Δ | n nonzero | p-value | Result |
|--------|--------|-----------|---------|--------|
| Dice | −0.00089 | 1327 / 12673 | 8.7 × 10⁻⁴⁸ | significant **degradation** |
| Precision | +0.00166 | 1360 / 12673 | 3.1 × 10⁻¹⁰¹ | significant **improvement** |

The opposite directions confirm the metric argument: CM3 systematically reduces FP at the cost of a very small Dice penalty driven by a handful of structures.

#### By crop region

| Crop | Metric | Mean Δ | p-value | Result |
|------|--------|--------|---------|--------|
| brain → heart | Dice | +0.00025 | 7.9 × 10⁻⁷ | improvement |
| brain → heart | Precision | +0.00046 | 7.9 × 10⁻⁷ | improvement |
| heart → kidney | Dice | −0.00041 | 3.5 × 10⁻²⁹ | degradation |
| heart → kidney | Precision | +0.00099 | 6.7 × 10⁻³⁹ | improvement |
| kidney → hip | Dice | −0.00143 | 1.2 × 10⁻¹⁸ | degradation |
| kidney → hip | Precision | +0.00235 | 1.1 × 10⁻⁵⁹ | improvement |

Precision improves in all three crops. Dice degrades in the two lower-body crops, driven by `small_bowel` and the iliac vessels.

#### By ghost intensity (r)

| r | Metric | Mean Δ | Result |
|---|--------|--------|--------|
| 0.25 | Dice | −0.00166 | degradation |
| 0.25 | Precision | −0.00086 | **degradation** |
| 0.50 | Dice | −0.00096 | degradation |
| 0.50 | Precision | +0.00162 | improvement |
| 0.75 | Dice | −0.00069 | degradation |
| 0.75 | Precision | +0.00267 | improvement |
| 1.00 | Dice | −0.00017 | degradation |
| 1.00 | Precision | +0.00350 | improvement |

At r = 0.25 (faintest ghost), both metrics degrade: the ghost is too faint to distinguish from real anatomy and CM3 over-removes. From r = 0.50 onward, precision improves consistently and strongly.

#### By shift fraction (d)

Precision improves significantly at every d value tested (0.05–0.50). Effect sizes are roughly constant across d — shift fraction has little bearing on cleaning quality. Dice degrades at most d values with effect sizes an order of magnitude smaller than the precision improvement.

#### Per structure — significantly degraded (Dice)

| Structure | Mean ΔDice | n nonzero |
|-----------|-----------|-----------|
| iliac_artery_right | −0.07768 | 42 |
| iliac_vena_right | −0.02490 | 19 |
| small_bowel | −0.00376 | 309 |
| stomach | −0.00040 | 113 |

The `iliac_artery_right` degradation is the largest single failure and accounts for most of the overall mean Dice drop.

#### Per structure — significantly improved (Precision)

| Structure | Mean ΔPrecision | n nonzero |
|-----------|----------------|-----------|
| urinary_bladder | +0.02810 | 21 |
| portal_vein_and_splenic_vein | +0.01916 | 93 |
| iliac_vena_left | +0.01625 | 8 |
| duodenum | +0.00975 | 61 |
| pancreas | +0.00728 | 64 |
| brain | +0.00427 | 31 |
| liver | +0.00353 | 108 |
| stomach | +0.00271 | 117 |
| kidney_right | +0.00096 | 104 |
| aorta | +0.00050 | 43 |

---

## Appendix: Data Flow Summary

```
TotalsegmentatorMRI GT
        │
        ├─ compute_com_landmark_normalized.py
        │     → data/structures/totalseg_mri_com_landmark.json
        │
        ├─ compute_empirical_poset.py  (--com_json above)
        │     → data/posets/empirical/totalseg_mri_empirical_poset.json
        │
        └─ simulate_wraparound_artifact.py  (test subjects only)
              → data/experiments/wraparound_v3/sXXXX/crop/dXXX_rXXX/mri_artifact.nii.gz
                          │
                          └─ TotalSegmentator --fast
                                → .../segmentations/*.nii.gz
                                          │
                                          └─ evaluate_cleaning_methods.py
                                               (CM3, poset, threshold)
                                                → results.csv
                                                → report.md
                                                → plots/
```

## Appendix: Key File Index

| File | Purpose |
|------|---------|
| `src/anatomy_poset/core/matrix_builder.py` | `MatrixBuilder`: gap-based query, tri-valued matrix, transitivity, bilateral symmetry |
| `src/anatomy_poset/core/matrix_aggregation.py` | Multi-annotator aggregation: `CellAggregate`, P(yes) projection, matrix alignment |
| `src/anatomy_poset/core/axis_models.py` | `Structure` dataclass, axis name constants |
| `src/anatomy_poset/core/io.py` | JSON load/save for sessions and posets |
| `src/anatomy_poset/gui/` | PySide6 GUI: session manager, Hasse diagram viewer, merge dialog |
| `scripts/data_prep/compute_com_landmark_normalized.py` | Vertebrae-landmark CoM atlas for MRI |
| `scripts/data_prep/compute_empirical_poset.py` | Empirical probability poset from GT bounding boxes |
| `scripts/data_prep/simulate_wraparound_artifact.py` | WM3 artifact simulation (d, r sweep; crop extraction) |
| `scripts/cleaning/evaluate_cleaning_methods.py` | CM3 implementation + full evaluation loop + plots |
| `scripts/cleaning/save_cleaned_segmentations.py` | Run CM3 on one condition and write cleaned NIfTIs |
| `data/posets/empirical/totalseg_mri_empirical_poset.json` | Primary poset: 50 structures, 606 MRI subjects, P ≥ 0.99 on 303 vertical pairs |
| `data/structures/totalseg_mri_com_landmark.json` | MRI CoM atlas: 50 structures, vertebrae-landmark normalised |
| `data/structures/totalseg_v2_com.json` | CT CoM atlas: 117 structures, image-extent normalised (legacy reference) |
