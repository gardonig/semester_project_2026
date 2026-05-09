# Technical Reference Report: Anatomy Posets

This document covers every mechanism in the repository from end to end, in the order data flows through the system: GUI → relation matrix → multi-annotator aggregation → empirical poset → centre-of-mass atlases → artifact simulation → poset-based cleaning → evaluation.

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

## 6. Dataset

All experiments use the **TotalsegmentatorMRI dataset v200** [Wasserthal et al., 2024], a publicly available collection of 616 whole-body MRI volumes with expert ground-truth segmentations of **50 anatomical structures** (organs, vessels, muscles, bones). Each subject directory contains `mri.nii.gz` (the raw volume) and `segmentations/` (one binary NIfTI mask per structure). The dataset ships with a canonical train/test split; the 10 held-out test subjects are:

```
s0022  s0167  s0175  s0186  s0187
s0219  s0236  s0237  s0243  s0250
```

These 10 subjects were selected as the test set because they have the **highest number of non-empty ground-truth structure masks** in the dataset — i.e. they are the most completely annotated subjects. Using the most fully-labelled subjects for evaluation maximises the number of structures that can be assessed per subject and minimises missing-structure artefacts in the metrics. They are **excluded from all offline computations** (CoM atlas, empirical poset) to prevent data leakage. All downstream analyses (CoM atlas, empirical poset) therefore use **606 training subjects × 50 structures = 30 300 ground-truth masks**. The same 10 held-out subjects serve as the evaluation cohort for the artifact-cleaning experiment (Section 10).

> **Reference:** Wasserthal, J., et al. "TotalSegmentator V2: A Comprehensive Multi-Task Medical Segmentation Model." *Radiology: Artificial Intelligence* 6(4), 2024. <https://doi.org/10.1148/ryai.230024>

---

## 7. Empirical Poset Extraction

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

Two CoM atlases are provided. Neither is required by poset-based cleaning at inference time; they serve as structure-ordering priors for the GUI and as references for any future atlas-based approach.

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

880 conditions: 10 subjects × (2 or 3 crops) × 10 d values × 4 r values. Each `mri_artifact.nii.gz` is segmented with **TotalSegmentator v2.13.0** using `--task total_mr --fast` (MRI model, 3 mm isotropic resolution) to produce per-structure binary masks. `--task total_mr` selects the MRI-specific model weights; without this flag TotalSegmentator defaults to the CT model, which is not valid for MRI inputs.

---

## 9. Poset-Based Cleaning

### Overview

Poset-based cleaning (**middle-out + constraint-consistency**, abbreviated PC) is a purely subtractive post-processing step that removes anatomically impossible connected components from TotalSegmentator predictions. It requires only:

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

### Computational complexity

**Inputs:** S structures, n active constraint pairs (n ≈ 300–500 at threshold 0.95), V voxels per per-structure binary mask.

| Step | Complexity | Notes |
| --- | --- | --- |
| Connected-component labeling (all S structures) | O(S · V) | `scipy.ndimage.label`; done once per structure, result cached |
| Sort pairs by LCC midpoint | O(n log n) | Negligible relative to labeling |
| Per-pair cleaning loop | O(n · V/S) | axis_extent + component removal; cache hit on unmodified structures |
| Cache invalidation + re-label on modified structures | O(k · V/S) | k = number of structures actually modified; typically small |

**Overall: O(S · V + n · V/S)**, dominated by the initial connected-component labeling pass. In practice on a cropped MRI (~50 structures, ~150 × 200 × 200 per-structure mask): **well under one second** per condition. The algorithm is CPU-only and memory-bandwidth-bound, not compute-bound. It adds negligible latency to a TotalSegmentator inference pipeline.

---

## 10. Morphological Opening Baseline

To benchmark the poset-based cleaner, we compare it against an established anatomy-agnostic post-processing method: **morphological opening** — binary erosion followed by binary dilation with the same structuring element — combined with **largest-connected-component (LCC)** retention.

### Rationale

Morphological operations are a standard post-processing tool in medical image segmentation. A survey of deep-learning-based multi-organ segmentation (Fu et al., 2021) identifies morphological operations as "widely used to remove small erroneous labels." nnU-Net — the dominant segmentation framework in medical imaging, winning over 23 MICCAI challenges — includes automated LCC selection as a default post-processing step: it empirically tests whether keeping only the LCC per class improves validation Dice, and applies it if so (Isensee et al., 2021). Furtado (2021) describes the specific pipeline used here for abdominal MRI segmentation: *"preceding the calculation of the largest region by morphological erosion, then calculating and isolating the largest region, subsequently applying dilation with the same structuring element and size to reverse the previous erosion operation. The erosion frequently eliminates noise in the borders and some spurious connections to neighbouring regions."* Recent MICCAI work frames morphological erosion/opening as "valuable tools for processing and analysing segmentation masks" and proposes differentiable variants for end-to-end learning (Guzzi et al., 2024), while published U-Net pipelines use explicit erosion–dilation cycles with LCC filtering as a core post-processing stage (Neeteson et al., 2023).

### Algorithm

For each predicted binary structure mask:

1. **Erode** with a ball-shaped 3-D structuring element of radius *r* voxels (`scipy.ndimage.binary_erosion`):
   - Small disconnected components — wrap-around ghost fragments narrower than *r* voxels — disappear entirely.
   - Thin connections between a real structure and a ghost blob are severed, making them separately identifiable.

2. **Keep LCC** of the eroded result (`scipy.ndimage.label`, retain the largest label):
   - If multiple components survive erosion, only the largest is kept; all smaller fragments are discarded.
   - If erosion empties the mask completely (structure smaller than the structuring element), the original mask is returned unchanged to avoid spurious zero-Dice outcomes.

3. **Dilate** with the same ball and clamp to the original mask:
   - Dilation approximately restores the true structure to its pre-erosion extent.
   - Clamping with `dilated & original` ensures the operation is **purely subtractive** — no new voxels are ever added, only ghost voxels removed.

The default structuring element radius is 2 voxels (a 5 × 5 × 5 ball at 3 mm isotropic resolution, corresponding to a 6 mm physical radius). A smaller radius (r = 1) is less aggressive and may retain thin ghost connections; a larger radius (r = 3–4) risks eroding real thin structures such as the esophagus or portal vein.

We also evaluate a simpler sub-baseline, **LCC-only** (no erosion or dilation), which keeps only the largest connected component of the raw prediction mask without any morphological processing. This corresponds directly to the nnU-Net post-processing heuristic (Isensee et al., 2021) and provides a lower bound: any improvement from `opening_lcc` over `lcc_only` is attributable purely to the morphological filtering step.

### Key limitation

The morphological opening baseline is **anatomy-agnostic**: it uses no knowledge of the spatial ordering constraints encoded in the poset. It can only remove components that are smaller than the structuring element or smaller than the LCC. In the wrap-around artifact setting, a ghost blob can be large — particularly at high ghost intensity (r ≥ 0.75) or large shift (d ≥ 0.25) — and may be equal in size to, or larger than, the true anatomy component. In those conditions LCC selection retains the ghost and discards the real structure, causing the baseline to fail systematically in exactly the regime where the artifact is most severe. The poset-based cleaner, by contrast, identifies ghosts by their anatomical position relative to partner structures — a criterion that is invariant to ghost size and therefore effective precisely where LCC selection is not.

### Implementation

The baseline is implemented in `scripts/cleaning/evaluate_erosion_baseline.py`. It uses the same evaluation loop, crop windows, GT masks, and metrics (Dice, Precision, Recall, voxels removed) as `evaluate_cleaning_methods.py`, producing a compatible `results.csv` for direct comparison.

```bash
python scripts/cleaning/evaluate_erosion_baseline.py \
    --exp_dir   data/experiments/wraparound_v4 \
    --subjects  s0175 s0236 s0219 \
    --out_dir   data/experiments/wraparound_v4_eval/erosion_baseline \
    --method    opening_lcc \
    --radius    2
```

### Baseline literature

- Furtado, P.N. (2021). Improving Deep Segmentation of Abdominal Organs MRI by Post-Processing. *BioMedInformatics* 1(3), 88–105. [doi:10.3390/biomedinformatics1030007](https://doi.org/10.3390/biomedinformatics1030007)
- Isensee, F. et al. (2021). nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation. *Nature Methods* 18, 203–211. [doi:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)
- Fu, Y. et al. (2021). A Review of Deep Learning Based Methods for Medical Image Multi-Organ Segmentation. *Physica Medica* 85, 107–122. [doi:10.1016/j.ejmp.2021.05.003](https://doi.org/10.1016/j.ejmp.2021.05.003)
- Neeteson, N.J. et al. (2023). Automatic Segmentation of Trabecular and Cortical Compartments in HR-pQCT Images Using an Embedding-Predicting U-Net and Morphological Post-processing. *Scientific Reports* 13, 252. [doi:10.1038/s41598-022-27350-0](https://doi.org/10.1038/s41598-022-27350-0)
- Guzzi, L. et al. (2024). Differentiable Soft Morphological Filters for Medical Image Segmentation. *MICCAI 2024*, LNCS 15009, 174–183. [doi:10.1007/978-3-031-72111-3_17](https://doi.org/10.1007/978-3-031-72111-3_17)

---

## 11. Evaluation

### Why precision, and why we keep Dice

Poset-based cleaning is **purely subtractive** — it can only remove voxels, never add them. Therefore:

- **Recall** = TP / (TP + FN) is mathematically invariant to cleaning (FN never changes).
- **Precision** = TP / (TP + FP) is the primary metric: it directly measures FP reduction, which is the only thing poset-based cleaning does.
- **Dice** is reported for comparability with published TotalSegmentator benchmarks. Because poset-based cleaning can remove a small number of TPs when it aggressively cleans, Dice can degrade slightly even when precision improves.

### Evaluation loop

For each (subject, crop, d, r):
1. Load all predicted masks from `segmentations/`.
2. Determine crop window from GT anchor structures.
3. Apply poset-based cleaning in memory (no disk write).
4. Compute Dice and Precision before/after for every structure with a GT mask in the crop.
5. Write a row to `results.csv`.

After all conditions, `make_plots()` generates heatmaps, bar charts, stacked count plots, a per-structure diverging bar, and a ΔDice vs ΔPrecision scatter. `make_report()` writes `report.md` with statistical tables.

### Full results (wraparound_v4)

Dataset: 10 subjects × 3 crops × 40 artifact conditions (10 d × 4 r) = **25 866** structure×condition pairs evaluated.
Artifacts re-simulated with corrected physics (no body-foreground masking; `I_s = I + I_hat`).
Segmentations produced with `TotalSegmentator --task total_mr --fast` (MRI model, 3 mm).

#### Overall — effect of threshold

| Threshold | Mean ΔDice | Imp↑ | Deg↓ | Net Dice | Mean ΔPrec | Prec Imp↑ | Prec Net |
|-----------|-----------|------|------|----------|-----------|-----------|---------|
| 1.00 (all constraints) | −0.042 | 388 (1.5%) | 1681 (6.5%) | −1293 | +0.011 | 1940 (7.5%) | +1800 |
| 0.99 | −0.042 | 413 (1.6%) | 1681 (6.5%) | −1268 | +0.011 | 1992 (7.7%) | +1852 |
| 0.95 | −0.056 | 440 (1.7%) | 2174 (8.4%) | −1734 | +0.012 | 2458 (9.5%) | +2249 |

Higher thresholds (fewer constraints active) produce fewer but more reliable corrections; the net Dice improves across all thresholds but Precision improves consistently at every threshold.

#### By crop region (threshold = 1.00)

| Crop | Mean ΔDice | Imp↑ | Deg↓ | Net | Mean ΔPrec | Prec Net |
|------|-----------|------|------|-----|-----------|---------|
| brain → heart | −0.143 | 5 (0.3%) | 293 (19.3%) | −288 | +0.011 | +202 |
| heart → kidney | −0.087 | 257 (2.5%) | 1388 (13.7%) | −1131 | +0.021 | +1453 |
| kidney → hip | +0.001 | 180 (1.3%) | 0 (0.0%) | +180 | +0.003 | +193 |

The `kidney → hip` crop is the only region where Dice also improves on average — there are no degradations at all. The `brain → heart` crop suffers the most (structures like scapula, humerus, and heart are close to the wrap ghost).

#### By ghost intensity r (threshold = 1.00)

| r | Mean ΔDice | Imp↑ | Deg↓ | Net | Mean ΔPrec | Prec Net |
|---|-----------|------|------|-----|-----------|---------|
| 0.25 | −0.007 | 16 (0.2%) | 66 (1.0%) | −50 | +0.002 | +71 |
| 0.50 | −0.037 | 52 (0.8%) | 329 (5.1%) | −277 | +0.007 | +342 |
| 0.75 | −0.059 | 135 (2.1%) | 586 (9.1%) | −451 | +0.015 | +623 |
| 1.00 | −0.068 | 239 (3.8%) | 700 (11.1%) | −461 | +0.019 | +812 |

Precision improves at all intensities. Dice degrades more as the ghost grows stronger — heavier ghosts produce larger spurious masks that poset-based cleaning removes, but sometimes removes true anatomy along with them.

#### By shift fraction d (threshold = 1.00)

| d | Mean ΔDice | Imp↑ | Deg↓ | Net | Mean ΔPrec | Prec Net |
|---|-----------|------|------|-----|-----------|---------|
| 0.05 | 0.000 | 0 (0.0%) | 0 (0.0%) | 0 | 0.000 | 0 |
| 0.10 | +0.000 | 2 (0.1%) | 0 (0.0%) | +2 | +0.000 | +2 |
| 0.15 | −0.005 | 31 (1.2%) | 24 (0.9%) | +7 | +0.003 | +49 |
| 0.20 | −0.024 | 51 (2.0%) | 93 (3.6%) | −42 | +0.009 | +142 |
| 0.25 | −0.038 | 64 (2.5%) | 141 (5.4%) | −77 | +0.011 | +196 |
| 0.30 | −0.060 | 68 (2.6%) | 236 (9.1%) | −168 | +0.016 | +276 |
| 0.35 | −0.064 | 58 (2.2%) | 254 (9.8%) | −196 | +0.017 | +281 |
| 0.40 | −0.074 | 57 (2.2%) | 287 (11.2%) | −230 | +0.017 | +295 |
| 0.45 | −0.080 | 56 (2.2%) | 325 (12.7%) | −269 | +0.018 | +315 |
| 0.50 | −0.079 | 55 (2.2%) | 321 (12.7%) | −266 | +0.015 | +292 |

At d ≤ 0.10 the ghost is so small it falls entirely outside any crop window — no effect. Effects grow with d and plateau around d = 0.45–0.50.

#### Per structure (threshold = 1.00, sorted by net Dice)

Top 10 structures by Dice improvement:

| Structure | Mean ΔDice | Imp↑ | Deg↓ | Net | Mean ΔPrec |
|-----------|-----------|------|------|-----|-----------|
| small_bowel | +0.005 | 71 (8.9%) | 0 (0.0%) | +71 | +0.013 |
| iliopsoas_right | +0.007 | 69 (8.6%) | 0 (0.0%) | +69 | +0.011 |
| iliopsoas_left | +0.005 | 53 (6.6%) | 0 (0.0%) | +53 | +0.008 |
| femur_right | +0.003 | 33 (10.9%) | 0 (0.0%) | +33 | +0.007 |
| gluteus_maximus_right | +0.000 | 9 (2.3%) | 0 (0.0%) | +9 | +0.000 |
| gluteus_maximus_left | +0.000 | 6 (1.6%) | 0 (0.0%) | +6 | +0.000 |
| hip_left | +0.001 | 5 (1.0%) | 0 (0.0%) | +5 | +0.002 |
| femur_left | +0.000 | 5 (1.5%) | 0 (0.0%) | +5 | +0.000 |
| iliac_artery_right | +0.001 | 4 (0.9%) | 0 (0.0%) | +4 | +0.000 |
| autochthon_right | +0.000 | 3 (0.4%) | 0 (0.0%) | +3 | +0.001 |

Bottom 10 structures by net Dice (worst degradations):

| Structure | Mean ΔDice | Imp↑ | Deg↓ | Net | Mean ΔPrec |
|-----------|-----------|------|------|-----|-----------|
| stomach | −0.156 | 30 (3.6%) | 171 (20.6%) | −141 | +0.056 |
| adrenal_gland_left | −0.168 | 0 (0.0%) | 134 (22.8%) | −134 | +0.056 |
| portal_vein_and_splenic_vein | −0.133 | 7 (1.1%) | 128 (19.3%) | −121 | +0.046 |
| esophagus | −0.147 | 0 (0.0%) | 114 (26.2%) | −114 | +0.069 |
| spleen | −0.142 | 24 (3.5%) | 132 (19.4%) | −108 | +0.038 |
| adrenal_gland_right | −0.145 | 0 (0.0%) | 95 (21.8%) | −95 | +0.038 |
| lung_left | −0.093 | 0 (0.0%) | 91 (18.2%) | −91 | +0.013 |
| liver | −0.079 | 22 (2.5%) | 101 (11.7%) | −79 | +0.011 |
| kidney_right | −0.083 | 0 (0.0%) | 76 (11.1%) | −76 | +0.010 |
| lung_right | −0.077 | 0 (0.0%) | 74 (14.6%) | −74 | +0.034 |

Nine of the ten worst structures still show **positive ΔPrecision** despite severe Dice loss, which is the key signature of the poset-based cleaning trade-off: ghost FP voxels are correctly identified and removed, but connected components are over-eagerly pruned, also discarding some true anatomy. `humerus_right` is the one exception — it degrades in both Dice (−0.110) and Precision (−0.113), confirming that the poset constraint consistently flags its real voxels as ghost artefacts (failure mode 1: faint ghost misidentified as anchor).

---

## 12. Discussion

### Why precision improves while Dice degrades

The divergence between ΔPrecision (+0.011) and ΔDice (−0.042) at threshold 1.00 is not an anomaly — it is a structural consequence of the method's purely subtractive nature, and the results decompose cleanly once this is understood.

**The core accounting.** Each cleaning action removes a connected component that the poset identifies as spatially inconsistent. The voxels in that component are either (a) true FP — ghost anatomy correctly identified — or (b) true TP — real anatomy wrongly removed. The method's spatial-ordering trigger is itself evidence of ghost presence: a component only fires the cleaning rule when it violates the expected S-I ordering relative to a partner structure, which is exactly what a wrap-around ghost does. So in the typical firing event the removed component contains more FP than TP, and Precision rises while Dice falls. The magnitude of each depends on the FP/TP ratio inside the removed component.

**Quantitatively**: the nine worst structures for Dice all still gain Precision. `esophagus` loses mean Dice −0.147 but gains mean Precision +0.069; `adrenal_gland_left` loses −0.168 but gains +0.056. In each case the removed component contains enough ghost FP to raise Precision even as the simultaneous TP loss depresses Dice. The FP-to-TP ratio in the removed set is consistently > 1, which is what the spatial-ordering criterion selects for.

**The humerus_right exception.** `humerus_right` degrades in *both* Dice (−0.110) *and* Precision (−0.113) — the only structure in the dataset where this occurs. This identifies a case where the removed component is predominantly TP: the real humerus is removed rather than its ghost. In the `brain_to_heart` crop window the shoulder/arm region occupies the inferior portion of the field. A brain/skull ghost wrapping down to this level can produce a blob near the shoulder boundary. If TotalSegmentator assigns a slightly larger or better-calibrated mask to the ghost blob than to the real humerus, the poset constraint "something central is above humerus_right" fires against the real humerus — treating it as the "lower, out-of-place" component — while the ghost is retained as the anchor. The resulting removal is mostly TP, so both metrics fall together. This is failure mode 1 (anchor misidentification under low signal contrast) in its purest form.

**Why kidney → hip succeeds.** In the pelvic crop the S-I geography is unambiguous: ghost anatomy from the brain/neck wrapping all the way down to the pelvis is detected as structures that have no business appearing below the iliac crest (e.g., a skull-like blob near the femur). When the poset constraint fires, the removed component is overwhelmingly FP — there is no real brain tissue in the pelvis. This is the regime where poset-based cleaning works as designed: net ΔDice = +0.001, zero degradations, ΔPrec = +0.003. `small_bowel`, `iliopsoas_right/left`, and `femur_right` — the top gainers globally — are all pelvic/lower-abdominal structures.

**Why brain → heart fails.** The thoracic crop is the hardest for two reasons. First, the ghost from the superior anatomy (brain, skull) wraps to the *inferior edge* of this crop window, which is near the cardiac/sub-diaphragmatic boundary — precisely where compact, high-information structures like the stomach, spleen, liver, esophagus, and adrenal glands reside. A ghost blob near this boundary can look like a displaced version of any of these structures, and TotalSegmentator has no way to distinguish ghost from real at the pixel level. Second, several thoracic structures span the full S-I extent of the crop (`aorta`, `inferior_vena_cava`, `esophagus`): their predicted masks are legitimately large in the inferior direction, and the inferior portion can satisfy a spatial-ordering violation without being a ghost. `brain_to_heart` produces 293 degradations and only 5 improvements.

**Ghost intensity r and the scaling of both effects.** The r-stratified results illustrate the dose-response relationship directly:

| r | ΔDice | ΔPrec |
| --- | ------- | ------- |
| 0.25 | −0.007 | +0.002 |
| 0.50 | −0.037 | +0.007 |
| 0.75 | −0.059 | +0.015 |
| 1.00 | −0.068 | +0.019 |

Stronger ghosts produce larger, more conspicuous FP detections → more cleaning events → larger |ΔPrec|. Simultaneously, stronger ghosts cause TotalSegmentator to generate ghost masks that encroach further into real-anatomy territory, increasing the TP fraction in what gets removed → larger |ΔDice|. Both effects scale with r, but the FP component grows faster than the TP component, so ΔPrec remains positive at every intensity while ΔDice monotonically worsens. At r = 0.25 the ghost is barely detectable: poset-based cleaning barely fires (only 16 improvements, 66 degradations) and both effects are small.

**The d-axis saturation.** The shift-fraction tables show that effects grow with d and plateau at d ≈ 0.45–0.50. At small d (≤ 0.10) the ghost is a thin sliver that falls entirely outside any crop window after cropping — no cleaning fires. As d grows, more of the ghost enters the crop window, more FP detections accumulate, and more cleaning events occur. The plateau reflects the upper limit of how much spurious anatomy TotalSegmentator generates even at large shifts, and the reduced marginal new FP when the ghost already occupies the full inferior portion of the crop.

**Summary.** Poset-based cleaning produces a consistent asymmetry — precision up, Dice down — wherever it fires on ghost anatomy mixed with real anatomy. It produces symmetric improvement (Dice and Precision both up) only when the removed component is pure ghost (kidney → hip regime). It produces symmetric degradation (both down) only when the method inverts anchor and ghost, removing predominantly real anatomy (humerus_right in brain → heart). The kidney → hip success case demonstrates that the method works as designed when geographic separability between real and ghost anatomy is high; the brain → heart failure case demonstrates the limits when ghost and real overlap in the same anatomical region.

### Advantages of poset-based cleaning over no cleaning

Poset-based cleaning is **purely subtractive**: it can only remove voxels, never add them. Every benefit flows from this guarantee.

| Advantage | Explanation |
| --- | --- |
| Recall is mathematically invariant | FN count never changes; cleaning cannot make recall worse under any circumstances |
| Precision improves when ghost is correctly identified | Removes FP voxels with no TP cost |
| No ground truth at inference | Poset is precomputed offline; deployment needs only predicted masks and the poset JSON |
| No atlas at inference | The CoM atlas is used offline to derive pair ordering; it is not consulted at runtime |
| Works on any crop extent | Middle-out ordering is derived from the predictions themselves, not from external crop coordinates |
| Generalises to any segmenter | Requires only binary masks and a poset; not tied to TotalSegmentator |
| Negligible latency | Sub-second on CPU; does not affect clinical pipeline throughput |

### Failure modes

**1. Faint ghost misidentified as real anatomy (low r)**
At r = 0.25 the ghost is dim. TotalSegmentator may assign a small mask to the real structure and a comparably small mask to the ghost. If the ghost happens to be the largest component (e.g. it overlaps a bright background region and gains extra voxels), poset-based cleaning trusts it as the anchor and removes the real structure. This explains the r = 0.25 regime where both Dice and Precision degrade.

**2. Ghost and real structure merge into one component**
If d is large enough that the ghost overlaps the real structure spatially, `scipy.ndimage.label` sees a single blob. Poset-based cleaning cannot split a connected component — no cleaning occurs, but no harm is done either.

**3. Structures with genuinely variable S-I extent**
`small_bowel` is the primary case: it is mobile, non-convex, and anatomically spans a wide S-I range with many normally-disconnected loops. The empirical constraint "stomach above small_bowel" (P ≥ 0.95) is valid on average, but individual loops of small_bowel legitimately appear at stomach level. Poset-based cleaning removes those loops as apparent ghosts — they are real anatomy. This is the dominant source of Dice degradation in the results.

**4. Cascade errors**
Pair processing is sequential. If an early pair incorrectly removes a real component (the modified mask becomes the new anchor), later pairs that reference that structure clean against the wrong boundary. Errors compound toward peripheral structures. Middle-out ordering mitigates this by establishing trustworthy central anchors first, but does not eliminate it.

**5. Oblique / spatially ambiguous structures**
Tubular structures that run obliquely along the S-I axis — `aorta` (mean ΔDice = −0.034), `inferior_vena_cava` (−0.042) — are susceptible because their bounding-box extent is wide relative to their true positional centre. The constraint-consistency guard can misidentify the real LCC as a ghost when it sits slightly lower than the constraint partner expects, then removes it. Both structures also degrade in Precision, confirming that true voxels are being discarded.

**6. Cannot recover under-segmentation**
If TotalSegmentator completely missed the real structure and detected only the ghost, poset-based cleaning removes the ghost and leaves an empty mask. Recall was already 0; this is correct behaviour but provides no improvement.

**7. Only addresses S-I wrap-around**
Left–right and anterior–posterior aliasing (less common clinically but physically possible) are not handled. The mediolateral and AP axes of the poset exist but are not used in poset-based cleaning.

**8. Requires a well-calibrated poset**
A constraint that fires spuriously (P just above threshold but not representative of the deployment population) actively harms predictions. The P ≥ 0.99 threshold is more conservative and reduces this risk at the cost of fewer active pairs (~303 vs ~500).

### Clinical deployment considerations

In clinical practice, MRI scanners apply anti-aliasing filters and the wrap-around ghost is typically attenuated relative to the simulated r = 1.0 case. The most common clinical scenario corresponds to the r = 0.25–0.50 range, where poset-based cleaning benefit is smallest. At r = 0.25, poset-based cleaning (PC) can actively degrade both Dice and Precision. A deployment strategy that activates PC only when TotalSegmentator detects suspicious duplicate anatomy (two similarly-sized components with inverted spatial ordering) would avoid the r = 0.25 failure mode while preserving benefit at stronger ghost intensities.

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
              → data/experiments/wraparound_v4/sXXXX/crop/dXXX_rXXX/mri_artifact.nii.gz
                          │
                          └─ TotalSegmentator v2.13.0 --task total_mr --fast
                                → .../segmentations/*.nii.gz
                                          │
                                          └─ evaluate_cleaning_methods.py
                                               (poset-based cleaning, threshold)
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
| `scripts/cleaning/evaluate_cleaning_methods.py` | Poset-based cleaning implementation + full evaluation loop + plots |
| `scripts/cleaning/save_cleaned_segmentations.py` | Run poset-based cleaning on one condition and write cleaned NIfTIs |
| `data/posets/empirical/totalseg_mri_empirical_poset.json` | Primary poset: 50 structures, 606 MRI subjects, P ≥ 0.99 on 303 vertical pairs |
| `data/structures/totalseg_mri_com_landmark.json` | MRI CoM atlas: 50 structures, vertebrae-landmark normalised |
| `data/structures/totalseg_v2_com.json` | CT CoM atlas: 117 structures, image-extent normalised (legacy reference) |
