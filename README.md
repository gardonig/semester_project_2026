# Enforcing Anatomical Spatial Consistency in Multi-Organ Segmentation via Posets

Clinicians specify relative spatial positions of anatomical structures (superior/inferior). These constraints are encoded as a **partially ordered set (poset)** and used to detect and remove anatomically impossible connected components from multi-organ segmentation outputs — no ground truth required at inference time.

---

## Table of Contents

1. [Setup](#setup)
2. [Project Structure](#project-structure)
3. [Poset Construction GUI](#poset-construction-gui)
4. [Empirical Poset](#empirical-poset-from-totalsegmentator-v201)
5. [Segmentation](#segmentation)
6. [Wrap-Around Artifact Simulation (WM3)](#wrap-around-artifact-simulation-wm3)
7. [Cleaning Method CM3](#cleaning-method-cm3)
8. [Evaluation](#evaluation)
9. [Technical Reference](#technical-reference)

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -e .
```

---

## Project Structure

```text
├── data/
│   ├── datasets/                    # Source MRI volumes + GT segmentations — gitignored
│   ├── experiments/                 # Artifact MRIs, segmentations, cleaned segs — gitignored
│   │   ├── wraparound_v3/           # V3 artifacts (880 conditions across 10 subjects)
│   │   ├── wraparound_v3_eval/      # Evaluation results t=0.95 (CSV, plots, report.md)
│   │   └── wraparound_v3_eval_t100/ # Evaluation results t=1.00 (CSV, plots, report.md)
│   ├── structures/
│   │   ├── totalseg_v2_empirical_poset.json  # Empirical poset from 1090 subjects
│   │   └── totalseg_v2_com.json              # Atlas centre-of-mass per structure
│   └── posets/                      # Saved poset JSON files (clinician/LLM sessions)
├── scripts/
│   ├── cleaning/
│   │   ├── evaluate_cleaning_methods.py      # Main evaluation script
│   │   ├── save_cleaned_segmentations.py     # Save cleaned NIfTIs to disk (Slicer)
│   │   ├── poset_constraint_postprocessing.py
│   │   ├── segment_artifacts_array.sh        # SLURM array: TotalSegmentator per artifact
│   │   ├── clean_artifacts_array.sh          # SLURM array: CM3 cleaning per artifact
│   │   └── evaluate_v3_batch.sh              # SLURM single job: full 10-subject eval
│   ├── data_prep/
│   │   ├── simulate_wraparound_artifact.py   # Artifact simulation (WM3)
│   │   ├── simulate_all_subjects.sh          # Run simulation for all 10 subjects
│   │   ├── compute_com_from_gt.py
│   │   └── compute_empirical_poset.py
│   └── dev/                         # Research and visualisation helpers
├── src/anatomy_poset/
│   ├── core/                        # axis_models, io, matrix_builder, matrix_aggregation
│   └── gui/                         # PySide6 GUI
└── run.py
```

---

## Wrap-Around Artifact Simulation (WM3)

### What is wrap-around?

MRI wrap-around (aliasing) occurs when anatomy extends beyond the scanner's field of view. The out-of-FOV signal folds back into the image — anatomy from the **superior** end of the patient appears as a ghost overlay at the **inferior** end. In the coronal view this looks like a faint copy of the skull/brain appearing below the thorax or abdomen.

### Simulation parameters

| Parameter | Symbol | Range used | Meaning |
|-----------|--------|------------|---------|
| Shift fraction | `d` | 0.05 – 0.50 | Fraction of the full volume height that wraps; `d=0.10` means the top 10% of slices appear at the bottom |
| Ghost intensity | `r` | 0.25 – 1.00 | Scaling factor for the ghost signal; `r=1.0` is full-strength, indistinguishable in intensity from the original |

### WM3 normalization — why no normalization

The paper (Eq. 4) applies a brightness normalization factor `I₂/I₁` to equalize the wrapped and non-wrapped region intensities. This works for **horizontal** wrap (large wrapped area). For **vertical S-I wrap** with small `d`, the wrapped strip is a thin sliver, so `I₂/I₁ ≪ 1` and the ghost appears nearly black — not physically realistic.

**WM3 skips normalization entirely:**

```python
# I_r = I + r * ghost_layer, masked to body foreground
I_s = I_r.copy()   # no scaling — ghost adds directly on top of signal
```

In real MRI hardware, aliased signal is added on top of the original signal. `r` alone controls ghost intensity. The wrapped area appears slightly brighter than baseline, which matches real-world scans. WM3 is physically correct and produces visually realistic artifacts across all `(d, r)` combinations.

### Crop windows

After simulating on the full volume, crops are extracted at three anatomical windows defined by GT anchor structures:

| Crop | Superior anchor | Inferior anchor | Subjects |
|------|----------------|-----------------|---------|
| `brain_to_heart` | brain | heart | s0175, s0236 only |
| `heart_to_kidney` | heart | kidney_left | all 10 |
| `kidney_to_hip` | kidney_left | hip_left | all 10 |

Total: **880 conditions** — 10 subjects × (2 or 3 crops) × 10 d-fracs × 4 r-vals.

### Running simulation

```bash
# All 10 subjects, all conditions → data/experiments/wraparound_v3/
bash scripts/data_prep/simulate_all_subjects.sh

# Single subject, custom sweep
python scripts/data_prep/simulate_wraparound_artifact.py \
    --mri_dir    data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --subjects   s0175 \
    --shift_fracs 0.10 0.25 0.50 \
    --intensities 0.25 0.50 0.75 1.00 \
    --out_dir    data/experiments/wraparound_v3
```

---

## Segmentation

TotalSegmentator (`--fast` mode) is run on each `mri_artifact.nii.gz` to produce per-structure binary masks. On the ETH cluster (SLURM / octopus03):

```bash
# generate artifact list
find /scratch/gardonig/wraparound_v3 -name "mri_artifact.nii.gz" | sort > ~/artifact_list_v3.txt

# submit array job (one GPU task per artifact, skips already-done)
ARTIFACT_LIST=~/artifact_list_v3.txt \
  sbatch --array=0-879 scripts/cleaning/segment_artifacts_array.sh
```

---

## Cleaning Method CM3

CM3 (**middle-out + spatial prior**) is the cleaning method used throughout this project. It processes poset constraint pairs and removes connected components of a structure that violate the spatial ordering constraints.

### Why not CM1 or CM2?

- **CM1 (unidirectional):** removes non-LCC components of A that sit below B's inferior boundary. Conservative, but only handles one wrap direction and always trusts the largest component even when it is the ghost.
- **CM2 (symmetric):** also removes non-LCC components of B above A's superior boundary. Catches both wrap directions but accumulates errors — an incorrectly cleaned anchor misleads subsequent pairs.
- **CM3** fixes both issues with three design decisions described below.

### Algorithm

For each constraint pair `(i above j)` from the poset:

1. **Determine the "real" component** of each structure using the atlas spatial prior (see below) rather than blindly trusting the LCC.
2. **Compute anchor extents:** find the S-I extent of the chosen real component of `j` (inferior limit) and `i` (superior limit).
3. **Remove violating components:**
   - Any non-anchor component of `i` that lies entirely *below* `j`'s inferior boundary → removed.
   - Any non-anchor component of `j` that lies entirely *above* `i`'s superior boundary → removed.

Pairs are processed in **middle-out order**: pairs whose combined atlas-expected position is closest to the crop centre are processed first. Central structures (aorta, spine) are cleaned before peripheral ones (hip, femur), so already-clean central anchors inform the removal of peripheral components.

### Spatial prior — choosing the real component

Instead of always picking the LCC, CM3 uses the atlas centre-of-mass (`totalseg_v2_com.json`) to identify which connected component is anatomically expected:

```python
def select_by_prior(mask, si_ax, expected_local, size_dominance=5.0):
    # If LCC is >5× larger than 2nd largest → trust it unconditionally
    if top2_sizes[0] >= size_dominance * top2_sizes[1]:
        return lcc_component

    # Otherwise, pick the component whose centroid is closest to expected_local
    best = min(components, key=lambda c: |centroid(c) - expected_local|)
    return best
```

**Size dominance guard:** when one component is overwhelmingly larger (>5×) it is almost certainly the correct one regardless of position — the prior is only consulted when two similarly-sized components are competing. This prevents the prior from accidentally selecting a tiny noise CC over the large true organ.

### Out-of-crop anchor guard

If the atlas-expected position of a structure falls **outside the current crop's S-I range `[0, N)`**, that structure is skipped as an anchor entirely:

```python
if exp is not None and not (0 <= exp < N):
    return None, None   # ghost prediction — skip as anchor
```

**Why this matters:** a wrap-around artifact may produce a ghost prediction of, say, the femur inside a heart–kidney crop. The femur's atlas position is far outside this crop, so its ghost prediction should not be trusted as an anchor. Without this guard, the ghost femur CC would act as an anchor and incorrectly remove legitimate large components of adjacent structures.

### Poset constraint threshold

Only constraints with empirical probability ≥ threshold are enforced:

| Threshold | Constraints used | Behaviour |
|-----------|-----------------|-----------|
| `0.95` | ≥95% of subjects show this ordering | Standard — uses most reliable constraints |
| `1.00` | 100% of subjects — never violated | Conservative — only rock-solid constraints |

### Running CM3

```bash
# evaluate CM3 on all subjects (runs cleaning in-memory, no pre-saved cleaned dirs needed)
python scripts/cleaning/evaluate_cleaning_methods.py \
    --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exp_dir   data/experiments/wraparound_v3 \
    --poset     data/structures/totalseg_v2_empirical_poset.json \
    --com       data/structures/totalseg_v2_com.json \
    --subjects  s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250 \
    --threshold 0.95 \
    --out_dir   data/experiments/wraparound_v3_eval

# save cleaned NIfTIs to disk for one condition (3D Slicer visualisation)
python scripts/cleaning/save_cleaned_segmentations.py \
    --pred_dir data/experiments/wraparound_v3/s0175/heart_to_kidney/d025_r100/segmentations \
    --out_dir  data/experiments/wraparound_v3/s0175/heart_to_kidney/d025_r100/cleaned \
    --poset    data/structures/totalseg_v2_empirical_poset.json \
    --com      data/structures/totalseg_v2_com.json \
    --method   cm3 --threshold 0.95
```

---

## Evaluation

### Metrics

**Dice** measures overlap between predicted and GT mask. Cleaning only removes false-positive voxels — it never adds voxels — so Dice improves only when FP removal outweighs any small TP loss from aggressive component removal. For artifacts where the ghost overlaps legitimate structure, Dice is relatively insensitive.

**Precision** = TP / (TP + FP) is more informative: it directly measures the FP reduction that cleaning achieves. A positive ΔPrecision means fewer false positives after cleaning with no sensitivity to whether TPs were preserved.

### Output

Each evaluation run produces in `out_dir/`:

| File | Contents |
|------|----------|
| `results.csv` | Per structure × condition rows: dice/precision before and after CM3 |
| `report.md` | Full numerical tables by d, r, crop, and per structure |
| `heatmap_delta_d_r.png` | Side-by-side ΔDice + ΔPrecision heatmap over all (d, r) cells |
| `bar_delta_by_d/r.png` | Mean ΔDice grouped bar charts |
| `bar_prec_by_d/r/crop.png` | Mean ΔPrecision grouped bar charts |
| `counts_stacked_dice/prec.png` | Improved / neutral / degraded counts |
| `counts_per_structure.png` | Per-structure diverging count bar, sorted by net Dice |
| `scatter_dice_vs_prec.png` | ΔDice vs ΔPrecision scatter per structure |

### Current results (WM3 artifacts, 10 subjects)

| Threshold | Conditions | Mean ΔDice | Dice improved | Dice degraded | Mean ΔPrecision | Prec improved |
|-----------|-----------|-----------|--------------|--------------|----------------|--------------|
| 0.95 | 880 | ≈0 | ~4–5% | ~1% | +0.002 | ~5% |
| 1.00 | 880 | ≈0 | ~4–5% | ~1% | +0.002 | ~5% |

Key finding: **precision improves consistently** at r≥0.75 across all d values. Dice is near-zero because cleaning removes mostly FP voxels (increasing precision) without substantially changing the TP/FN balance that Dice measures.

---

## Empirical Poset from TotalSegmentator v2.01

`data/structures/totalseg_v2_empirical_poset.json` encodes the empirical spatial ordering of 117 anatomical structures, extracted from **1090 subjects** using `scripts/data_prep/compute_empirical_poset.py`.

For every ordered pair (i, j):

```text
P(i strictly above j) = count(subjects where inferior boundary of i > superior boundary of j)
                        / count(subjects where both i and j are present)
```

Values are `null` for pairs co-occurring in fewer than 5 subjects. The matrix is strongly bimodal: 5567 pairs exactly zero, 669 pairs exactly one — very little uncertainty.

```bash
python scripts/data_prep/compute_com_from_gt.py \
    --gt_dir /scratch/gardonig/totalseg_v201 \
    --out    data/structures/totalseg_v2_com.json

python scripts/data_prep/compute_empirical_poset.py \
    --gt_dir   /scratch/gardonig/totalseg_v201 \
    --com_json data/structures/totalseg_v2_com.json \
    --out      data/structures/totalseg_v2_empirical_poset.json
```

---

## Poset Construction GUI

### Launching

```bash
anatomy-poset
# or
python run.py
```

### Workflow

Choose to **continue an existing file** or **create a new file**. The GUI presents structure pairs one at a time and asks whether structure A is strictly above structure B along the active axis.

**Keyboard shortcuts:** Yes `[F]` · No `[S]` · Unsure `[D]`

Answers are autosaved. The Hasse diagram updates in real time.

### Input JSON format

```json
{
  "structures": [
    { "name": "Skull", "com_vertical": 90.0, "com_lateral": 50.0, "com_anteroposterior": 50.0 }
  ]
}
```

| Field | Axis | `0` | `100` |
|-------|------|-----|-------|
| `com_vertical` | Superior–inferior | Feet | Head |
| `com_lateral` | Right–left | Far right | Far left |
| `com_anteroposterior` | Back–front | Dorsal | Ventral |

### Merging sessions

Use **Merge JSON files…** in the poset viewer to combine multiple annotator sessions into a probability poset (`P(yes)` per directed pair).

---

## Technical Reference

### Relation matrix

Each axis has an `n×n` tri-valued matrix `M`:

| Value | Meaning |
|-------|---------|
| `+1` | Structure `i` is strictly above `j` |
| `-1` | `i` is not strictly above `j` |
| `0` | Asked; annotator is unsure |
| `null` | Not yet asked |

### Matrix construction algorithm

1. Sort structures by CoM descending.
2. Initialize: diagonal `-1`, lower triangle `-1`, upper triangle `null`.
3. Query pairs `(i, j)` with `j = i + gap`, increasing gap.
4. After each answer, propagate transitivity, bilateral mirroring for left/right pairs.
5. Save three matrices + structures list as JSON.
