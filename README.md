# Enforcing Anatomical Spatial Consistency in Multi-Organ Segmentation via Posets

Clinicians specify relative spatial positions of anatomical structures (superior/inferior). These constraints are encoded as a **partially ordered set (poset)** and used to detect and remove anatomically impossible connected components from multi-organ segmentation outputs — no ground truth required at inference time.

---

## Table of Contents

1. [Setup](#setup)
2. [Project Structure](#project-structure)
3. [Poset Construction GUI](#poset-construction-gui)
4. [Empirical Poset](#empirical-poset)
5. [Centre-of-Mass Atlases](#centre-of-mass-atlases)
6. [Segmentation](#segmentation)
7. [Wrap-Around Artifact Simulation (WM3)](#wrap-around-artifact-simulation-wm3)
8. [Poset-Based Cleaning](#poset-based-cleaning)
9. [Evaluation](#evaluation)
10. [Technical Reference](#technical-reference)

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
│   ├── experiments/                 # Artifact MRIs, segmentations, eval results — gitignored
│   │   ├── wraparound_v3/           # V3 artifacts (880 conditions across 10 subjects)
│   │   ├── wraparound_v3_eval/      # Evaluation results t=0.95
│   │   └── wraparound_v3_eval_t100/ # Evaluation results t=1.00
│   ├── structures/
│   │   ├── totalseg_mri_com_landmark.json  # MRI CoM atlas, landmark-normalised (test-set excluded)
│   │   └── totalseg_v2_com.json            # CT CoM atlas, image-extent normalised (legacy reference)
│   └── posets/
│       ├── empirical/
│       │   └── totalseg_mri_empirical_poset.json  # Empirical poset — 50 structures, 606 MRI subjects
│       ├── clinician_sessions/      # Human-annotated poset sessions
│       ├── llm_sessions/            # LLM-generated poset sessions
│       └── merged_sessions/         # Multi-annotator merged probability posets
├── scripts/
│   ├── cleaning/
│   │   ├── evaluate_cleaning_methods.py      # Main evaluation script (poset-based cleaning)
│   │   ├── save_cleaned_segmentations.py     # Save cleaned NIfTIs for 3D Slicer
│   │   ├── poset_constraint_postprocessing.py
│   │   ├── segment_artifacts_array.sh        # SLURM array: TotalSegmentator per artifact
│   │   ├── clean_artifacts_array.sh          # SLURM array: poset-based cleaning per artifact
│   │   └── evaluate_v3_batch.sh              # SLURM: full 10-subject evaluation
│   └── data_prep/
│       ├── simulate_wraparound_artifact.py   # WM3 artifact simulation
│       ├── simulate_all_subjects.sh          # Run simulation for all 10 subjects
│       ├── compute_com_landmark_normalized.py # Compute MRI CoM atlas (landmark-normalised)
│       ├── compute_empirical_poset.py        # Compute empirical probability poset from GT
│       ├── rank_mri_subjects.py              # Rank subjects by GT coverage
│       ├── analyze_mri_coverage.py           # Analyse structure coverage across MRI dataset
│       └── visualize_mri_wraparound.py       # Visualise artifact coronal slices
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

### WM3 normalisation — why no normalisation

The paper (Eq. 4) applies a brightness normalisation factor `I₂/I₁` to equalise the wrapped and non-wrapped region intensities. This works for **horizontal** wrap (large wrapped area). For **vertical S-I wrap** with small `d`, the wrapped strip is a thin sliver, so `I₂/I₁ ≪ 1` and the ghost appears nearly black — not physically realistic.

**WM3 skips normalisation entirely:**

```python
# I_r = I + r * ghost_layer, masked to body foreground
I_s = I_r.copy()   # no scaling — ghost adds directly on top of signal
```

In real MRI hardware, aliased signal is added on top of the original signal. `r` alone controls ghost intensity. The wrapped area appears slightly brighter than baseline, which matches real-world scans.

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

## Poset-Based Cleaning

Poset-based cleaning (**middle-out + constraint-consistency**) processes poset constraint pairs and removes connected components that violate the spatial ordering. It requires only the predicted masks, the poset, and the image orientation — no atlas, no crop coordinates, no external prior.

### Design motivation

Two internal preliminary variants (M1 and M2) were developed iteratively:

- **Preliminary method M1 (unidirectional):** removes non-LCC components of i that sit below j's LCC inferior boundary. Conservative, but only handles one wrap direction and always trusts the LCC even when it is the ghost.
- **Preliminary method M2 (symmetric):** also removes non-LCC components of j above i's superior boundary. Catches both wrap directions but accumulates errors — an incorrectly cleaned anchor misleads subsequent pairs.
- **Poset-based cleaning** fixes both issues with two design decisions:

### Algorithm

For each constraint pair `(i above j)` from the poset, processed in middle-out order:

**Step 1 — Constraint-consistency guard.**
If i's LCC is entirely below j's LCC, the constraint is already violated. Because the pair ordering encodes the CoM prior (i should be superior to j), i's LCC is displaced from its expected position — i's LCC is the ghost. All i-components below j's inferior boundary are removed with no protection (`protect_anchor=False`), including the ghost LCC itself. Any real fragment of i that already sits above j is preserved and becomes the new anchor for subsequent pairs.

**Step 2 — Normal symmetric cleaning.**
Using the surviving LCC of each structure as anchor:

- Remove any non-anchor component of i that lies entirely *below* j's inferior boundary.
- Remove any non-anchor component of j that lies entirely *above* i's superior boundary.

**Pair ordering — middle-out.** Pairs whose combined observed LCC midpoint is closest to the image centre are processed first. Central structures (aorta, spine) are cleaned before peripheral ones (brain, femur), so already-clean central anchors inform the removal of peripheral components.

**Why the pair ordering identifies the ghost.** The empirical poset pair `(i above j)` is derived from the CoM atlas: it fires only when i's anatomical position is consistently superior to j's. If i's LCC appears below j's LCC, i is in the wrong position → i is the ghost. No explicit atlas lookup is needed at inference time.

### Poset constraint threshold

Only constraints with empirical probability ≥ threshold are enforced:

| Threshold | Behaviour |
| --------- | --------- |
| `0.95` | ≥95% of subjects show this ordering — uses most reliable constraints |
| `0.99` | ≥99% — tighter, fewer pairs active |
| `1.00` | 100% — only constraints never violated in any training subject |

### Running poset-based cleaning

```bash
# evaluate poset-based cleaning on all subjects (runs cleaning in-memory)
python scripts/cleaning/evaluate_cleaning_methods.py \
    --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exp_dir   data/experiments/wraparound_v3 \
    --poset     data/posets/empirical/totalseg_mri_empirical_poset.json \
    --subjects  s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250 \
    --threshold 0.95 \
    --out_dir   data/experiments/wraparound_v3_eval

# save cleaned NIfTIs to disk for one condition (3D Slicer visualisation)
python scripts/cleaning/save_cleaned_segmentations.py \
    --pred_dir data/experiments/wraparound_v3/s0175/heart_to_kidney/d025_r100/segmentations \
    --out_dir  data/experiments/wraparound_v3/s0175/heart_to_kidney/d025_r100/cleaned \
    --poset    data/posets/empirical/totalseg_mri_empirical_poset.json \
    --method   pc --threshold 0.95
```

---

## Evaluation

### Metrics

**Dice** measures overlap between predicted and GT mask. Cleaning only removes false-positive voxels — it never adds voxels — so Dice improves only when FP removal outweighs any small TP loss from aggressive component removal.

**Precision** = TP / (TP + FP) is more informative: it directly measures the FP reduction that cleaning achieves. A positive ΔPrecision means fewer false positives after cleaning. Because recall = TP / (TP + FN) is invariant to cleaning (poset-based cleaning never adds voxels), precision is the metric that most faithfully reflects what poset-based cleaning does.

### Output

Each evaluation run produces in `out_dir/`:

| File | Contents |
|------|----------|
| `results.csv` | Per structure × condition rows: dice/precision before and after poset-based cleaning |
| `report.md` | Full numerical tables by d, r, crop, and per structure |
| `heatmap_delta_d_r.png` | ΔDice + ΔPrecision heatmap over all (d, r) cells |
| `bar_delta_by_d/r.png` | Mean ΔDice grouped bar charts |
| `bar_prec_by_d/r/crop.png` | Mean ΔPrecision grouped bar charts |
| `counts_stacked_dice/prec.png` | Improved / neutral / degraded counts |
| `counts_per_structure.png` | Per-structure diverging count bar, sorted by net Dice |
| `scatter_dice_vs_prec.png` | ΔDice vs ΔPrecision scatter per structure |

---

## Empirical Poset

`data/posets/empirical/totalseg_mri_empirical_poset.json` encodes the empirical spatial ordering of **50 anatomical structures** extracted from **606 MRI subjects** (10 test subjects excluded to prevent data leakage) using `scripts/data_prep/compute_empirical_poset.py`.

For every ordered pair (i, j):

```text
P(i strictly above j) = count(subjects where inferior boundary of i > superior boundary of j)
                        / count(subjects where both i and j are present)
```

Values are `null` for pairs co-occurring in fewer than 5 subjects. 303 pairs have P ≥ 0.99 on the vertical axis.

```bash
# recompute (excluding test subjects)
python scripts/data_prep/compute_empirical_poset.py \
    --gt_dir      /scratch/gardonig/TotalsegmentatorMRI_dataset_v200 \
    --com_json    data/structures/totalseg_mri_com_landmark.json \
    --out         data/posets/empirical/totalseg_mri_empirical_poset.json \
    --exclude     s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250
```

---

## Centre-of-Mass Atlases

Two CoM atlases are provided. Neither is required by poset-based cleaning at inference time, but they are available if a spatial prior is needed for future work.

### `totalseg_mri_com_landmark.json` — MRI, landmark-normalised (primary)

Computed from **TotalsegmentatorMRI v200**, excluding the 10 evaluation subjects. Covers the same **50 structures** that TotalSegmentator MRI predicts.

Each structure's centroid is normalised by the **vertebrae span** present in each subject (inferior edge of bottommost vertebra ≈ L5 to superior edge of topmost ≈ C1 = 0–100). Structures outside the spine (brain, femur) extrapolate naturally beyond this range. This reference frame is anatomically consistent regardless of scan field-of-view.

```bash
python scripts/data_prep/compute_com_landmark_normalized.py \
    --mri_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exclude s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250 \
    --out     data/structures/totalseg_mri_com_landmark.json
```

### `totalseg_v2_com.json` — CT, image-extent normalised (legacy reference)

Computed from the **TotalSegmentator v2.01 CT** benchmark dataset. Covers **117 structures** (the full CT label set), including individual vertebrae levels (C1–L5) not available in the MRI label set.

Each structure's centroid is expressed as a fraction of the **image extent** of the source CT scan. Because different scans have different fields of view, averages across subjects are less anatomically consistent than the landmark-normalised MRI version. Kept as a reference for CT-specific work or if the larger structure set is needed.

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
2. Initialise: diagonal `-1`, lower triangle `-1`, upper triangle `null`.
3. Query pairs `(i, j)` with `j = i + gap`, increasing gap.
4. After each answer, propagate transitivity, bilateral mirroring for left/right pairs.
5. Save three matrices + structures list as JSON.
