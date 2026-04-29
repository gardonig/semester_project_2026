# Enforcing Anatomical Spatial Consistency in Multi-Organ Segmentation via Posets

Clinicians specify relative spatial positions of anatomical structures (superior/inferior, left/right, anterior/posterior). These constraints are encoded as a **partially ordered set (poset)** and used to detect and remove anatomically impossible predictions from multi-organ segmentation outputs.

---

## Table of Contents

1. [Setup](#setup)
2. [Project Structure](#project-structure)
3. [Poset Construction (GUI)](#poset-construction-gui)
4. [Segmentation Cleaning](#segmentation-cleaning)
5. [Evaluating Results](#evaluating-results)
6. [Data](#data)
7. [Technical Reference](#technical-reference)

---

## Setup

**Requirements:** Python 3.9+

```bash
pip install -e .
```

---

## Project Structure

```text
├── assets/                      # UI images, tensors, diagrams
├── data/
│   ├── datasets/                # Source datasets — gitignored (large)
│   ├── predictions/             # TotalSegmentator outputs — gitignored (large)
│   ├── results/                 # Dice evaluation CSVs — tracked
│   ├── posets/                  # Saved poset JSON files
│   │   ├── llm_sessions/        # LLM-generated posets
│   │   ├── clinician_sessions/  # Human-annotated sessions
│   │   └── merged_sessions/     # Multi-rater merged posets
│   ├── structures/              # Input CoM structure JSON files
│   └── README.md                # Dataset and results overview
├── src/anatomy_poset/
│   ├── core/                    # axis_models, io, matrix_builder, matrix_aggregation
│   └── gui/                     # PySide6 GUI (main window, dialogs, poset_viewer)
├── scripts/
│   ├── cleaning/                    # Segmentation constraint cleaning
│   │   ├── poset_constraint_postprocessing.py
│   │   ├── summarize_results.py
│   │   ├── visualize_cleaning.py
│   │   └── truncated_fov_experiment.py
│   ├── poset_construction/          # Poset building (LLM-assisted)
│   │   ├── llm_poset_builder.py
│   │   ├── generate_llm_poset_knowledge.py
│   │   └── generate_llm_poset_v157.py
│   ├── segmentation/                # Third-party segmentor tools
│   │   ├── compare_segmenters.py
│   │   ├── run_medsam.py
│   │   ├── setup_medsam.sh
│   │   └── setup_vibeseg.sh
│   ├── data_prep/                   # Data extraction utilities
│   │   └── CoM_extractor.ipynb
│   └── dev/                         # Research / visualisation helpers
│       ├── view_segmentation.py
│       ├── view_full_body_male.py
│       ├── algorithm1_matrix_walkthrough.py
│       └── stand_alone_poset_anatomy.py
├── run.py                       # Quick-start GUI launcher
└── README.md
```

---

## Poset Construction (GUI)

### Launching

```bash
# After installation
anatomy-poset

# Without installation
python run.py

# With a specific structures file
python run.py data/structures/test_structures_2.json
```

### Workflow

Click **Start** to begin. Choose to **Continue an existing file** or **Create a new file**. The GUI presents structure pairs one at a time and asks whether structure A is strictly above structure B along the active axis.

**Keyboard shortcuts:** Yes `[F]` · No `[S]` · Unsure `[D]`

Answers are autosaved after each query. The Hasse diagram updates in real time showing only confirmed `+1` edges (transitive reduction).

### Input JSON format

Structures are defined by their centre-of-mass coordinates, scaled to `[0, 100]`:

```json
{
  "structures": [
    {
      "name": "Skull",
      "com_vertical": 90.0,
      "com_lateral": 50.0,
      "com_anteroposterior": 50.0
    }
  ]
}
```

| Field | Axis | `0` | `100` |
| --- | --- | --- | --- |
| `com_vertical` | Superior–inferior | Feet | Head |
| `com_lateral` | Right–left (patient) | Far right | Far left |
| `com_anteroposterior` | Back–front | Dorsal | Ventral |

### Merging sessions

Use **Merge JSON files…** in the poset viewer to combine multiple annotator sessions. Matrices are aligned by structure name + CoM, aggregated per cell, and saved as a probability poset (`P(yes)` per directed pair). Only pairs with `p == 1.0` appear as edges in the Hasse diagram.

---

## Segmentation Cleaning

The script `scripts/cleaning/poset_constraint_postprocessing.py` removes anatomically impossible voxels from TotalSegmentator predictions using poset constraints — **no ground truth required**.

### What it corrects

- **Truncated scans** — stray rib or vertebra fragments near a cropped scan boundary
- **Adjacent structure confusion** — a rib fragment predicted at the level of a neighbouring rib
- **Out-of-distribution anatomy** — unusual body habitus where the model's spatial priors break down

### Algorithm

For each constraint *"A is strictly above B"*:

1. Find B's **largest connected component** (LCC) — the trusted main body.
2. Find all connected components of A.
3. For each non-LCC component of A: if it lies entirely below B's superior boundary → **remove it**.
4. A's own LCC is **never removed**.

Only the **vertical axis** is used. Mediolateral and anteroposterior constraints are not applied — adjacent structures legitimately share lateral/AP extents in real patients.

### Modes

| Flag | Behaviour |
| --- | --- |
| *(default)* conservative | Remove a non-LCC blob only if it is **entirely** below B's top |
| `--aggressive` | Remove a non-LCC blob if **any part** of it dips below B's top |

### Usage

```bash
# Clean a single subject
python scripts/cleaning/poset_constraint_postprocessing.py \
    --pred_dir data/predictions/amos_v157 \
    --poset    data/posets/llm_sessions/llm_claude_v157.json \
    --subject  amos_0102 \
    --out_dir  data/predictions/amos_v157_cleaned

# Clean all subjects and evaluate Dice against GT
python scripts/cleaning/poset_constraint_postprocessing.py \
    --pred_dir  data/predictions/v201 \
    --gt_dir    data/datasets/totalseg_v201 \
    --gt_format totalseg_per_subject \
    --poset     data/posets/llm_sessions/llm_claude_v157.json \
    --all_subjects \
    --csv       data/results/v201_conservative.csv

# Aggressive mode
python scripts/cleaning/poset_constraint_postprocessing.py \
    --pred_dir  data/predictions/v201 \
    --gt_dir    data/datasets/totalseg_v201 \
    --gt_format totalseg_per_subject \
    --poset     data/posets/llm_sessions/llm_claude_v157.json \
    --all_subjects --aggressive \
    --csv       data/results/v201_aggressive.csv
```

`--gt_dir` is used **only for Dice evaluation** — it has no effect on the cleaning decisions.

Supported `--gt_format` values: `totalseg_per_subject`, `amos_multilabel`, `flare22_multilabel`, `verse`.

---

## Evaluating Results

```bash
# Summary for one results directory
python scripts/cleaning/summarize_results.py data/results/v201_conservative/

# Summary for multiple directories
python scripts/cleaning/summarize_results.py data/results/v201_conservative/ data/results/v201_aggressive/

# Side-by-side comparison
python scripts/cleaning/summarize_results.py \
    data/results/v201_conservative/ data/results/v201_aggressive/ --compare

# Sort options: delta (default), name, improved, voxels
python scripts/cleaning/summarize_results.py data/results/v201_conservative/ --sort improved
```

Output shows per-structure mean Dice before/after, delta, number of subjects improved/degraded, and total voxels removed.

---

## Empirical Poset from TotalSegmentator v2.01

`data/structures/totalseg_v2_empirical_poset.json` encodes the empirical spatial ordering of 117 anatomical structures, extracted from **1090 subjects** in the TotalSegmentator v2.01 dataset using `scripts/data_prep/compute_empirical_poset.py`.

### What is stored

For every ordered pair (i, j) and each of the three anatomical axes, the matrix stores the empirical probability:

```text
P(i strictly above j) = count(subjects where i is entirely above j) / count(subjects where both i and j are present)
```

"Strictly above" on the vertical axis means the **inferior boundary of i is above the superior boundary of j** — the bounding boxes do not overlap at all. The same strict non-overlap criterion applies to the mediolateral and anteroposterior axes.

Values are `null` for the diagonal and for pairs seen together in fewer than 5 subjects.

### How it was computed

1. **Per subject:** for each segmentation mask, compute its bounding box in normalised [0, 1] image coordinates using the NIfTI affine to map voxel axes to anatomical directions (handles arbitrary orientations via `nibabel`). Structures with fewer than 10 voxels are skipped.
2. **Per pair:** across all subjects where both structures are present, count how often the strict non-overlap condition holds.
3. **Probability:** divide the count by the co-occurrence count, rounded to 4 decimal places.

### Structure ordering

Structures are sorted in topological order (top to bottom) using the vertical axis matrix:

- An edge i → j is added if P(i above j) > 0.5.
- Kahn's algorithm produces a topological sort with no cycles (verified: 0 violations).
- The sort is consistent with the CoM-based ordering from `totalseg_v2_com.json`.

The resulting matrix has a clean lower triangle (all values ≤ 0.5) and is strongly bimodal: most pairs are either near 0.0 (never above, 5567 exactly zero) or near 1.0 (always above, 669 exactly one), with very little uncertainty in between.

### Reproducing

```bash
# Step 1 — compute average CoM per structure
python scripts/data_prep/compute_com_from_gt.py \
    --gt_dir /scratch/gardonig/totalseg_v201 \
    --out    data/structures/totalseg_v2_com.json

# Step 2 — compute empirical poset probabilities
python scripts/data_prep/compute_empirical_poset.py \
    --gt_dir   /scratch/gardonig/totalseg_v201 \
    --com_json data/structures/totalseg_v2_com.json \
    --out      data/structures/totalseg_v2_empirical_poset.json
```

---

## Data

See [`data/README.md`](data/README.md) for a full table of datasets, predictions, and results.

### Full-body volume tensors

The GUI's Full-Body Volume panel requires precomputed NumPy tensors (not tracked in git):

- `assets/visible_human_tensors/full_body_tensor_rgb.npy`
- `assets/visible_human_tensors/full_body_tensor.npy`

Contact **Gian** for the download link.

---

## Technical Reference

### Relation matrix

Each axis has an `n×n` tri-valued matrix `M`:

| Value | Meaning |
| --- | --- |
| `+1` | Structure `i` is strictly above `j` |
| `0` | Asked; annotator is unsure |
| `-1` | `i` is not strictly above `j` |
| `null` | Not yet asked |

The diagonal is fixed to `-1`. After sorting structures by CoM (descending), the lower triangle is sealed to `-1`. The upper triangle is filled by expert queries and transitivity propagation.

### Matrix construction algorithm

1. Sort structures by CoM on the active axis (descending).
2. Initialize: diagonal `-1`, lower triangle `-1`, upper triangle `null`.
3. Query pairs `(i, j)` with `j = i + gap` in order of increasing gap.
4. After each answer, propagate: transitive `+1` chains, inverse `-1` from `+1`, bilateral mirroring for left/right pairs.
5. Save: three matrices (`matrix_vertical`, `matrix_mediolateral`, `matrix_anteroposterior`) + structures list.

Bilateral structures (e.g. Left Lung / Right Lung) are asked as a single joint question on the vertical axis.
