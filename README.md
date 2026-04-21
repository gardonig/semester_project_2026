# Enforcing Anatomical Spatial Consistency in Multi-Organ Segmentation via Posets

## Overview

This project enforces **anatomical spatial consistency** in multi-organ segmentation using partially ordered sets (posets).  
Clinicians specify relative positions of structures (vertical, mediolateral, anteroposterior), and the resulting posets can be used to check and correct segmentation outputs.

---

## Relation matrices (one per axis)

For each anatomical axis (vertical, mediolateral, anteroposterior), the app maintains a square **tri-valued relation matrix** \(M\) with one row/column per structure (after sorting—see below).

| Entry | Meaning |
|-------|---------|
| **+1** | Structure \(i\) is **strictly above** structure \(j\) along that axis (directed “yes”). |
| **0** | The pair was **asked**; the expert is **unsure** (still directional). |
| **−1** | \(i\) is **not** strictly above \(j\) (includes “no”, overlap, or opposite direction implied by the CoM prior). |
| **null** | **Not asked yet** (or never queried for that directed cell). |

**Diagonal** entries are fixed to **−1** (self-relations are not “strictly above”).
After **canonical sorting** by CoM on the active axis (descending), **lower triangle** entries with **column index \(j < i\)** are **sealed** to **−1**: in that order, structure \(i\) cannot lie strictly above \(j\) on the \(i\!>\!j\) side of the diagonal. The **strict upper triangle** (\(j > i\) in row-major layout) is where expert queries and inference fill **+1**, **0**, or **−1**; **null** is “still open”.

---

## How the matrix is built (algorithm)

1. **Sort structures** by the chosen axis CoM (descending), matching `matrix_builder.MatrixBuilder`.
2. **Initialize** \(M\): diagonal **−1**, lower triangle **−1**, upper triangle **null** (plus bilateral / equal-CoM rules as implemented).
3. **Gap-based queries** (`next_pair`): pairs \((i, j)\) with \(j = i + \text{gap}\) are considered in order of increasing gap (1, 2, …). Pairs already decided (not **null**), implied by **transitive +1** reachability, or skipped by vertical bilateral symmetry rules are not asked again.
4. **Optional region subsets**: you can restrict **which pairs are asked** to those whose **both** endpoints lie in selected body-region presets; the **saved JSON still lists every structure** and the same **n×n** matrix size so merges stay compatible.
5. **After each answer**, **propagation** updates the matrix: transitive **+1** chains, inverse **−1** when a side is **+1**, mirroring for left/right cores on the vertical axis, and **closure of unknowns** where reachability on **+1** edges forces a direction.  
6. **Saved file** stores three matrices (`matrix_vertical`, `matrix_mediolateral`, `matrix_anteroposterior`) plus the full **structures** list.

**Bilateral structures** (e.g. “Left Lung” / “Right Lung”) on the vertical axis are merged into a single question: *”Are Left Lung and Right Lung strictly above …?”* so both sides are answered simultaneously.

**Hasse diagram (poset viewer)** shows only **cover edges** derived from **+1** entries (transitive reduction of the strict “above” relation). It does **not** draw “unsure” **0** pairs.

---

## Merging multiple raters / sessions

The poset viewer’s **Merge JSON files…** combines several saved posets that describe the **same** set of anatomical structures.

1. **Align** non-reference files to the **first** file’s structure order: index \(i\) must refer to the same organ across files. If the JSON order differs, matrices are **permuted** by matching **name + CoM** (within tolerance), or merge fails if sets are incompatible.
2. **Canonical sort per axis** (vertical / mediolateral / anteroposterior) reorders each rater’s matrix for that axis (CoM descending) and **reseals** the lower triangle to **−1**.
3. **Per-cell aggregation** (`matrix_aggregation.aggregate_matrices_with_counts`): for each directed pair \((i,j)\), each rater contributes **−2** (not asked), or **−1 / 0 / +1** if answered. **−2** is excluded from the mean and vote counts.
4. **Probability matrix** (`matrix_aggregation.aggregate_to_p_yes_matrix`): for each directed pair, compute **\(P(\text{yes})=(\mu+1)/2\)** where **\(\mu\)** is the mean of answered codes only (`−2` excluded). If nobody answered that cell, the saved value is JSON **`null`**.
5. **Hasse extraction rule**: use only edges with **`p == 1`** (strict certainty). Values in \(0 < p < 1\) are partial evidence shown in the heatmap/list summaries but not strict order edges.
6. **Save merged** writes one JSON with **structures** in vertical-CoM order and **reindexed** mediolateral / anteroposterior matrices in `matrix_vertical`, `matrix_mediolateral`, `matrix_anteroposterior` (**P(yes)** cells live there as floats / `null`). The default filename is composed of the stems of the merged files (e.g. `session_alice+session_bob.json`). Optional **`matrix_*_n_answered`** (Σw per cell) and **`matrix_*_n_notasked`** are stored in the JSON to support **weighted** re-merge of already-merged files, but are not shown in the matrix viewer. **`save_poset_to_json`** / **`load_poset_from_json`** also accept arbitrary **`extra`** top-level fields (merge metadata).

**Chained probability merges:** Per file and cell, `μ = 2P - 1` with weight **1** unless the loaded file has **`matrix_*_n_answered ≥ 1`** at that cell (then that value is the weight). `μ` is **Σw-weighted** across files; then `P = (μ+1)/2`. Without sidecars this matches an unweighted mean of `P` when each file contributes once. **Pooling all original experts** still requires merging **raw** tri-valued sessions (or equivalent detail), not only summaries.

---

## Segmentation cleaning (`poset_constraint_postprocessing.py`)

The script `scripts/poset_constraint_postprocessing.py` applies the poset constraints as a **post-processing step** on top of an existing multi-organ segmentation (e.g. TotalSegmentator output). It detects and removes anatomically impossible voxels — predictions that violate the spatial ordering relationships encoded in the poset.

### What it corrects

TotalSegmentator and similar models occasionally predict voxels of a structure in locations that are anatomically impossible given the known position of other structures. Common failure cases include:

- **Truncated scans**: the top or bottom of the scan is cut off, causing stray fragments of ribs or vertebrae to appear at the wrong end of the image.
- **Adjacent structure confusion**: lung lobe voxels bleeding into the territory of a lower lobe, or a rib fragment predicted at the level of a neighboring rib.
- **Out-of-distribution anatomy**: pediatric patients, pathological displacements, or unusual body habitus where the model's spatial priors break down.

The poset encodes constraints of the form **"structure A is strictly superior to structure B"**. Any disconnected fragment of A that appears entirely below B is anatomically impossible and can be safely removed.

Only the **vertical (superior–inferior) axis** is used for cleaning. Mediolateral and anteroposterior constraints are not applied because adjacent structures frequently share the same lateral/AP extent in real patients, making hard cuts unsafe.

### Cleaning method

The script uses **connected component analysis** on the predictions themselves — no ground truth required or used:

```
For each constraint "A is above B":
  1. Extract the largest connected component (LCC) of B
     — this is the trusted main body of B.
  2. Find all connected components of A.
  3. For each non-LCC component of A:
       if the entire component lies below B's LCC inferior boundary
       → remove it (disconnected false positive in the wrong zone).
  4. Conservative rule: A's own LCC is never removed, even if it
     violates the constraint.
```

This works because TotalSegmentator errors are almost always **small disconnected blobs** in anatomically wrong locations, not wholesale misplacements of the main structure body. The largest connected component of each structure is almost always correct; only the outlier fragments in clearly wrong positions are removed.

The conservative rule ensures the method never destroys the main body of a structure based on ambiguous evidence.

### Usage examples

```bash
# Clean a single subject, save output
python scripts/poset_constraint_postprocessing.py \
    --pred_dir data/predictions/amos_v157 \
    --poset    data/posets/llm_sessions/llm_claude_v157.json \
    --subject  amos_0102 \
    --out_dir  data/predictions/amos_v157_cleaned

# Clean all subjects, evaluate Dice against GT
python scripts/poset_constraint_postprocessing.py \
    --pred_dir  data/predictions/v201 \
    --gt_dir    data/datasets/totalseg_v201 \
    --gt_format totalseg_per_subject \
    --poset     data/posets/llm_sessions/llm_claude_v157.json \
    --all_subjects \
    --csv       data/results/v201_v157.csv

# Aggressive mode (remove any non-LCC blob partially below B's top)
python scripts/poset_constraint_postprocessing.py \
    --pred_dir  data/predictions/v201 \
    --gt_dir    data/datasets/totalseg_v201 \
    --gt_format totalseg_per_subject \
    --poset     data/posets/llm_sessions/llm_claude_v157.json \
    --all_subjects \
    --aggressive \
    --csv       data/results/v201_v157_aggressive.csv
```

Note: `--gt_dir` is used **only for Dice evaluation** after cleaning — it plays no role in the cleaning decisions themselves.

---

## Setup

Requirements:

- Python 3.9+

Install (recommended in a virtual environment):

```bash
pip install -e .
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

## Running the GUI

### Starting a session

When you click **Start**, a dialog asks whether to **Continue an existing file** (picks up where that session left off — no answers are lost) or **Create a new file** (blank session). Cancelling returns to the main window without starting. The answer buttons in the query dialog are **Yes \[F\]**, **No \[S\]**, and **Unsure \[D\]** (keyboard shortcuts in brackets).

### Via installed CLI

After installation you can launch the GUI with:

```bash
anatomy-poset
```

To start with a specific structures file (new CoMs):

```bash
anatomy-poset data/structures/test_structures_2.json
```

### Via `run.py` (no installation)

From the project root:

```bash
python run.py
```

With an explicit input file:

```bash
python run.py data/structures/test_structures_2.json
```

---

## Input JSON format

Example (`data/structures/test_structures.json`):

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

- **`com_vertical`**: CoM along the superior–inferior (vertical) axis, scaled to \([0, 100]\):  
  `0` = toes / feet (most inferior), `100` = vertex / head (most superior).
- **`com_lateral`**: CoM along the right–left (lateral) axis (patient’s perspective), scaled to \([0, 100]\):  
  `0` = far right (e.g. right thumb), `100` = far left (e.g. left thumb).
- **`com_anteroposterior`**: CoM along the back–front (anteroposterior) axis, scaled to \([0, 100]\):  
  `0` = back / dorsal side, `100` = front / ventral side.

Output posets are saved under `data/posets/` (organized by `tests/`, `clinician_sessions/`, and `merged_sessions/` subdirectories). Autosave triggers during each query.

---

## Full-body volume tensors

The GUI’s **Full-Body Volume** panel and the helper script `scripts/view_full_body_male.py` expect precomputed NumPy tensors:

- `assets/visible_human_tensors/full_body_tensor_rgb.npy` — RGB tensor `(Z, Y, X, 3)`
- `assets/visible_human_tensors/full_body_tensor.npy` — grayscale tensor `(Z, Y, X)`

These files are **large and not tracked in git**.

- To obtain them, please contact **Gian** (e-mail); he will provide a download link (e.g. via Dropbox).  
- After downloading, place the `.npy` files into `assets/visible_human_tensors/` so the GUI and scripts can find them automatically.

---

## Project structure (short)

```text
├── assets/                      # UI images, tensors, diagrams
├── data/
│   ├── datasets/                # Source datasets (gitignored — large)
│   ├── predictions/             # TotalSegmentator outputs (gitignored — large)
│   ├── results/                 # Dice evaluation CSVs (tracked)
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
│   ├── poset_constraint_postprocessing.py  # CC-based segmentation cleaning
│   ├── truncated_fov_experiment.py         # Simulated truncated FOV evaluation
│   └── llm_poset_builder.py               # LLM-assisted poset construction
├── run.py                       # Quick-start GUI launcher
└── README.md
```
