# Reproducing the Wrap-Around Artifact Experiment

Complete end-to-end guide. All commands run locally from the repository root.
No cluster or SLURM required.

---

## Prerequisites

| Component | Notes |
| --- | --- |
| Python 3.10+ | |
| TotalSegmentator ≥ 2.13 | `pip install TotalSegmentator` — needs ≈ 8 GB disk for model weights |
| TotalsegmentatorMRI dataset v200 | 616 subjects, 50 GT masks each — download below |

Install Python dependencies:

```bash
pip install -e .
pip install nibabel numpy scipy pandas matplotlib
```

**Dataset:** Download the TotalsegmentatorMRI v200 dataset from Zenodo
(Wasserthal et al., *Radiology: AI* 2024, https://doi.org/10.1148/ryai.230024).
Place it at:

```
data/datasets/TotalsegmentatorMRI_dataset_v200/
```

**Test subjects** (held out from all offline computations):
```
s0022  s0167  s0175  s0186  s0187
s0219  s0236  s0237  s0243  s0250
```
These 10 were chosen for having the highest number of non-empty GT masks.

---

## Step 0 — (Optional) Recompute CoM atlas and empirical poset

The precomputed files are already in the repo:

- `data/structures/totalseg_mri_com_landmark.json`
- `data/posets/empirical/totalseg_mri_empirical_poset.json`

Skip this step unless you want to regenerate them from scratch:

```bash
# CoM atlas — 606 training subjects, test set excluded
python scripts/data_prep/compute_com_landmark_normalized.py \
    --data_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exclude  s0022 s0167 s0175 s0186 s0187 s0219 s0236 s0237 s0243 s0250 \
    --out      data/structures/totalseg_mri_com_landmark.json

# Empirical poset — same exclusions
python scripts/data_prep/compute_empirical_poset.py \
    --data_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --com_json data/structures/totalseg_mri_com_landmark.json \
    --exclude  s0022 s0167 s0175 s0186 s0187 s0219 s0236 s0237 s0243 s0250 \
    --out      data/posets/empirical/totalseg_mri_empirical_poset.json
```

---

## Step 1 — Simulate wrap-around artifacts

Generate all 40 artifact conditions (10 d × 4 r) per subject and crop, plus the
clean d=0 baseline crop. Output goes to `data/wraparound_experiments/wraparound_v4/`.

```bash
# Artifact conditions for all 10 test subjects
python scripts/data_prep/simulate_wraparound_artifact.py \
    --mri_dir     data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --subjects    s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250 \
    --shift_fracs 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
    --intensities 0.25 0.50 0.75 1.00 \
    --out_dir     data/wraparound_experiments/wraparound_v4

# Clean baseline crops (d=0, r=0) — one per subject × crop region
python scripts/data_prep/simulate_wraparound_artifact.py \
    --mri_dir     data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --subjects    s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250 \
    --shift_fracs 0.05 \
    --intensities 0.25 \
    --out_dir     data/wraparound_experiments/wraparound_v4 \
    --baseline
```

Both commands are idempotent — already-existing files are skipped.

Crop regions per subject:

| Crop | Subjects |
|------|---------|
| `brain_to_heart` | s0175, s0236 only |
| `heart_to_kidney` | all 10 |
| `kidney_to_hip` | all 10 |

Total: **880 artifact conditions** + **22 clean baselines**.

---

## Step 2 — Segment all conditions with TotalSegmentator

Run TotalSegmentator (`--task total_mr --fast`, MRI model at 3 mm) on every
`mri_artifact.nii.gz`. The `--run_totalseg` flag in the simulation script can do
this inline, or run it separately as shown here:

```bash
find data/wraparound_experiments/wraparound_v4 -name "mri_artifact.nii.gz" | sort | \
while read -r mri; do
    seg_dir="$(dirname "$mri")/segmentations"
    if [ -d "$seg_dir" ] && [ "$(ls -A "$seg_dir")" ]; then
        continue   # already segmented
    fi
    TotalSegmentator -i "$mri" -o "$seg_dir" --task total_mr --fast
done
```

**Note on s0022:** Six conditions (`s0022/heart_to_kidney`, d=0.05 all r and d=0.10
r=0.25/0.50) may fail on first pass if TotalSegmentator times out. Re-run them
explicitly if their `segmentations/` directories are missing or empty:

```bash
BASE=data/wraparound_experiments/wraparound_v4/s0022/heart_to_kidney
for tag in d005_r025 d005_r050 d005_r075 d005_r100 d010_r025 d010_r050; do
    TotalSegmentator \
        -i  $BASE/$tag/mri_artifact.nii.gz \
        -o  $BASE/$tag/segmentations \
        --task total_mr --fast
done
```

Expected total after all re-runs: **902 segmentation directories**
(880 artifact + 22 clean baseline).

---

## Step 3 — Poset-based cleaning evaluation (CM4, threshold = 1.00)

Evaluate poset-based cleaning (method CM4) on each of the 10 test subjects.
Each subject writes its results to its own partial subdirectory.

```bash
for SUBJ in s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250; do
    python scripts/cleaning/evaluate_cleaning_methods.py \
        --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
        --exp_dir   data/wraparound_experiments/wraparound_v4 \
        --poset     data/posets/empirical/totalseg_mri_empirical_poset.json \
        --subjects  $SUBJ \
        --threshold 1.00 \
        --method    cm4 \
        --out_dir   data/wraparound_experiments/wraparound_v4_eval_cm4/t100/partial_${SUBJ}
done
```

Running subjects sequentially takes several hours. To run them in parallel (10
background processes):

```bash
for SUBJ in s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250; do
    python scripts/cleaning/evaluate_cleaning_methods.py \
        --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
        --exp_dir   data/wraparound_experiments/wraparound_v4 \
        --poset     data/posets/empirical/totalseg_mri_empirical_poset.json \
        --subjects  $SUBJ \
        --threshold 1.00 \
        --method    cm4 \
        --out_dir   data/wraparound_experiments/wraparound_v4_eval_cm4/t100/partial_${SUBJ} &
done
wait
```

---

## Step 4 — Patch the 6 missing s0022 conditions

After Step 2's s0022 re-run, patch the evaluation for those 6 conditions:

```bash
python scripts/cleaning/evaluate_cleaning_methods.py \
    --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exp_dir   data/wraparound_experiments/wraparound_v4 \
    --poset     data/posets/empirical/totalseg_mri_empirical_poset.json \
    --subjects  s0022 \
    --d_fracs   0.05 0.10 \
    --r_vals    0.25 0.50 0.75 1.00 \
    --threshold 1.00 \
    --method    cm4 \
    --out_dir   data/wraparound_experiments/wraparound_v4_eval_cm4/t100/partial_fix_s0022

python scripts/cleaning/patch_missing_rows.py \
    --eval_dir  data/wraparound_experiments/wraparound_v4_eval_cm4
```

---

## Step 5 — Merge partial results

Combine the per-subject CSVs, regenerate plots and `report.md`:

```bash
python scripts/cleaning/merge_eval_results.py \
    --eval_base data/wraparound_experiments/wraparound_v4_eval_cm4 \
    --tags      t100
```

Output: `data/wraparound_experiments/wraparound_v4_eval_cm4/t100/results.csv`
(expected **25 866 rows**) plus plots and `report.md` in the same directory.

---

## Step 6 — No-artifact baseline metrics

Compute Dice/Precision for the clean (d=0) segmentations and merge with the
evaluation results to show artifact impact:

```bash
python scripts/cleaning/compute_no_artifact_metrics.py \
    --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exp_dir   data/wraparound_experiments/wraparound_v4 \
    --eval_dirs data/wraparound_experiments/wraparound_v4_eval_cm4/t100 \
    --out_dir   data/wraparound_experiments/wraparound_v4_eval_cm4
```

Outputs:
- `wraparound_v4_eval_cm4/no_artifact_metrics.csv` — Dice/Precision per (subject, crop, structure) at d=0
- `wraparound_v4_eval_cm4/t100/results_with_no_artifact.csv` — merged with `delta_dice_artifact` and `delta_prec_artifact` columns

---

## Step 7 — Generate paper figures

```bash
python scripts/cleaning/plot_wraparound_method_figures.py \
    --eval_root    data/wraparound_experiments/wraparound_v4_eval \
    --cm4_eval_root data/wraparound_experiments/wraparound_v4_eval_cm4

python scripts/cleaning/compute_pvalues.py
```

Figures land in `data/wraparound_experiments/wraparound_v4_eval_cm4/`.

---

## Output structure

```
data/wraparound_experiments/
  wraparound_v4/
    sXXXX/
      <crop>/
        d000_r000/                    # clean baseline crop
          mri_artifact.nii.gz
          segmentations/*.nii.gz
        d005_r025/ … d050_r100/       # 40 artifact conditions each
          mri_artifact.nii.gz
          segmentations/*.nii.gz

  wraparound_v4_eval_cm4/
    no_artifact_metrics.csv
    t100/
      results.csv                     # 25 866 rows
      results_with_no_artifact.csv
      report.md
      *.png                           # metric plots
      partial_s0175/ … partial_s0250/ # per-subject raw CSVs (intermediate)
    cm4_real_cases/                   # qualitative case studies
    cm4_visuals/                      # method explanation figures
```

---

## Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/data_prep/simulate_wraparound_artifact.py` | WM3 artifact simulation + clean baseline crops |
| `scripts/cleaning/segment_artifacts_array.sh` | SLURM version of Step 2 (optional, for cluster use) |
| `scripts/cleaning/evaluate_cleaning_methods.py` | CM4 poset-based cleaning + per-subject evaluation |
| `scripts/cleaning/merge_eval_results.py` | Merge partial CSVs → `results.csv` + plots |
| `scripts/cleaning/patch_missing_rows.py` | Append missing rows and regenerate reports |
| `scripts/cleaning/compute_no_artifact_metrics.py` | Dice/Precision at d=0 for artifact impact analysis |
| `scripts/cleaning/plot_wraparound_method_figures.py` | Generate all paper figures |
| `scripts/compute_pvalues.py` | Wilcoxon p-values for report tables |
