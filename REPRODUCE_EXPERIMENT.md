# Reproducing the Wrap-Around Artifact Experiment

This document describes every step needed to reproduce the poset-based cleaning evaluation
from scratch, in the order they were originally executed.

---

## Prerequisites

| Component | Version | Location |
|-----------|---------|----------|
| Python | 3.10+ | cluster conda env |
| TotalSegmentator | **2.13.0** | `/scratch/gardonig/totalseg_env/` |
| SLURM cluster | — | octopus03 (GPU node) |
| TotalsegmentatorMRI dataset | **v200** | `/scratch/gardonig/TotalsegmentatorMRI_dataset_v200/` |

The dataset contains 616 whole-body MRI subjects with 50 ground-truth segmentation
masks each. Download it from Zenodo (Wasserthal et al., *Radiology: AI* 2024,
https://doi.org/10.1148/ryai.230024).

Install dependencies on the cluster:
```bash
pip install nibabel numpy scipy pandas matplotlib
```

---

## Step 0 — Project setup on cluster

Clone the repository and copy the poset/CoM files to the cluster:

```bash
# on your Mac
rsync -av \
    data/posets/empirical/totalseg_mri_empirical_poset.json \
    data/structures/totalseg_mri_com_landmark.json \
    gardonig@129.132.159.29:/scratch/gardonig/Anatomy_Posets/data/posets/empirical/
```

The empirical poset and CoM atlas were precomputed on the **606 training subjects**
(616 total minus the 10 test subjects listed below). If you need to recompute them:

```bash
# CoM atlas (606 training subjects, test set excluded)
python scripts/data_prep/compute_com_landmark_normalized.py \
    --data_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exclude s0022 s0167 s0175 s0186 s0187 s0219 s0236 s0237 s0243 s0250 \
    --out     data/structures/totalseg_mri_com_landmark.json

# Empirical poset (same exclusions)
python scripts/data_prep/compute_empirical_poset.py \
    --data_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --com_json data/structures/totalseg_mri_com_landmark.json \
    --exclude  s0022 s0167 s0175 s0186 s0187 s0219 s0236 s0237 s0243 s0250 \
    --out      data/posets/empirical/totalseg_mri_empirical_poset.json
```

**Test subjects** (held out from all offline computations, used only for evaluation):
```
s0022  s0167  s0175  s0186  s0187
s0219  s0236  s0237  s0243  s0250
```
These 10 were selected because they have the highest number of non-empty GT masks,
maximising evaluation coverage.

---

## Step 1 — Simulate wrap-around artifacts

Run a SLURM array (one task per subject) on octopus03.
Each task generates all 40 artifact conditions (10 d × 4 r) for each valid crop
of that subject.

```bash
# on octopus03
sbatch scripts/data_prep/simulate_artifacts_array.sh
```

**Script:** [scripts/data_prep/simulate_artifacts_array.sh](scripts/data_prep/simulate_artifacts_array.sh)

Parameters:
- Shift fractions `d` ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}
- Ghost intensities `r` ∈ {0.25, 0.50, 0.75, 1.00}
- Crop regions: `brain_to_heart`, `heart_to_kidney`, `kidney_to_hip`
- Output: `/scratch/gardonig/wraparound_v4/`

Physics model (`simulate_wraparound_artifact.py`):
```
I_s = I + I_hat      # add ghost directly — no body-foreground masking
```
No `F = (I > 0)` masking is applied. Real MRI Fourier aliasing adds signal
everywhere including background.

Each condition is skipped if `mri_artifact.nii.gz` already exists (idempotent).

---

## Step 2 — Simulate clean baseline crops (d = 0)

Generate one clean (no-artifact) crop per subject × crop region for use as a
Dice/Precision baseline.

```bash
sbatch --job-name=baseline_seg \
    --nodelist=octopus03 \
    --gres=gpu:1 \
    --mem=16G \
    --time=00:30:00 \
    --output=/home/gardonig/baseline_seg_%j.out \
    --wrap="python /scratch/gardonig/Anatomy_Posets/scripts/data_prep/simulate_wraparound_artifact.py \
        --mri_dir /scratch/gardonig/TotalsegmentatorMRI_dataset_v200 \
        --subjects s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250 \
        --shift_fracs 0.05 \
        --intensities 0.25 \
        --out_dir /scratch/gardonig/wraparound_v4 \
        --baseline \
        --run_totalseg \
        --totalseg /scratch/gardonig/totalseg_env/bin/TotalSegmentator"
```

The `--baseline` flag saves `d000_r000/mri_artifact.nii.gz` (clean crop) and runs
TotalSegmentator on it. The dummy `--shift_fracs 0.05 --intensities 0.25` are
required by the argument parser but are skipped because those files already exist.

Output: `wraparound_v4/sXXXX/<crop>/d000_r000/segmentations/`

---

## Step 3 — Segment all artifact conditions

Build the artifact file list, then submit a SLURM array with one task per artifact:

```bash
# on octopus03 — build the list
find /scratch/gardonig/wraparound_v4 \
    -path "*/d[0-9]*/mri_artifact.nii.gz" \
    ! -path "*/d000_r000/*" \
    | sort > ~/artifact_list_v4.txt

N=$(wc -l < ~/artifact_list_v4.txt)
echo "Total artifacts: $N"    # expect 880

sbatch --array=0-$((N-1)) \
    --export=ALL,ARTIFACT_LIST=/home/gardonig/artifact_list_v4.txt \
    scripts/cleaning/segment_artifacts_array.sh
```

**Script:** [scripts/cleaning/segment_artifacts_array.sh](scripts/cleaning/segment_artifacts_array.sh)

- Model: `TotalSegmentator v2.13.0 --task total_mr --fast` (MRI model, 3 mm)
- Each task skips if `segmentations/` already exists and is non-empty (idempotent)

**Note:** 6 conditions for `s0022/heart_to_kidney` (d=0.05 all r, d=0.10 r=0.25 and r=0.50)
initially failed and were re-run manually:

```bash
TOTALSEG=/scratch/gardonig/totalseg_env/bin/TotalSegmentator
BASE=/scratch/gardonig/wraparound_v4/s0022/heart_to_kidney

for tag in d005_r025 d005_r050 d005_r075 d005_r100 d010_r025 d010_r050; do
    $TOTALSEG -i $BASE/$tag/mri_artifact.nii.gz \
              -o $BASE/$tag/segmentations \
              --task total_mr --fast
done
```

After re-running, the total is **896 segmentation directories**:
- 874 artifact conditions (880 − 6 initial failures, all later fixed)
- 22 clean baselines (d000_r000)

---

## Step 4 — Poset-based cleaning evaluation

Run 30 parallel jobs: 3 thresholds × 10 subjects.

```bash
sbatch scripts/cleaning/evaluate_v4_batch.sh
```

**Script:** [scripts/cleaning/evaluate_v4_batch.sh](scripts/cleaning/evaluate_v4_batch.sh)

- Thresholds: 0.95, 0.99, 1.00
- Each task writes to `wraparound_v4_eval/<tag>/partial_<subject>/results.csv`

After all 30 tasks complete, merge the partial CSVs:

```bash
python scripts/cleaning/merge_eval_results.py \
    --eval_base /scratch/gardonig/wraparound_v4_eval \
    --tags t095 t099 t100
```

This produces `results.csv`, `report.md`, and plots for each threshold.

---

## Step 5 — Patch the 6 missing conditions

After re-running TotalSegmentator on the 6 missing s0022 conditions (Step 3),
run a partial evaluation locally and patch the results:

```bash
# Partial eval for the 2 affected d values at all 3 thresholds
for THRESH in 1.00 0.99 0.95; do
    TAG=$(python3 -c "t='$THRESH'; print('t'+t.replace('.','')[:3])")
    python scripts/cleaning/evaluate_cleaning_methods.py \
        --data_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \
        --exp_dir  data/experiments/wraparound_v4 \
        --poset    data/posets/empirical/totalseg_mri_empirical_poset.json \
        --subjects s0022 \
        --d_fracs 0.05 0.10 \
        --r_vals 0.25 0.50 0.75 1.00 \
        --threshold $THRESH \
        --out_dir  data/experiments/wraparound_v4_eval/${TAG}/partial_fix_s0022
done

# Append new rows (deduplicates automatically) and regenerate reports
python scripts/cleaning/patch_missing_rows.py \
    --eval_dir data/experiments/wraparound_v4_eval
```

**Script:** [scripts/cleaning/patch_missing_rows.py](scripts/cleaning/patch_missing_rows.py)

Final row counts after patching: **25 866 rows** per threshold.

---

## Step 6 — Baseline metrics (artifact vs clean)

Compute Dice/Precision for the clean (d=0) segmentations and merge into a
combined CSV showing how much the artifact degrades TotalSegmentator:

```bash
python scripts/cleaning/compute_baseline_metrics.py \
    --data_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exp_dir  data/experiments/wraparound_v4 \
    --eval_dir data/experiments/wraparound_v4_eval/t100 \
    --out_dir  data/experiments/wraparound_v4_eval
```

Outputs:
- `wraparound_v4_eval/baseline_metrics.csv` — Dice/Precision per (subject, crop, structure) at d=0
- `wraparound_v4_eval/results_with_baseline.csv` — merged with `delta_dice_artifact` and `delta_prec_artifact` columns

---

## Output structure

```
data/experiments/wraparound_v4/
  sXXXX/
    <crop>/
      d000_r000/                    # clean baseline crop
        mri_artifact.nii.gz
        segmentations/*.nii.gz
      d005_r025/ … d050_r100/       # 40 artifact conditions
        mri_artifact.nii.gz
        segmentations/*.nii.gz

data/experiments/wraparound_v4_eval/
  baseline_metrics.csv
  results_with_baseline.csv
  t095/
    results.csv                     # 25 866 rows
    report.md
    plots/
  t099/
    results.csv
    report.md
    plots/
  t100/
    results.csv
    report.md
    plots/
```

---

## Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/data_prep/simulate_wraparound_artifact.py` | WM3 artifact simulation + clean baseline crops |
| `scripts/data_prep/simulate_artifacts_array.sh` | SLURM array: simulate all 10 subjects |
| `scripts/cleaning/segment_artifacts_array.sh` | SLURM array: TotalSegmentator on all artifacts |
| `scripts/cleaning/evaluate_v4_batch.sh` | SLURM array: poset-based cleaning evaluation (30 tasks) |
| `scripts/cleaning/merge_eval_results.py` | Merge partial CSVs → results.csv + plots |
| `scripts/cleaning/patch_missing_rows.py` | Append missing rows + regenerate reports |
| `scripts/cleaning/evaluate_cleaning_methods.py` | Poset-based cleaning implementation + evaluation loop |
| `scripts/cleaning/compute_baseline_metrics.py` | Dice/Precision at d=0 for artifact impact analysis |
