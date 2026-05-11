#!/bin/bash
#SBATCH --job-name=erosion_sweep
#SBATCH --nodelist=octopus03
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gardonig/logs/erosion_sweep_%A_%a.out
#SBATCH --error=/scratch/gardonig/logs/erosion_sweep_%A_%a.err

# Array layout: 10 tasks, one per subject.
# Each task runs the full radius sweep (lcc_only + radii 1-5) for its subject.
# Output structure:
#   OUT_BASE/partial_s0175/lcc_only/results.csv
#   OUT_BASE/partial_s0175/radius_1/results.csv
#   ...
#   OUT_BASE/partial_s0175/radius_5/results.csv
# After all tasks complete, run merge_erosion_sweep.py to combine and plot.

PROJECT="/scratch/gardonig/Anatomy_Posets"
PYTHON="/scratch/gardonig/totalseg_env/bin/python"
DATA_DIR="/scratch/gardonig/TotalsegmentatorMRI_dataset_v200"
EXP_DIR="/scratch/gardonig/wraparound_v4"
OUT_BASE="/scratch/gardonig/wraparound_v4_eval/erosion_baseline"

SUBJECTS=(s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250)
SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${OUT_BASE}/partial_${SUBJ}"

mkdir -p "${OUT_DIR}"
mkdir -p "/scratch/gardonig/logs"

echo "=================================================="
echo "Task ${SLURM_ARRAY_TASK_ID}: subject=${SUBJ}"
echo "Output: ${OUT_DIR}"
echo "=================================================="

"${PYTHON}" "${PROJECT}/scripts/cleaning/evaluate_erosion_baseline.py" \
    --data_dir "${DATA_DIR}" \
    --exp_dir  "${EXP_DIR}" \
    --subject  "${SUBJ}" \
    --out_dir  "${OUT_DIR}" \
    --radii    1 2

echo "Done: ${SUBJ} (task ${SLURM_ARRAY_TASK_ID})"
