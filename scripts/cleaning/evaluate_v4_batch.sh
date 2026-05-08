#!/bin/bash
#SBATCH --job-name=eval_v4
#SBATCH --nodelist=octopus03
#SBATCH --array=0-29
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gardonig/logs/eval_v4_%A_%a.out
#SBATCH --error=/scratch/gardonig/logs/eval_v4_%A_%a.err

# 30 tasks: 3 thresholds × 10 subjects (task_id = thresh_idx * 10 + subj_idx)
# Each task runs one subject at one threshold, writes partial results to its own subdir.
# After all tasks finish, run merge_eval_v4.sh to combine CSVs and regenerate plots.

PROJECT="/scratch/gardonig/Anatomy_Posets"
PYTHON="/scratch/gardonig/totalseg_env/bin/python"
DATA_DIR="/scratch/gardonig/TotalsegmentatorMRI_dataset_v200"
EXP_DIR="/scratch/gardonig/wraparound_v4"
POSET="${PROJECT}/data/posets/empirical/totalseg_mri_empirical_poset.json"
OUT_BASE="/scratch/gardonig/wraparound_v4_eval"

THRESHOLDS=(0.95 0.99 1.00)
TAGS=(t095 t099 t100)
SUBJECTS=(s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250)

THRESH_IDX=$((SLURM_ARRAY_TASK_ID / 10))
SUBJ_IDX=$((SLURM_ARRAY_TASK_ID % 10))

THRESH="${THRESHOLDS[$THRESH_IDX]}"
TAG="${TAGS[$THRESH_IDX]}"
SUBJ="${SUBJECTS[$SUBJ_IDX]}"
OUT_DIR="${OUT_BASE}/${TAG}/partial_${SUBJ}"

mkdir -p "${OUT_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID}: threshold=${THRESH}  subject=${SUBJ}  →  ${OUT_DIR}"

"${PYTHON}" "${PROJECT}/scripts/cleaning/evaluate_cleaning_methods.py" \
    --data_dir  "${DATA_DIR}" \
    --exp_dir   "${EXP_DIR}" \
    --poset     "${POSET}" \
    --subjects  "${SUBJ}" \
    --threshold "${THRESH}" \
    --out_dir   "${OUT_DIR}"

echo "Done: task ${SLURM_ARRAY_TASK_ID}"
