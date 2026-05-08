#!/bin/bash
#SBATCH --job-name=eval_v3_landmark
#SBATCH --nodelist=octopus03
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gardonig/logs/eval_v3_%A_%a.out
#SBATCH --error=/scratch/gardonig/logs/eval_v3_%A_%a.err

PROJECT="/scratch/gardonig/Anatomy_Posets"
PYTHON="/scratch/gardonig/totalseg_env/bin/python"
DATA_DIR="/scratch/gardonig/TotalsegmentatorMRI_dataset_v200"
EXP_DIR="/scratch/gardonig/wraparound_v3"
POSET="${PROJECT}/data/posets/empirical/totalseg_mri_empirical_poset.json"
OUT_BASE="/scratch/gardonig/wraparound_v3_eval"
SUBJECTS="s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250"

THRESHOLDS=(0.95 0.99 1.00)
TAGS=(t095 t099 t100)

THRESH="${THRESHOLDS[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "${OUT_BASE}/${TAG}"

echo "Array task ${SLURM_ARRAY_TASK_ID}: threshold=${THRESH}  →  ${OUT_BASE}/${TAG}"

"${PYTHON}" "${PROJECT}/scripts/cleaning/evaluate_cleaning_methods.py" \
    --data_dir  "${DATA_DIR}" \
    --exp_dir   "${EXP_DIR}" \
    --poset     "${POSET}" \
    --subjects  ${SUBJECTS} \
    --threshold "${THRESH}" \
    --out_dir   "${OUT_BASE}/${TAG}"

echo "Done. Results at: ${OUT_BASE}/${TAG}"
