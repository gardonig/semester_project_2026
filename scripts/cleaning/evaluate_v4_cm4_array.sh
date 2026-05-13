#!/bin/bash
#SBATCH --job-name=eval_v4_cm4
#SBATCH --partition=cpu.all
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/gardonig/logs/eval_v4_cm4_%A_%a.out
#SBATCH --error=/scratch/gardonig/logs/eval_v4_cm4_%A_%a.err

PROJECT="/scratch/gardonig/Anatomy_Posets"
PYTHON="/scratch/gardonig/totalseg_env/bin/python"
DATA_DIR="/scratch/gardonig/TotalsegmentatorMRI_dataset_v200"
EXP_DIR="/scratch/gardonig/wraparound_v4"
POSET="${PROJECT}/data/posets/empirical/totalseg_mri_empirical_poset.json"

# Keep tag=t100 so merge script maps threshold correctly.
OUT_BASE="/scratch/gardonig/wraparound_v4_eval_cm4"
TAG="t100"
THRESH="1.00"

SUBJECTS=(s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250)
SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${OUT_BASE}/${TAG}/partial_${SUBJ}"

mkdir -p "${OUT_DIR}" /scratch/gardonig/logs

echo "Task ${SLURM_ARRAY_TASK_ID}: method=cm4 threshold=${THRESH} subject=${SUBJ} -> ${OUT_DIR}"

"${PYTHON}" "${PROJECT}/scripts/cleaning/evaluate_cleaning_methods.py" \
  --data_dir  "${DATA_DIR}" \
  --exp_dir   "${EXP_DIR}" \
  --poset     "${POSET}" \
  --subjects  "${SUBJ}" \
  --threshold "${THRESH}" \
  --method    cm4 \
  --out_dir   "${OUT_DIR}"

echo "Done task ${SLURM_ARRAY_TASK_ID}"
