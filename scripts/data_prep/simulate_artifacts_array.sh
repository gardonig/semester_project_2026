#!/bin/bash
#SBATCH --job-name=simulate_v4
#SBATCH --nodelist=octopus03
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/gardonig/logs/simulate_v4_%A_%a.out
#SBATCH --error=/scratch/gardonig/logs/simulate_v4_%A_%a.err

# Job array: one task per subject (10 subjects → tasks 0-9).
# Each task runs all 40 (d, r) conditions × 3 crops for one subject.
#
# Submit:
#   sbatch scripts/data_prep/simulate_artifacts_array.sh
#
# Override output dir:
#   OUT_DIR=/scratch/gardonig/wraparound_v5 sbatch scripts/data_prep/simulate_artifacts_array.sh

PROJECT="/scratch/gardonig/Anatomy_Posets"
MRI_DIR="/scratch/gardonig/TotalsegmentatorMRI_dataset_v200"
OUT_DIR="${OUT_DIR:-/scratch/gardonig/wraparound_v4}"
PYTHON="/scratch/gardonig/totalseg_env/bin/python"

SUBJECTS=(s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250)
SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p /scratch/gardonig/logs

echo "Array task ${SLURM_ARRAY_TASK_ID}: subject=${SUBJ}"
echo "  OUT_DIR = ${OUT_DIR}"
echo "  START   = $(date)"

"${PYTHON}" "${PROJECT}/scripts/data_prep/simulate_wraparound_artifact.py" \
    --mri_dir    "${MRI_DIR}" \
    --subjects   "${SUBJ}" \
    --shift_fracs 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
    --intensities 0.25 0.50 0.75 1.00 \
    --out_dir    "${OUT_DIR}"

echo "  END     = $(date)"
