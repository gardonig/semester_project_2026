#!/bin/bash
#SBATCH --job-name=clean_cm3_arr
#SBATCH --nodelist=octopus03
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/gardonig/logs/clean_cm3_%A_%a.out
#SBATCH --error=/scratch/gardonig/logs/clean_cm3_%A_%a.err

SEGDIRS_LIST="${SEGDIRS_LIST:-/home/gardonig/segdirs_list.txt}"
THRESHOLD="${THRESHOLD:-0.95}"
CLEAN_SUFFIX="${CLEAN_SUFFIX:-cleaned}"
PROJECT="/scratch/gardonig/Anatomy_Posets"
POSET="${PROJECT}/data/structures/totalseg_v2_empirical_poset.json"
COM="${PROJECT}/data/structures/totalseg_v2_com.json"
PYTHON="/scratch/gardonig/totalseg_env/bin/python"

SEG_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SEGDIRS_LIST}")
[ -z "${SEG_DIR}" ] && exit 0

CLEAN_DIR="$(dirname "${SEG_DIR}")/${CLEAN_SUFFIX}"

if [ -d "${CLEAN_DIR}" ] && [ "$(ls -A "${CLEAN_DIR}" 2>/dev/null)" ]; then
    echo "SKIP: already cleaned"
    exit 0
fi

if [ -z "$(ls -A "${SEG_DIR}" 2>/dev/null)" ]; then
    echo "SKIP: empty segmentations dir"
    exit 0
fi

"${PYTHON}" "${PROJECT}/scripts/cleaning/save_cleaned_segmentations.py" \
    --pred_dir  "${SEG_DIR}" \
    --out_dir   "${CLEAN_DIR}" \
    --poset     "${POSET}" \
    --com       "${COM}" \
    --method    cm3 \
    --threshold "${THRESHOLD}"
