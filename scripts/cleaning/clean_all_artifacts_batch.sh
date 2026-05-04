#!/bin/bash
#SBATCH --job-name=clean_m3
#SBATCH --nodelist=octopus03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/gardonig/logs/clean_m3_%j.out
#SBATCH --error=/scratch/gardonig/logs/clean_m3_%j.err
#
# Run M3 cleaning on every segmented artifact condition and save cleaned
# NIfTI files to a cleaned/ sibling directory.
#
# Submit from polaris:
#   sbatch /scratch/gardonig/Anatomy_Posets/scripts/cleaning/clean_all_artifacts_batch.sh

DATA_ROOT="${DATA_ROOT:-/scratch/gardonig/wraparound_all}"
PROJECT="${PROJECT:-/scratch/gardonig/Anatomy_Posets}"
POSET="${PROJECT}/data/structures/totalseg_v2_empirical_poset.json"
COM="${PROJECT}/data/structures/totalseg_v2_com.json"
PYTHON="${PYTHON:-/scratch/gardonig/totalseg_env/bin/python}"

echo "START clean_m3 on $(hostname) at $(date)"
echo "DATA_ROOT = ${DATA_ROOT}"

source /scratch/gardonig/totalseg_env/bin/activate

SEG_DIRS=$(find "${DATA_ROOT}" -type d -name "segmentations" | sort)
TOTAL=$(echo "${SEG_DIRS}" | grep -c "." || true)
echo "Found ${TOTAL} segmentation directories"

COUNT=0
SKIPPED=0
FAILED=0

for SEG_DIR in ${SEG_DIRS}; do
    COUNT=$((COUNT + 1))
    COND_DIR="$(dirname "${SEG_DIR}")"
    CLEAN_DIR="${COND_DIR}/cleaned"

    # Skip if cleaned/ already has files
    if [ -d "${CLEAN_DIR}" ] && [ "$(ls -A "${CLEAN_DIR}" 2>/dev/null)" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if segmentations dir is empty (failed TotalSegmentator run)
    if [ -z "$(ls -A "${SEG_DIR}" 2>/dev/null)" ]; then
        echo "  [${COUNT}/${TOTAL}] SKIP empty: ${SEG_DIR#${DATA_ROOT}/}"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "  [${COUNT}/${TOTAL}] Cleaning: ${COND_DIR#${DATA_ROOT}/}"

    "${PYTHON}" "${PROJECT}/scripts/cleaning/save_cleaned_segmentations.py" \
        --pred_dir "${SEG_DIR}" \
        --out_dir  "${CLEAN_DIR}" \
        --poset    "${POSET}" \
        --com      "${COM}" \
        --method   m3
    CODE=$?

    if [ ${CODE} -ne 0 ]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (exit=${CODE}): ${COND_DIR#${DATA_ROOT}/}"
    fi
done

echo ""
echo "Done at $(date)"
echo "Total=${TOTAL}  Skipped=${SKIPPED}  Failed=${FAILED}  Cleaned=$((COUNT - SKIPPED - FAILED))"
