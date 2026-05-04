#!/bin/bash
#SBATCH --job-name=totalseg_%j
#SBATCH --nodelist=octopus03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/gardonig/logs/totalseg_%j.out
#SBATCH --error=/scratch/gardonig/logs/totalseg_%j.err

# Usage:
#   sbatch --export=SUBJECT=s0236,DATA_ROOT=/scratch/gardonig/wraparound \
#          scripts/cleaning/segment_one_subject.sh
#
# SUBJECT   : subject ID, e.g. s0236
# DATA_ROOT : root on octopus03 local scratch containing sXXXX/ subdirectories
#             (default: /scratch/gardonig/wraparound)

set -euo pipefail

SUBJECT="${SUBJECT:?ERROR: set SUBJECT before submitting}"
DATA_ROOT="${DATA_ROOT:-/scratch/gardonig/wraparound}"
TOTALSEG="${TOTALSEG:-/scratch/gardonig/totalseg_env/bin/TotalSegmentator}"

SUBJ_DIR="${DATA_ROOT}/${SUBJECT}"
LOG_PREFIX="[${SUBJECT}]"

echo "${LOG_PREFIX} Starting on $(hostname) at $(date)"
echo "${LOG_PREFIX} DATA_ROOT = ${DATA_ROOT}"
echo "${LOG_PREFIX} TotalSegmentator = ${TOTALSEG}"

if [ ! -d "${SUBJ_DIR}" ]; then
    echo "${LOG_PREFIX} ERROR: subject directory not found: ${SUBJ_DIR}"
    exit 1
fi

mkdir -p /scratch/gardonig/logs

ARTIFACTS=$(find "${SUBJ_DIR}" -name "mri_artifact.nii.gz" | sort)
TOTAL=$(echo "${ARTIFACTS}" | grep -c "." || true)
echo "${LOG_PREFIX} Found ${TOTAL} artifact volumes"

COUNT=0
SKIPPED=0
FAILED=0

for ART in ${ARTIFACTS}; do
    COUNT=$((COUNT + 1))
    SEG_DIR="$(dirname "${ART}")/segmentations"

    if [ -d "${SEG_DIR}" ] && [ "$(ls -A "${SEG_DIR}" 2>/dev/null)" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "${LOG_PREFIX} [${COUNT}/${TOTAL}] SKIP (already done): ${ART#${DATA_ROOT}/}"
        continue
    fi

    echo "${LOG_PREFIX} [${COUNT}/${TOTAL}] Segmenting: ${ART#${DATA_ROOT}/}"
    if "${TOTALSEG}" -i "${ART}" -o "${SEG_DIR}" --fast 2>&1; then
        echo "${LOG_PREFIX} [${COUNT}/${TOTAL}] OK"
    else
        FAILED=$((FAILED + 1))
        echo "${LOG_PREFIX} [${COUNT}/${TOTAL}] FAILED: ${ART#${DATA_ROOT}/}"
    fi
done

echo ""
echo "${LOG_PREFIX} Done at $(date)"
echo "${LOG_PREFIX} Total=${TOTAL}  Skipped=${SKIPPED}  Failed=${FAILED}  Processed=$((COUNT - SKIPPED - FAILED))"
