#!/bin/bash
#SBATCH --job-name=totalseg_arr
#SBATCH --nodelist=octopus03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gardonig/logs/totalseg_%A_%a.out
#SBATCH --error=/scratch/gardonig/logs/totalseg_%A_%a.err

# Array job: one task per artifact.
# Submit after generating ~/artifact_list.txt (see setup instructions below).
#
#   sbatch --array=0-<N-1> scripts/cleaning/segment_artifacts_array.sh
#
# where N = number of lines in ~/artifact_list.txt

ARTIFACT_LIST="${ARTIFACT_LIST:-/home/gardonig/artifact_list.txt}"
TOTALSEG="${TOTALSEG:-/scratch/gardonig/totalseg_env/bin/TotalSegmentator}"

mkdir -p /scratch/gardonig/logs

# Read artifact path for this task (0-indexed)
ART=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${ARTIFACT_LIST}")

if [ -z "${ART}" ]; then
    echo "ERROR: no entry at index ${SLURM_ARRAY_TASK_ID} in ${ARTIFACT_LIST}"
    exit 1
fi

SEG_DIR="$(dirname "${ART}")/segmentations"

if [ -d "${SEG_DIR}" ] && [ "$(ls -A "${SEG_DIR}" 2>/dev/null)" ]; then
    echo "SKIP [task ${SLURM_ARRAY_TASK_ID}]: already done — ${ART}"
    exit 0
fi

echo "START [task ${SLURM_ARRAY_TASK_ID}] on $(hostname) at $(date)"
echo "  ${ART}"

"${TOTALSEG}" -i "${ART}" -o "${SEG_DIR}" --fast

CODE=$?
echo "END [task ${SLURM_ARRAY_TASK_ID}] exit=${CODE} at $(date)"
exit ${CODE}
