#!/bin/bash
# Submit one sbatch segmentation job per unsegmented subject.
#
# Run this from the PROJECT ROOT on polaris (login node):
#   bash scripts/cleaning/submit_segmentation_jobs.sh
#
# Prerequisites:
#   1. Artifact MRIs are on octopus03 local scratch at DATA_ROOT/sXXXX/...
#      (see rsync instructions below)
#   2. TotalSegmentator is installed at /scratch/gardonig/totalseg_env/
#   3. /scratch/gardonig/logs/ exists on octopus03 (created automatically)

DATA_ROOT="${DATA_ROOT:-/scratch/gardonig/wraparound}"

# Subjects that still need segmentation (s0175 already done)
SUBJECTS=(s0022 s0167 s0186 s0187 s0219 s0236 s0237 s0243 s0250)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/segment_one_subject.sh"

echo "Submitting ${#SUBJECTS[@]} jobs (one per subject)"
echo "DATA_ROOT = ${DATA_ROOT}"
echo ""

for SUBJ in "${SUBJECTS[@]}"; do
    JOB_ID=$(sbatch \
        --export="SUBJECT=${SUBJ},DATA_ROOT=${DATA_ROOT}" \
        "${SBATCH_SCRIPT}" \
        | awk '{print $NF}')
    echo "  Submitted ${SUBJ} → job ${JOB_ID}"
done

echo ""
echo "Monitor with:  squeue -u gardonig"
echo "Logs at:       /scratch/gardonig/logs/totalseg_<jobid>.out"
