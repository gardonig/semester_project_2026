#!/bin/bash
# Regenerate all wrap-around artifacts (WM3 normalisation) for all 10 subjects.
# CPU-only — no GPU needed. Run in an srun session or as a non-GPU sbatch.
#
# Usage (on octopus03 srun session):
#   bash /scratch/gardonig/Anatomy_Posets/scripts/data_prep/simulate_all_subjects.sh

PROJECT="${PROJECT:-/scratch/gardonig/Anatomy_Posets}"
MRI_DIR="${MRI_DIR:-/scratch/gardonig/Anatomy_Posets/data/datasets/TotalsegmentatorMRI_dataset_v200}"
OUT_DIR="${OUT_DIR:-/scratch/gardonig/wraparound_v3}"
PYTHON="${PYTHON:-/scratch/gardonig/totalseg_env/bin/python}"

SUBJECTS=(s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250)

echo "Simulating WM3 artifacts → ${OUT_DIR}"
echo "MRI_DIR = ${MRI_DIR}"
echo ""

source /scratch/gardonig/totalseg_env/bin/activate

for SUBJ in "${SUBJECTS[@]}"; do
    echo "========================================"
    echo "  Subject: ${SUBJ}"
    echo "========================================"
    "${PYTHON}" "${PROJECT}/scripts/data_prep/simulate_wraparound_artifact.py" \
        --mri_dir    "${MRI_DIR}" \
        --subjects   "${SUBJ}" \
        --shift_fracs 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 \
        --intensities 0.25 0.50 0.75 1.00 \
        --out_dir    "${OUT_DIR}"
    echo ""
done

echo "All done. Artifacts at: ${OUT_DIR}"
find "${OUT_DIR}" -name "mri_artifact.nii.gz" \
  | sed 's|.*/\(s[0-9]*\)/.*|\1|' | sort | uniq -c
