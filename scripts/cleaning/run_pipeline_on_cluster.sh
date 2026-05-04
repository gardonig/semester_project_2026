#!/bin/bash
# Full wrap-around pipeline on the cluster:
#   1. TotalSegmentator (GPU) on all new mri_artifact.nii.gz files
#   2. evaluate_cleaning_methods.py (CPU) on all completed conditions
#
# Run this inside a GPU srun session on the cluster:
#   srun --nodelist=octopus03 --gres=gpu:1 --pty bash
#   bash ~/run_pipeline_on_cluster.sh
#
# Environment assumptions:
#   - TOTALSEG_BIN: path to TotalSegmentator binary (default below)
#   - RESULTS_DIR:  where wraparound artifact volumes live (default below)
#   - PROJECT_DIR:  repo root on scratch (default below)

TOTALSEG_BIN="${TOTALSEG_BIN:-/scratch/gardonig/totalseg_env/bin/TotalSegmentator}"
RESULTS_DIR="${RESULTS_DIR:-/scratch/gardonig/wraparound_results/data/experiments/wraparound/s0175}"
PROJECT_DIR="${PROJECT_DIR:-/scratch/gardonig/Anatomy_Posets}"
SUBJECT="s0175"

# --- Step 1: TotalSegmentator on all pending artifact volumes ---
echo "============================================================"
echo "STEP 1: TotalSegmentator"
echo "============================================================"

ARTIFACTS=$(find "$RESULTS_DIR" -name "mri_artifact.nii.gz" | sort)
TOTAL=$(echo "$ARTIFACTS" | wc -l | tr -d ' ')
echo "Found $TOTAL artifact volumes"

COUNT=0
for ART in $ARTIFACTS; do
    COUNT=$((COUNT + 1))
    SEG_DIR="$(dirname "$ART")/segmentations"
    if [ -d "$SEG_DIR" ] && [ "$(ls -A "$SEG_DIR" 2>/dev/null)" ]; then
        echo "[$COUNT/$TOTAL] SKIP: $ART"
        continue
    fi
    echo "[$COUNT/$TOTAL] Segmenting: $ART"
    "$TOTALSEG_BIN" -i "$ART" -o "$SEG_DIR" --fast
    if [ $? -eq 0 ]; then
        echo "[$COUNT/$TOTAL] OK"
    else
        echo "[$COUNT/$TOTAL] FAILED"
    fi
done

echo "TotalSegmentator done."

# --- Step 2: Cleaning evaluation (M1, M2, M3) ---
echo ""
echo "============================================================"
echo "STEP 2: Cleaning evaluation (M1 / M2 / M3)"
echo "============================================================"

# Activate the project Python env (adjust if your env path differs)
source /scratch/gardonig/totalseg_env/bin/activate

cd "$PROJECT_DIR" || { echo "ERROR: PROJECT_DIR not found: $PROJECT_DIR"; exit 1; }

python scripts/cleaning/evaluate_cleaning_methods.py \
    --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
    --exp_dir   "$RESULTS_DIR/.." \
    --poset     data/structures/totalseg_v2_empirical_poset.json \
    --com       data/structures/totalseg_v2_com.json \
    --subject   "$SUBJECT" \
    --out_dir   data/experiments/wraparound_cleaning_eval_finer_d \
    --threshold 0.95

echo ""
echo "All done. Sync results back to Mac with:"
echo "  rsync -av --exclude='mri_artifact.nii.gz' \\"
echo "    gardonig@129.132.159.29:/scratch/gardonig/wraparound_results/ \\"
echo "    data/experiments/wraparound/"
