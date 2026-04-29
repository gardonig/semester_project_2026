#!/bin/bash
# Run TotalSegmentator on all mri_artifact.nii.gz files under a results directory.
#
# Usage (on octopus03 via srun):
#   bash ~/run_totalseg_on_artifacts.sh ~/wraparound_results/s0175 \
#        /scratch/gardonig/totalseg_env/bin/TotalSegmentator

RESULTS_DIR="$1"
TOTALSEG_BIN="$2"

if [ -z "$RESULTS_DIR" ] || [ -z "$TOTALSEG_BIN" ]; then
    echo "Usage: $0 <results_dir> <totalseg_binary>"
    exit 1
fi

# Find all mri_artifact.nii.gz files
ARTIFACTS=$(find "$RESULTS_DIR" -name "mri_artifact.nii.gz" | sort)
TOTAL=$(echo "$ARTIFACTS" | wc -l)
echo "Found $TOTAL artifact volumes to segment"

COUNT=0
for ART in $ARTIFACTS; do
    COUNT=$((COUNT + 1))
    SEG_DIR="$(dirname "$ART")/segmentations"

    # Skip if already done
    if [ -d "$SEG_DIR" ] && [ "$(ls -A "$SEG_DIR")" ]; then
        echo "[$COUNT/$TOTAL] SKIP (already done): $ART"
        continue
    fi

    echo "[$COUNT/$TOTAL] Running TotalSegmentator: $ART"
    "$TOTALSEG_BIN" -i "$ART" -o "$SEG_DIR" --fast
    if [ $? -eq 0 ]; then
        echo "[$COUNT/$TOTAL] OK"
    else
        echo "[$COUNT/$TOTAL] FAILED"
    fi
done

echo "All done."
