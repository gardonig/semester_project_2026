#!/bin/bash
#SBATCH --job-name=eval_v3
#SBATCH --nodelist=octopus03
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gardonig/logs/eval_v3_%j.out
#SBATCH --error=/scratch/gardonig/logs/eval_v3_%j.err

PROJECT="/scratch/gardonig/Anatomy_Posets"
PYTHON="/scratch/gardonig/totalseg_env/bin/python"

"${PYTHON}" "${PROJECT}/scripts/cleaning/evaluate_cleaning_methods.py" \
    --data_dir  /scratch/gardonig/TotalsegmentatorMRI_dataset_v200 \
    --exp_dir   /scratch/gardonig/wraparound_v3 \
    --poset     "${PROJECT}/data/structures/totalseg_v2_empirical_poset.json" \
    --com       "${PROJECT}/data/structures/totalseg_v2_com.json" \
    --subjects  s0175 s0236 s0219 s0187 s0022 s0167 s0186 s0237 s0243 s0250 \
    --out_dir   /scratch/gardonig/wraparound_v3_eval
