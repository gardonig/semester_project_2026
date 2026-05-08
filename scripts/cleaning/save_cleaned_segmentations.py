"""
Run all three cleaning methods on one artifact condition and save the cleaned
segmentations as NIfTI files for visualization in 3D Slicer.

Usage
-----
    python scripts/cleaning/save_cleaned_segmentations.py \
        --pred_dir  data/experiments/wraparound/s0175/heart_to_kidney/d025_r100/segmentations \
        --out_dir   data/experiments/wraparound/s0175/heart_to_kidney/d025_r100/cleaned \
        --poset     data/posets/empirical/totalseg_mri_empirical_poset.json \
        --com       data/structures/totalseg_v2_com.json
"""

import argparse, sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from anatomy_poset.core.io import load_poset_from_json

# Import cleaning methods from the evaluation script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_cleaning_methods import (
    method1_unidirectional, method2_symmetric, method3_middle_out_prior,
    get_si_info,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_dir",  required=True, help="Path to segmentations/ folder")
    p.add_argument("--out_dir",   required=True, help="Output directory for cleaned segs")
    p.add_argument("--poset",     default="data/posets/empirical/totalseg_mri_empirical_poset.json")
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--method",    choices=["cm1","cm2","cm3","all"], default="all")
    args = p.parse_args()

    pred_dir = Path(args.pred_dir)
    out_dir  = Path(args.out_dir)
    poset    = load_poset_from_json(args.poset)

    # Load all non-empty predictions
    preds = {}
    affine = None
    for seg_file in sorted(pred_dir.glob("*.nii.gz")):
        img = nib.load(str(seg_file))
        data = np.asarray(img.dataobj).astype(bool)
        if data.any():
            preds[seg_file.stem.replace(".nii", "")] = data
            if affine is None:
                affine = img.affine
                header = img.header

    print(f"Loaded {len(preds)} non-empty predictions")
    si_ax, si_sign = get_si_info(affine)

    methods_to_run = ["cm1", "cm2", "cm3"] if args.method == "all" else [args.method]
    method_fns = {
        "cm1": lambda: method1_unidirectional(preds, poset, si_ax, si_sign, args.threshold),
        "cm2": lambda: method2_symmetric(preds, poset, si_ax, si_sign, args.threshold),
        "cm3": lambda: method3_middle_out_prior(preds, poset, si_ax, si_sign, args.threshold),
    }

    for mname in methods_to_run:
        print(f"\nRunning {mname.upper()}...")
        cleaned, removed = method_fns[mname]()

        save_dir = out_dir / mname
        save_dir.mkdir(parents=True, exist_ok=True)

        total_removed = sum(removed.values())
        changed = sum(1 for n, v in removed.items() if v > 0)
        print(f"  {changed} structures modified, {total_removed:,} voxels removed")

        for name, mask in cleaned.items():
            out_path = save_dir / f"{name}.nii.gz"
            img_out = nib.Nifti1Image(mask.astype(np.uint8), affine, header)
            nib.save(img_out, str(out_path))

        print(f"  Saved {len(cleaned)} files → {save_dir}/")

    print("\nDone. Load in Slicer with:")
    print(f'  base = "{out_dir}"')


if __name__ == "__main__":
    main()
