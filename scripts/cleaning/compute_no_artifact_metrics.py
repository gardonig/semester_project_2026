"""
Compute Dice and Precision for the no-artifact (d=0, r=0) segmentations and
merge with existing results.csv files to produce:

  out_dir/
    no_artifact_metrics.csv       — one row per (subject, crop, structure)

  For each --eval_dir given:
    <eval_dir>/
      results_with_no_artifact.csv — results.csv + dice_no_artifact, prec_no_artifact,
                                     delta_dice_artifact, delta_prec_artifact columns

Usage
-----
    # Single threshold dir:
    python scripts/cleaning/compute_no_artifact_metrics.py \\
        --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \\
        --exp_dir   data/experiments/wraparound_v4 \\
        --eval_dirs data/experiments/wraparound_v4_eval/t100 \\
        --out_dir   data/experiments/wraparound_v4_eval

    # All three threshold dirs at once:
    python scripts/cleaning/compute_no_artifact_metrics.py \\
        --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \\
        --exp_dir   data/experiments/wraparound_v4 \\
        --eval_dirs data/experiments/wraparound_v4_eval/t095 \\
                    data/experiments/wraparound_v4_eval/t099 \\
                    data/experiments/wraparound_v4_eval/t100 \\
        --out_dir   data/experiments/wraparound_v4_eval
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import aff2axcodes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_si_info(affine) -> Tuple[int, int]:
    codes = aff2axcodes(affine)
    si_ax = next(i for i, c in enumerate(codes) if c in ("S", "I"))
    si_sign = +1 if codes[si_ax] == "S" else -1
    return si_ax, si_sign


def axis_extent(mask: np.ndarray, si_ax: int) -> Optional[Tuple[int, int]]:
    other = tuple(ax for ax in range(mask.ndim) if ax != si_ax)
    proj = mask.any(axis=other)
    idx = np.where(proj)[0]
    return (int(idx.min()), int(idx.max())) if len(idx) > 0 else None


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return float(2.0 * (p & g).sum() / denom) if denom > 0 else 1.0


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum())
    pp = int(p.sum())
    return tp / pp if pp > 0 else 1.0


def crop_gt(gt_full: np.ndarray, si_ax: int, crop_lo: int, crop_hi: int) -> np.ndarray:
    slices = [slice(None)] * gt_full.ndim
    slices[si_ax] = slice(crop_lo, crop_hi)
    return gt_full[tuple(slices)]


CROPS = [
    ("brain_to_heart",  "brain",       "heart"),
    ("heart_to_kidney", "heart",       "kidney_left"),
    ("kidney_to_hip",   "kidney_left", "hip_left"),
]
FALLBACKS = {"kidney_left": "kidney_right", "hip_left": "hip_right"}
MARGIN = 5

SUBJECTS = [
    "s0175", "s0236", "s0219", "s0187", "s0022",
    "s0167", "s0186", "s0237", "s0243", "s0250",
]


def compute_crop_window(gt_dir: Path, anchors: Tuple[str, str],
                        si_ax: int, full_height: int) -> Tuple[int, int]:
    lo_vals, hi_vals = [], []
    for name in anchors:
        p = gt_dir / f"{name}.nii.gz"
        if not p.exists():
            name = FALLBACKS.get(name, name)
            p = gt_dir / f"{name}.nii.gz"
        if not p.exists():
            continue
        d = np.asarray(nib.load(str(p)).dataobj).astype(bool)
        ext = axis_extent(d, si_ax)
        if ext:
            lo_vals.append(ext[0])
            hi_vals.append(ext[1])
    if not lo_vals:
        return 0, full_height
    lo = max(0, min(lo_vals) - MARGIN)
    hi = min(full_height, max(hi_vals) + MARGIN)
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir",  required=True, type=Path,
                   help="TotalsegmentatorMRI_dataset_v200 root (has GT segmentations)")
    p.add_argument("--exp_dir",   required=True, type=Path,
                   help="wraparound_v4 root (has d000_r000 subdirs)")
    p.add_argument("--eval_dirs", required=True, type=Path, nargs="+",
                   help="One or more eval dirs with results.csv "
                        "(e.g. wraparound_v4_eval/t095 wraparound_v4_eval/t099 wraparound_v4_eval/t100). "
                        "results_with_no_artifact.csv is written into each.")
    p.add_argument("--out_dir",   required=True, type=Path,
                   help="Where to write no_artifact_metrics.csv (shared across thresholds)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: compute no-artifact metrics (threshold-independent)
    # ------------------------------------------------------------------
    rows: List[dict] = []

    for subj in SUBJECTS:
        gt_dir = args.data_dir / subj / "segmentations"
        mri_full = nib.load(str(args.data_dir / subj / "mri.nii.gz"))
        si_ax, si_sign = get_si_info(mri_full.affine)
        full_height = mri_full.shape[si_ax]

        gt_names = {p.stem.replace(".nii", "") for p in gt_dir.glob("*.nii.gz")}

        print(f"\n{'='*60}\n  {subj}\n{'='*60}")

        for crop_name, anc_a, anc_b in CROPS:
            no_artifact_dir = args.exp_dir / subj / crop_name / "d000_r000" / "segmentations"
            if not no_artifact_dir.exists():
                print(f"  [{crop_name}] d000_r000 missing — skipping")
                continue

            crop_lo, crop_hi = compute_crop_window(
                gt_dir, (anc_a, anc_b), si_ax, full_height
            )
            print(f"  [{crop_name}] lo={crop_lo} hi={crop_hi}")

            preds: Dict[str, np.ndarray] = {}
            for pred_file in sorted(no_artifact_dir.glob("*.nii.gz")):
                name = pred_file.stem.replace(".nii", "")
                data = np.asarray(nib.load(str(pred_file)).dataobj).astype(bool)
                if data.any():
                    preds[name] = data

            if not preds:
                print(f"  [{crop_name}] no predictions — skipping")
                continue

            for name, pred in sorted(preds.items()):
                if name not in gt_names:
                    continue
                gt_full = np.asarray(
                    nib.load(str(gt_dir / f"{name}.nii.gz")).dataobj
                ).astype(bool)
                gt_crop = crop_gt(gt_full, si_ax, crop_lo, crop_hi)

                if pred.shape != gt_crop.shape:
                    print(f"    [warn] {name}: pred {pred.shape} != gt {gt_crop.shape}, skipping")
                    continue
                if not gt_crop.any():
                    continue

                rows.append({
                    "subject":          subj,
                    "crop":             crop_name,
                    "structure":        name,
                    "dice_no_artifact": round(dice(pred, gt_crop), 5),
                    "prec_no_artifact": round(precision(pred, gt_crop), 5),
                })

            n = len([r for r in rows if r["subject"] == subj and r["crop"] == crop_name])
            print(f"    {n} structures evaluated")

    # Save shared no-artifact CSV
    no_artifact_path = args.out_dir / "no_artifact_metrics.csv"
    fieldnames = ["subject", "crop", "structure", "dice_no_artifact", "prec_no_artifact"]
    with open(no_artifact_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {no_artifact_path}")

    df_no_artifact = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Step 2: merge with results.csv from each eval_dir
    # ------------------------------------------------------------------
    for eval_dir in args.eval_dirs:
        results_path = eval_dir / "results.csv"
        if not results_path.exists():
            print(f"\n[skip] results.csv not found at {results_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Merging → {eval_dir.name}")
        print(f"{'='*60}")

        df_res = pd.read_csv(results_path)
        df_merged = df_res.merge(
            df_no_artifact, on=["subject", "crop", "structure"], how="left"
        )

        # How much worse did TotalSegmentator get due to the artifact?
        # Positive delta_dice_artifact means the artifact IMPROVED Dice (rare edge case).
        # Negative means the artifact hurt performance.
        df_merged["delta_dice_artifact"] = (
            df_merged["dice_before"] - df_merged["dice_no_artifact"]
        ).round(5)
        df_merged["delta_prec_artifact"] = (
            df_merged["precision_before"] - df_merged["prec_no_artifact"]
        ).round(5)

        merged_path = eval_dir / "results_with_no_artifact.csv"
        df_merged.to_csv(merged_path, index=False)
        print(f"Saved merged results → {merged_path}")

        # Summary stats
        df = df_merged.dropna(subset=["dice_no_artifact"])
        print(f"\nARTIFACT IMPACT  (artifact condition vs no-artifact condition)")
        print(f"Negative = artifact hurts TotalSegmentator\n")

        print(f"  Overall mean ΔDice:      {df['delta_dice_artifact'].mean():+.4f}")
        print(f"  Overall mean ΔPrecision: {df['delta_prec_artifact'].mean():+.4f}")
        print(f"  Fraction degraded (Dice): {(df['delta_dice_artifact'] < -0.001).mean():.1%}")

        print(f"\n  By crop:")
        for crop, grp in df.groupby("crop"):
            print(f"    {crop:<22} ΔDice={grp['delta_dice_artifact'].mean():+.4f}  "
                  f"ΔPrec={grp['delta_prec_artifact'].mean():+.4f}  "
                  f"Deg={(grp['delta_dice_artifact'] < -0.001).mean():.1%}")

        print(f"\n  By ghost intensity (r):")
        for r_val, grp in df.groupby("r_val"):
            print(f"    r={r_val:<4}  ΔDice={grp['delta_dice_artifact'].mean():+.4f}  "
                  f"ΔPrec={grp['delta_prec_artifact'].mean():+.4f}")

        print(f"\n  By shift fraction (d):")
        for d_frac, grp in df.groupby("d_frac"):
            print(f"    d={d_frac:.2f}  ΔDice={grp['delta_dice_artifact'].mean():+.4f}  "
                  f"ΔPrec={grp['delta_prec_artifact'].mean():+.4f}")

        print(f"\n  Worst 10 structures (mean ΔDice, artifact vs no artifact):")
        per_struct = df.groupby("structure")["delta_dice_artifact"].mean().sort_values()
        print(per_struct.head(10).to_string())


if __name__ == "__main__":
    main()
