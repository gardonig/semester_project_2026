"""
Truncated FOV Experiment
========================

Simulates truncated field-of-view scans by cropping the superior portion of
v201 CT scans, running TotalSegmentator on the cropped CTs, then applying
poset constraint cleaning and evaluating Dice improvement vs the original GT.

When a scan is cropped at the top, TotalSegmentator predicts stray fragments
of superior structures (upper ribs, upper vertebrae, clavicles) near the new
top boundary. The CC-based cleaning should detect and remove these.

Usage
-----
    python scripts/truncated_fov_experiment.py \
        --data_dir  /scratch/gardonig/totalseg_v201 \
        --out_dir   /scratch/gardonig/truncated_fov_experiment \
        --poset     ~/llm_claude_v157.json \
        --subjects  s0000 s0001 s0002 s0003 s0004 \
        --crop_frac 0.25
"""

from __future__ import annotations

import argparse
import os
import subprocess
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy.ndimage import label as cc_label

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from anatomy_poset.core.io import load_poset_from_json, PosetFromJson


# ---------------------------------------------------------------------------
# Shared helpers (copied from postprocessing script)
# ---------------------------------------------------------------------------

def axis_sign_map(affine):
    codes = aff2axcodes(affine)
    result = {}
    for vox_ax, code in enumerate(codes):
        if   code == 'S': result['vertical'] = (vox_ax, +1)
        elif code == 'I': result['vertical'] = (vox_ax, -1)
    return result


def axis_extent(mask, vox_ax):
    other = tuple(ax for ax in range(mask.ndim) if ax != vox_ax)
    proj  = mask.any(axis=other)
    idx   = np.where(proj)[0]
    return (int(idx.min()), int(idx.max())) if len(idx) > 0 else None


def largest_connected_component(mask):
    if not mask.any():
        return mask.copy()
    labeled, n = cc_label(mask)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(bool)


def dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return float(2.0 * (p & g).sum() / denom) if denom > 0 else 1.0


def apply_constraints_gt_free(predictions, poset, asmap):
    constrained    = {n: m.copy() for n, m in predictions.items()}
    voxels_removed = {n: 0 for n in predictions}

    if 'vertical' not in asmap:
        return constrained, voxels_removed
    vox_ax, sign = asmap['vertical']

    for i, si in enumerate(poset.structures):
        for j, sj in enumerate(poset.structures):
            if i == j:
                continue
            cell = poset.matrix_vertical[i][j]
            if cell is None or cell != 1:
                continue
            name_i, name_j = si.name, sj.name
            if name_i not in constrained or name_j not in constrained:
                continue
            mask_i = constrained[name_i]
            mask_j = constrained[name_j]
            if not mask_i.any() or not mask_j.any():
                continue

            ext_j = axis_extent(largest_connected_component(mask_j), vox_ax)
            if ext_j is None:
                continue
            min_j, max_j = ext_j

            labeled_i, n_i = cc_label(mask_i)
            if n_i == 0:
                continue
            sizes_i = np.bincount(labeled_i.ravel())
            sizes_i[0] = 0
            lcc_label_i = int(sizes_i.argmax())

            for comp_label in range(1, n_i + 1):
                if comp_label == lcc_label_i:
                    continue
                comp = labeled_i == comp_label
                ext_c = axis_extent(comp, vox_ax)
                if ext_c is None:
                    continue
                c_min, c_max = ext_c
                if sign == +1:
                    entirely_violated = c_max < min_j
                else:
                    entirely_violated = c_min > max_j
                if entirely_violated:
                    voxels_removed[name_i] += int(comp.sum())
                    constrained[name_i][comp] = False

    return constrained, voxels_removed


# ---------------------------------------------------------------------------
# Crop CT
# ---------------------------------------------------------------------------

def crop_superior(ct_path: Path, out_path: Path, crop_frac: float) -> None:
    """Remove the top crop_frac of slices along the superior axis."""
    img  = nib.load(str(ct_path))
    data = np.asarray(img.dataobj)
    affine = img.affine
    codes = aff2axcodes(affine)

    # Find which voxel axis is superior-inferior
    si_ax = next((i for i, c in enumerate(codes) if c in ('S', 'I')), 2)
    n_slices = data.shape[si_ax]
    n_crop   = int(n_slices * crop_frac)

    slices = [slice(None)] * data.ndim
    if codes[si_ax] == 'S':
        # increasing index = more superior → crop from the end
        slices[si_ax] = slice(0, n_slices - n_crop)
    else:
        # increasing index = more inferior → crop from the start
        slices[si_ax] = slice(n_crop, n_slices)

    cropped = data[tuple(slices)]

    # Update affine origin if we removed slices from the start
    new_affine = affine.copy()
    if codes[si_ax] == 'I':
        new_affine[:3, 3] += affine[:3, si_ax] * n_crop

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(cropped.astype(np.int16), new_affine), str(out_path))
    print(f"    Cropped {n_crop}/{n_slices} superior slices → {out_path.name}")


# ---------------------------------------------------------------------------
# Run TotalSegmentator
# ---------------------------------------------------------------------------

def run_totalseg(ct_path: Path, out_dir: Path, totalseg_bin: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [totalseg_bin, "-i", str(ct_path), "-o", str(out_dir), "--fast"]
    env = os.environ.copy()
    # Ensure TMPDIR exists so nnU-Net can write temp files
    tmpdir = env.get("TMPDIR", "/tmp")
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"    [error] TotalSegmentator failed:\n{result.stderr[-600:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_subject(subject: str, data_dir: Path, out_dir: Path,
                poset: PosetFromJson, crop_frac: float,
                totalseg_bin: str) -> List[dict]:

    print(f"\n{'='*60}\n  Subject: {subject}  (crop={crop_frac:.0%})\n{'='*60}")

    ct_path  = data_dir / subject / "ct.nii.gz"
    gt_dir   = data_dir / subject / "segmentations"
    if not ct_path.exists() or not gt_dir.exists():
        print("  [skip] CT or GT not found")
        return []

    subj_out   = out_dir / subject
    cropped_ct = subj_out / "ct_cropped.nii.gz"
    pred_dir   = subj_out / "segmentations_pred"

    # Step 1: crop CT
    crop_superior(ct_path, cropped_ct, crop_frac)

    # Step 2: run TotalSegmentator on cropped CT
    print(f"    Running TotalSegmentator...")
    if not run_totalseg(cropped_ct, pred_dir, totalseg_bin):
        return []

    # Step 3: load predictions
    pred_files = {p.stem.replace(".nii", ""): p for p in pred_dir.glob("*.nii.gz")}
    poset_names = {s.name for s in poset.structures}
    common = poset_names & set(pred_files)
    if not common:
        print("  [skip] no structure overlap")
        return []

    predictions = {}
    affine = None
    for name in sorted(common):
        img = nib.load(str(pred_files[name]))
        predictions[name] = np.asarray(img.dataobj).astype(bool)
        if affine is None:
            affine = img.affine

    asmap = axis_sign_map(affine)

    # Step 4: load GT (full, not cropped)
    gt_masks = {}
    for name in common:
        p = gt_dir / f"{name}.nii.gz"
        if p.exists():
            gt_masks[name] = np.asarray(nib.load(str(p)).dataobj).astype(bool)

    # Step 5: apply CC cleaning
    constrained, vox_removed = apply_constraints_gt_free(predictions, poset, asmap)

    # Step 6: evaluate Dice vs full GT
    # NOTE: predictions are on cropped volume, GT is on full volume.
    # We compare only the slice range that exists in the prediction.
    img_full  = nib.load(str(ct_path))
    img_crop  = nib.load(str(cropped_ct))
    codes     = aff2axcodes(img_full.affine)
    si_ax     = next((i for i, c in enumerate(codes) if c in ('S', 'I')), 2)
    n_full    = img_full.shape[si_ax]
    n_cropped = img_crop.shape[si_ax]
    n_removed_slices = n_full - n_cropped

    rows = []
    eval_names = sorted(common & set(gt_masks))
    col_w = max(len(n) for n in eval_names) + 2

    print(f"\n  {'structure':<{col_w}} {'dice_before':>11} {'dice_after':>10} {'delta':>8} {'vox_removed':>12}")
    print(f"  {'-'*(col_w+47)}")

    improved = degraded = unchanged = 0
    for name in eval_names:
        gt_full = gt_masks[name]

        # Align GT to cropped volume space
        slices = [slice(None)] * gt_full.ndim
        if codes[si_ax] == 'S':
            slices[si_ax] = slice(0, n_cropped)
        else:
            slices[si_ax] = slice(n_removed_slices, n_full)
        gt_crop = gt_full[tuple(slices)]

        if not gt_crop.any():
            continue

        # Resize prediction to match GT crop if shapes differ slightly
        if predictions[name].shape != gt_crop.shape:
            continue

        d_before = dice(predictions[name], gt_crop)
        d_after  = dice(constrained[name], gt_crop)
        delta    = d_after - d_before
        removed  = vox_removed.get(name, 0)

        marker = "▲" if delta > 0.0001 else ("▼" if delta < -0.0001 else " ")
        print(f"  {name:<{col_w}} {d_before:>11.4f} {d_after:>10.4f} {delta:>+8.4f} {marker}  {removed:>10,}")

        if delta > 0.0001:   improved += 1
        elif delta < -0.0001: degraded += 1
        else:                 unchanged += 1

        rows.append({
            "subject": subject, "structure": name, "crop_frac": crop_frac,
            "dice_before": round(d_before, 4), "dice_after": round(d_after, 4),
            "delta": round(delta, 4), "voxels_removed": removed,
        })

    if rows:
        mb = sum(r["dice_before"] for r in rows) / len(rows)
        ma = sum(r["dice_after"]  for r in rows) / len(rows)
        print(f"  {'-'*(col_w+47)}")
        print(f"  Improved: {improved}  Degraded: {degraded}  Unchanged: {unchanged}")
        print(f"  Mean Dice  before={mb:.4f}  after={ma:.4f}  delta={ma-mb:+.4f}")

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir",   required=True, help="v201 dataset root (contains sXXXX/ct.nii.gz)")
    p.add_argument("--out_dir",    required=True, help="Output directory for experiment")
    p.add_argument("--poset",      required=True, help="Poset JSON path")
    p.add_argument("--subjects",   nargs="+",     help="Subject IDs to process")
    p.add_argument("--crop_frac",  type=float, default=0.25, help="Fraction of superior slices to remove (default 0.25)")
    p.add_argument("--totalseg",   default="TotalSegmentator", help="Path to TotalSegmentator binary")
    p.add_argument("--csv",        default=None,  help="Save results to CSV")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    poset    = load_poset_from_json(args.poset)

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(d.name for d in data_dir.iterdir()
                          if d.is_dir() and (d / "ct.nii.gz").exists())[:5]

    all_rows = []
    for subject in subjects:
        rows = run_subject(subject, data_dir, out_dir, poset,
                           args.crop_frac, args.totalseg)
        all_rows.extend(rows)

    if args.csv and all_rows:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
