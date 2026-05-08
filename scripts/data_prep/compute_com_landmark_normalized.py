"""
Compute landmark-normalised centre-of-mass for every TotalSegmentator MRI structure.

Normalization frame
-------------------
Each structure's S-I centroid is expressed as a fraction of the distance between
the inferior and superior extents of the *combined vertebrae mask* (`vertebrae.nii.gz`),
which is present in virtually every trunk MRI scan:

    com_vertical = 100 × (centroid_si - vert_inf) / (vert_sup - vert_inf)

where
  vert_sup  = S-I voxel index of the vertebrae superior edge  (≈ C1 top)
  vert_inf  = S-I voxel index of the vertebrae inferior edge  (≈ L5 bottom)
  centroid  = mean S-I voxel index of the target structure

Interpretation
--------------
  com_vertical =   0  → at L5 inferior edge
  com_vertical = 100  → at C1 superior edge
  com_vertical <   0  → below L5 (femur, hip, gluteus, sacrum)
  com_vertical > 100  → above C1 (brain)

This is scan-extent independent: two subjects with different FOV lengths but
the same anatomy will produce the same com_vertical for every organ, because
both the centroid and the landmark are expressed in the same subject-local frame.

At inference time CM3 inverts the mapping using the subject's own vertebrae:
    expected_vox = vert_inf + (com_v / 100.0) × (vert_sup - vert_inf)

Usage
-----
    python scripts/data_prep/compute_com_landmark_normalized.py \\
        --mri_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \\
        --out      data/structures/totalseg_mri_com_landmark.json

    # Lateral / AP columns still use image-extent normalization (CM3 only uses vertical).
    # com_vertical values outside [0, 100] are stored as-is; the out-of-crop guard
    # in CM3 handles structures whose expected position falls outside the current crop.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def get_si_axis(affine: np.ndarray) -> Tuple[int, int]:
    """Return (si_ax, si_sign): si_sign=+1 if high voxel index = superior."""
    codes = aff2axcodes(affine)
    si_ax = next(i for i, c in enumerate(codes) if c in ("S", "I"))
    si_sign = +1 if codes[si_ax] == "S" else -1
    return si_ax, si_sign


def axis_sign_map(affine: np.ndarray) -> Dict[str, Tuple[int, int]]:
    codes = aff2axcodes(affine)
    result: Dict[str, Tuple[int, int]] = {}
    for vox_ax, code in enumerate(codes):
        if   code == "S": result["vertical"]        = (vox_ax, +1)
        elif code == "I": result["vertical"]        = (vox_ax, -1)
        elif code == "L": result["mediolateral"]    = (vox_ax, +1)
        elif code == "R": result["mediolateral"]    = (vox_ax, -1)
        elif code == "A": result["anteroposterior"] = (vox_ax, +1)
        elif code == "P": result["anteroposterior"] = (vox_ax, -1)
    return result


# ---------------------------------------------------------------------------
# Landmark and centroid computation
# ---------------------------------------------------------------------------

def vertebrae_landmarks(mask: np.ndarray, si_ax: int, si_sign: int
                        ) -> Optional[Tuple[float, float]]:
    """Return (vert_sup_vox, vert_inf_vox) or None if mask is empty.

    vert_sup_vox: voxel index of the vertebrae's anatomically superior edge
    vert_inf_vox: voxel index of the vertebrae's anatomically inferior edge

    For si_sign=+1 (S-axis): sup = max index, inf = min index.
    For si_sign=-1 (I-axis): sup = min index, inf = max index.
    """
    other = tuple(ax for ax in range(mask.ndim) if ax != si_ax)
    proj = mask.any(axis=other)
    idx = np.where(proj)[0]
    if len(idx) == 0:
        return None
    lo, hi = int(idx.min()), int(idx.max())
    if si_sign == +1:
        return float(hi), float(lo)   # sup=hi, inf=lo
    else:
        return float(lo), float(hi)   # sup=lo (small index = superior), inf=hi


def si_centroid(mask: np.ndarray, si_ax: int) -> Optional[float]:
    other = tuple(ax for ax in range(mask.ndim) if ax != si_ax)
    proj = mask.any(axis=other)
    idx = np.where(proj)[0]
    return float(idx.mean()) if len(idx) > 0 else None


def lateral_ap_com_normalized(mask: np.ndarray, affine: np.ndarray
                               ) -> Optional[Dict[str, float]]:
    """Lateral and AP CoM as fractions of image extent (0-100)."""
    vox_idx = np.argwhere(mask)
    if len(vox_idx) < 10:
        return None
    shape = np.array(mask.shape, dtype=float)
    com_frac = vox_idx.mean(axis=0) / (shape - 1)
    asmap = axis_sign_map(affine)
    result: Dict[str, float] = {}
    for anat_axis, gui_key in [
        ("mediolateral",    "com_lateral"),
        ("anteroposterior", "com_anteroposterior"),
    ]:
        if anat_axis not in asmap:
            return None
        vox_ax, sign = asmap[anat_axis]
        v = com_frac[vox_ax] if sign == +1 else (1.0 - com_frac[vox_ax])
        result[gui_key] = float(np.clip(v * 100.0, 0.0, 100.0))
    return result


# ---------------------------------------------------------------------------
# Per-structure accumulation
# ---------------------------------------------------------------------------

def process_subject(seg_dir: Path, affine: np.ndarray,
                    min_voxels: int = 10) -> Optional[Dict[str, dict]]:
    """Return {struct_name: {com_vertical, com_lateral, com_ap}} for one subject,
    or None if vertebrae mask is missing / empty."""
    vert_path = seg_dir / "vertebrae.nii.gz"
    if not vert_path.exists():
        return None

    vert_mask = np.asarray(nib.load(str(vert_path)).dataobj, dtype=bool)
    si_ax, si_sign = get_si_axis(affine)
    landmarks = vertebrae_landmarks(vert_mask, si_ax, si_sign)
    if landmarks is None:
        return None
    vert_sup, vert_inf = landmarks
    span = vert_sup - vert_inf        # signed; nonzero because sup ≠ inf
    if abs(span) < 5:                 # degenerate vertebrae mask (< 5 voxels tall)
        return None

    results: Dict[str, dict] = {}
    for seg_file in sorted(seg_dir.glob("*.nii.gz")):
        struct_name = seg_file.stem.replace(".nii", "")
        mask = np.asarray(nib.load(str(seg_file)).dataobj, dtype=bool)
        if mask.sum() < min_voxels:
            continue

        centroid_si = si_centroid(mask, si_ax)
        if centroid_si is None:
            continue

        com_v = (centroid_si - vert_inf) / span * 100.0   # can be < 0 or > 100

        lat_ap = lateral_ap_com_normalized(mask, affine)
        if lat_ap is None:
            continue

        results[struct_name] = {
            "com_vertical": com_v,
            "com_lateral":  lat_ap["com_lateral"],
            "com_anteroposterior": lat_ap["com_anteroposterior"],
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mri_dir",     required=True, type=Path,
                   help="Root of TotalsegmentatorMRI dataset (subjects with mri.nii.gz + segmentations/)")
    p.add_argument("--out",         required=True, type=Path,
                   help="Output JSON path")
    p.add_argument("--min_voxels",  default=10, type=int)
    p.add_argument("--min_subjects",default=5,  type=int,
                   help="Drop structures seen in fewer than N subjects")
    p.add_argument("--exclude",     nargs="*", default=[],
                   help="Subject IDs to exclude (e.g. --exclude s0175 s0236)")
    args = p.parse_args()

    exclude_set = set(args.exclude)
    subjects = sorted(d.name for d in args.mri_dir.iterdir()
                      if d.is_dir() and d.name.startswith("s")
                      and d.name not in exclude_set)
    if exclude_set:
        print(f"Excluding {len(exclude_set)} subjects: {sorted(exclude_set)}")
    print(f"Found {len(subjects)} subjects in {args.mri_dir}")

    # Accumulate per-structure com lists
    per_struct: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    n_used = 0
    n_skipped = 0

    for idx, subj in enumerate(subjects):
        seg_dir = args.mri_dir / subj / "segmentations"
        mri_path = args.mri_dir / subj / "mri.nii.gz"
        if not seg_dir.exists() or not mri_path.exists():
            n_skipped += 1
            continue

        affine = nib.load(str(mri_path)).affine
        result = process_subject(seg_dir, affine, args.min_voxels)
        if result is None:
            n_skipped += 1
            if (idx + 1) % 50 == 0:
                print(f"  [{idx+1}/{len(subjects)}] {subj}: skipped (no valid vertebrae)")
            continue

        for struct_name, coms in result.items():
            per_struct[struct_name].append(coms)
        n_used += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(subjects)}] {subj}: ok  "
                  f"(running total: {n_used} valid subjects)")

    print(f"\nDone: {n_used} subjects used, {n_skipped} skipped.")

    # Filter and average
    filtered = {k: v for k, v in per_struct.items() if len(v) >= args.min_subjects}
    print(f"{len(per_struct)} structures seen; {len(filtered)} kept (≥{args.min_subjects} subjects)")

    entries = []
    for name, coms in filtered.items():
        entry = {
            "name": name,
            "n_subjects_used": len(coms),
        }
        for key in ("com_vertical", "com_lateral", "com_anteroposterior"):
            vals = [c[key] for c in coms if key in c]
            entry[key] = round(sum(vals) / len(vals), 4) if vals else 50.0
        entries.append(entry)

    # Sort head → feet (descending vertebrae-normalised vertical)
    entries.sort(key=lambda e: -e["com_vertical"])

    output = {
        "normalization": "vertebrae_landmark",
        "landmark_structure":    "vertebrae",
        "landmark_sup_extent":   "superior",
        "landmark_inf_extent":   "inferior",
        "com_vertical_meaning":  "0=L5_inferior  100=C1_superior  <0=below_L5  >100=above_C1",
        "n_subjects_with_valid_vertebrae": n_used,
        "structures": entries,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{len(entries)} structures → {args.out}")
    print("\nVertical range sample (head → feet):")
    for e in entries[:6]:
        print(f"  {e['name']:<45} {e['com_vertical']:+7.1f}  (n={e['n_subjects_used']})")
    print("  ...")
    for e in entries[-6:]:
        print(f"  {e['name']:<45} {e['com_vertical']:+7.1f}  (n={e['n_subjects_used']})")


if __name__ == "__main__":
    main()
