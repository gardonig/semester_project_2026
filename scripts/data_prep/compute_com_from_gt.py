"""
Compute average centre-of-mass (CoM) for every TotalSegmentator structure
from ground-truth segmentation masks and write a GUI-ready structures JSON.

Input layout (TotalSegmentator per-subject format):
    gt_dir/
      subject_001/
        segmentations/
          aorta.nii.gz
          liver.nii.gz
          vertebrae_C1.nii.gz
          ...
        ct.nii.gz

Output JSON (normalised to [0, 100] per axis, averaged across subjects):
    {
      "structures": [
        {"name": "brain", "com_vertical": 94.1, "com_lateral": 50.2, "com_anteroposterior": 48.7},
        ...
      ]
    }

Axis conventions (matching the GUI):
    com_vertical:         0 = feet (inferior),  100 = head (superior)
    com_lateral:          0 = patient far right, 100 = patient far left
    com_anteroposterior:  0 = dorsal (posterior), 100 = ventral (anterior)

Usage:
    python scripts/data_prep/compute_com_from_gt.py \\
        --gt_dir /scratch/gardonig/totalseg_v201 \\
        --out    data/structures/totalseg_v2_com.json

    # Limit to first 20 subjects for a quick test run
    python scripts/data_prep/compute_com_from_gt.py \\
        --gt_dir /scratch/gardonig/totalseg_v201 \\
        --out    data/structures/totalseg_v2_com.json \\
        --max_subjects 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def axis_sign_map(affine: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """Return {anatomical_axis: (voxel_axis_index, sign)}.

    sign=+1 means increasing voxel index → more superior / left / anterior.
    """
    codes = aff2axcodes(affine)
    result: Dict[str, Tuple[int, int]] = {}
    for vox_ax, code in enumerate(codes):
        if   code == 'S': result['vertical']        = (vox_ax, +1)
        elif code == 'I': result['vertical']        = (vox_ax, -1)
        elif code == 'L': result['mediolateral']    = (vox_ax, +1)
        elif code == 'R': result['mediolateral']    = (vox_ax, -1)
        elif code == 'A': result['anteroposterior'] = (vox_ax, +1)
        elif code == 'P': result['anteroposterior'] = (vox_ax, -1)
    return result


def compute_com_normalised(
    mask: np.ndarray,
    affine: np.ndarray,
    min_voxels: int = 10,
) -> Optional[Dict[str, float]]:
    """Return normalised CoM dict or None if mask is too small / empty."""
    vox_idx = np.argwhere(mask)
    if len(vox_idx) < min_voxels:
        return None

    shape = np.array(mask.shape, dtype=float)
    # CoM as fraction of image extent along each voxel axis [0, 1]
    com_frac = vox_idx.mean(axis=0) / (shape - 1)   # shape - 1 avoids /0 on 1-vox dims

    asmap = axis_sign_map(affine)

    def frac_to_norm(frac: float, sign: int) -> float:
        # sign=+1: frac directly maps (0→inferior/right/posterior, 1→superior/left/anterior)
        # sign=-1: frac is inverted
        v = frac if sign == +1 else (1.0 - frac)
        # clamp to [0, 100] and keep away from hard edges
        return float(np.clip(v * 100.0, 0.0, 100.0))

    result: Dict[str, float] = {}
    for anat_axis, gui_key in [
        ('vertical',        'com_vertical'),
        ('mediolateral',    'com_lateral'),
        ('anteroposterior', 'com_anteroposterior'),
    ]:
        if anat_axis not in asmap:
            return None   # unknown orientation — skip this subject
        vox_ax, sign = asmap[anat_axis]
        result[gui_key] = frac_to_norm(com_frac[vox_ax], sign)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_from_dataset(
    gt_dir: Path,
    max_subjects: Optional[int],
    min_voxels: int,
    min_subjects: int,
) -> Dict[str, List[Dict[str, float]]]:
    """Returns {structure_name: [normalised_com_dict, ...]} across subjects."""
    per_struct: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    subjects = sorted(d.name for d in gt_dir.iterdir() if d.is_dir())
    if max_subjects:
        subjects = subjects[:max_subjects]

    print(f"Processing {len(subjects)} subjects from {gt_dir} ...")

    for idx, subject in enumerate(subjects):
        seg_dir = gt_dir / subject / "segmentations"
        ct_path = gt_dir / subject / "ct.nii.gz"

        if not seg_dir.exists():
            continue

        # get affine from ct.nii.gz if available, else from first seg file
        affine = None
        if ct_path.exists():
            affine = nib.load(str(ct_path)).affine

        nii_files = sorted(seg_dir.glob("*.nii.gz"))
        if not nii_files:
            continue

        n_added = 0
        for nii_path in nii_files:
            struct_name = nii_path.name.replace(".nii.gz", "")
            img = nib.load(str(nii_path))

            if affine is None:
                affine = img.affine   # fallback to seg file affine

            mask = np.asarray(img.dataobj, dtype=bool)
            com = compute_com_normalised(mask, affine, min_voxels=min_voxels)
            if com is not None:
                per_struct[struct_name].append(com)
                n_added += 1

        print(f"  [{idx+1}/{len(subjects)}] {subject}: {n_added} structures")

    # filter structures seen in too few subjects
    filtered = {k: v for k, v in per_struct.items() if len(v) >= min_subjects}
    print(f"\n{len(per_struct)} structures found; "
          f"{len(filtered)} kept (≥{min_subjects} subjects each)")
    return filtered


def average_coms(per_struct: Dict[str, List[Dict[str, float]]]) -> List[dict]:
    entries = []
    for name, coms in per_struct.items():
        entry = {"name": name}
        for key in ("com_vertical", "com_lateral", "com_anteroposterior"):
            vals = [c[key] for c in coms if key in c]
            entry[key] = round(sum(vals) / len(vals), 4) if vals else 50.0
        entries.append(entry)

    # sort head → feet (descending vertical)
    entries.sort(key=lambda e: -e["com_vertical"])
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt_dir",       required=True, type=Path,
                   help="Root directory of the TotalSegmentator per-subject GT dataset")
    p.add_argument("--out",          required=True, type=Path,
                   help="Output JSON path")
    p.add_argument("--max_subjects", default=None, type=int,
                   help="Limit to first N subjects (useful for quick tests)")
    p.add_argument("--min_voxels",   default=10, type=int,
                   help="Skip structure masks with fewer than N voxels (default: 10)")
    p.add_argument("--min_subjects", default=5, type=int,
                   help="Drop structures seen in fewer than N subjects (default: 5)")
    args = p.parse_args()

    per_struct = compute_from_dataset(
        args.gt_dir, args.max_subjects, args.min_voxels, args.min_subjects
    )
    entries = average_coms(per_struct)

    output = {"structures": entries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{len(entries)} structures written → {args.out}")
    print("\nVertical range (head→feet):")
    for e in entries[:5]:
        print(f"  {e['name']:<40} {e['com_vertical']:.1f}")
    print("  ...")
    for e in entries[-5:]:
        print(f"  {e['name']:<40} {e['com_vertical']:.1f}")


if __name__ == "__main__":
    main()
