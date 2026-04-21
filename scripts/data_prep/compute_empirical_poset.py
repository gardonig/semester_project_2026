"""
Compute an empirical probability poset from TotalSegmentator GT segmentations.

For every ordered pair (i, j) and every anatomical axis, counts across all
subjects how often structure i is STRICTLY above/left-of/anterior-to structure j
(i.e. the bounding boxes are fully non-overlapping with i on the superior side).
The probability P(i > j) = count / subjects_where_both_present.

Output is a probability-poset JSON in the same format as merged annotator sessions,
loadable directly by the GUI and cleaning scripts.

Definition of "strictly above" on vertical axis (sign=+1):
    inferior boundary of i  >  superior boundary of j
    i.e.  min_vox_i  >  max_vox_j  (no overlap, i entirely above j)

Usage:
    python scripts/data_prep/compute_empirical_poset.py \\
        --gt_dir /scratch/gardonig/totalseg_v201 \\
        --out    data/structures/totalseg_v2_empirical_poset.json

    # Quick test on 30 subjects
    python scripts/data_prep/compute_empirical_poset.py \\
        --gt_dir /scratch/gardonig/totalseg_v201 \\
        --out    data/structures/totalseg_v2_empirical_poset.json \\
        --max_subjects 30

    # Use pre-computed CoM JSON to fix the structure list and order
    python scripts/data_prep/compute_empirical_poset.py \\
        --gt_dir      /scratch/gardonig/totalseg_v201 \\
        --com_json    data/structures/totalseg_v2_com.json \\
        --out         data/structures/totalseg_v2_empirical_poset.json
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
# Orientation helpers (same as postprocessing script)
# ---------------------------------------------------------------------------

def axis_sign_map(affine: np.ndarray) -> Dict[str, Tuple[int, int]]:
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


def bbox_normalised(
    mask: np.ndarray,
    affine: np.ndarray,
    min_voxels: int = 10,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Returns {axis_name: (inferior_frac, superior_frac)} in [0,1] anatomical coords.
    inferior_frac=0 means bottom of image, superior_frac=1 means top.
    Returns None if mask is empty or too small.
    """
    idx = np.argwhere(mask)
    if len(idx) < min_voxels:
        return None

    shape = np.array(mask.shape, dtype=float)
    asmap = axis_sign_map(affine)
    result: Dict[str, Tuple[float, float]] = {}

    for anat_axis in ('vertical', 'mediolateral', 'anteroposterior'):
        if anat_axis not in asmap:
            return None
        vox_ax, sign = asmap[anat_axis]
        vals = idx[:, vox_ax]
        lo = int(vals.min())
        hi = int(vals.max())
        denom = max(shape[vox_ax] - 1, 1)

        if sign == +1:
            # increasing voxel index = more superior/left/anterior
            inf_frac = lo / denom   # bottom = low index
            sup_frac = hi / denom   # top    = high index
        else:
            # increasing voxel index = more inferior/right/posterior
            inf_frac = 1.0 - hi / denom
            sup_frac = 1.0 - lo / denom

        result[anat_axis] = (inf_frac, sup_frac)

    return result


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

AXES = ('vertical', 'mediolateral', 'anteroposterior')


def process_subject(
    seg_dir: Path,
    affine: np.ndarray,
    structure_names: List[str],
    min_voxels: int,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Returns {structure_name: {axis: (inf_frac, sup_frac)}} for all present structures.
    """
    bboxes: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for name in structure_names:
        nii_path = seg_dir / f"{name}.nii.gz"
        if not nii_path.exists():
            continue
        img  = nib.load(str(nii_path))
        mask = np.asarray(img.dataobj, dtype=bool)
        bb   = bbox_normalised(mask, affine, min_voxels)
        if bb is not None:
            bboxes[name] = bb
    return bboxes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt_dir",       required=True,  type=Path)
    p.add_argument("--out",          required=True,  type=Path)
    p.add_argument("--com_json",     default=None,   type=Path,
                   help="Optional CoM JSON to fix structure list and order")
    p.add_argument("--max_subjects", default=None,   type=int)
    p.add_argument("--min_voxels",   default=10,     type=int)
    p.add_argument("--min_subjects", default=5,      type=int,
                   help="Pairs seen in fewer subjects get null probability")
    args = p.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Discover all structures
    # ------------------------------------------------------------------ #
    subjects = sorted(d.name for d in args.gt_dir.iterdir() if d.is_dir())
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    print(f"{len(subjects)} subjects found in {args.gt_dir}")

    if args.com_json:
        com_data = json.load(open(args.com_json))
        structure_names = [s["name"] for s in com_data["structures"]]
        com_map = {s["name"]: s for s in com_data["structures"]}
        print(f"Using {len(structure_names)} structures from {args.com_json}")
    else:
        # Discover from dataset
        all_names: set = set()
        for subj in subjects[:50]:  # sample first 50 to find all structure names
            seg_dir = args.gt_dir / subj / "segmentations"
            if seg_dir.exists():
                all_names.update(p.stem.replace(".nii", "") for p in seg_dir.glob("*.nii.gz"))
        structure_names = sorted(all_names)
        com_map = {}
        print(f"Discovered {len(structure_names)} unique structures")

    n = len(structure_names)
    name_to_idx = {name: i for i, name in enumerate(structure_names)}

    # ------------------------------------------------------------------ #
    # 2. Accumulate counts across subjects
    # ------------------------------------------------------------------ #
    # For each axis and each ordered pair (i,j):
    #   strictly_above[axis][i][j] = count of subjects where i strictly above j
    #   both_present[i][j]         = count of subjects where both i and j present
    strictly_above = {ax: np.zeros((n, n), dtype=np.int32) for ax in AXES}
    both_present   = np.zeros((n, n), dtype=np.int32)

    for s_idx, subject in enumerate(subjects):
        subj_dir = args.gt_dir / subject
        seg_dir  = subj_dir / "segmentations"
        ct_path  = subj_dir / "ct.nii.gz"

        if not seg_dir.exists():
            continue

        # Get affine
        affine = None
        if ct_path.exists():
            affine = nib.load(str(ct_path)).affine
        else:
            first = next(seg_dir.glob("*.nii.gz"), None)
            if first:
                affine = nib.load(str(first)).affine
        if affine is None:
            continue

        bboxes = process_subject(seg_dir, affine, structure_names, args.min_voxels)
        present = list(bboxes.keys())

        # Update pair counts
        for a_name in present:
            i = name_to_idx[a_name]
            for b_name in present:
                if a_name == b_name:
                    continue
                j = name_to_idx[b_name]
                both_present[i, j] += 1

                bb_i = bboxes[a_name]
                bb_j = bboxes[b_name]
                for ax in AXES:
                    inf_i, sup_i = bb_i[ax]
                    inf_j, sup_j = bb_j[ax]
                    # i strictly above/left-of/anterior-to j:
                    # entire i is on the superior side — bottom of i > top of j
                    if inf_i > sup_j:
                        strictly_above[ax][i, j] += 1

        if (s_idx + 1) % 100 == 0:
            print(f"  [{s_idx+1}/{len(subjects)}] processed {subject}")

    print(f"\nDone. Computing probabilities...")

    # ------------------------------------------------------------------ #
    # 3. Compute probabilities → build output matrices
    # ------------------------------------------------------------------ #
    # prob[ax][i][j] = None if too few subjects, else float in [0,1]
    matrices = {}
    for ax in AXES:
        mat = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(None)
                elif both_present[i, j] < args.min_subjects:
                    row.append(None)
                else:
                    prob = float(strictly_above[ax][i, j]) / float(both_present[i, j])
                    row.append(round(prob, 4))
            mat.append(row)
        matrices[ax] = mat

    # ------------------------------------------------------------------ #
    # 4. Build structure entries (with CoM if available)
    # ------------------------------------------------------------------ #
    structures_out = []
    for name in structure_names:
        if com_map and name in com_map:
            structures_out.append(com_map[name])
        else:
            structures_out.append({"name": name,
                                   "com_vertical": 50.0,
                                   "com_lateral": 50.0,
                                   "com_anteroposterior": 50.0})

    # ------------------------------------------------------------------ #
    # 5. Write output
    # ------------------------------------------------------------------ #
    output = {
        "structures":              structures_out,
        "matrix_vertical":         matrices["vertical"],
        "matrix_mediolateral":     matrices["mediolateral"],
        "matrix_anteroposterior":  matrices["anteroposterior"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    # ------------------------------------------------------------------ #
    # 6. Print summary
    # ------------------------------------------------------------------ #
    print(f"\n{n} structures  |  {len(subjects)} subjects")
    print(f"Output → {args.out}")

    for ax in AXES:
        mat = matrices[ax]
        high = [(structure_names[i], structure_names[j], mat[i][j])
                for i in range(n) for j in range(n)
                if mat[i][j] is not None and mat[i][j] >= 0.99]
        print(f"\n{ax}: {len(high)} pairs with P ≥ 0.99")
        for a, b, prob in sorted(high, key=lambda x: -x[2])[:10]:
            print(f"  P({a} > {b}) = {prob:.4f}")


if __name__ == "__main__":
    main()
