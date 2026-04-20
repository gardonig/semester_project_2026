"""
Anatomical Constraint Post-Processing Pipeline
===============================================

Applies poset-derived anatomical ordering constraints to TotalSegmentator
predictions and measures Dice improvement (if GT is available).

Algorithm (per axis, per ordered pair i → j where matrix[i][j] == 1,
meaning structure i is STRICTLY above/left-of/anterior-to structure j):

  For each such pair, remove predicted voxels of i that lie BELOW j's
  inferior (bottom-most) GT boundary.  Any voxel of i that is more inferior
  than the very bottom of j cannot be anatomically correct.

  Using j's inferior boundary (rather than j's superior boundary) is the
  conservative choice: it only removes i voxels that are clearly displaced
  below j entirely, which is safe for closely-spaced structures
  (e.g. adjacent vertebrae, ribs) whose bounding boxes may legitimately
  overlap.

Note: boundaries are derived from GT, so GT is required for cleaning.

Usage (AMOS v157, just clean + save):
--------------------------------------
    python scripts/poset_constraint_postprocessing.py \\
        --pred_dir data/imaging_datasets/totalseg_output_amos_v157 \\
        --poset    data/posets/llm_sessions/llm_claude_v157.json \\
        --out_dir  data/imaging_datasets/totalseg_output_amos_v157_cleaned

Usage (with GT for Dice evaluation, single subject):
----------------------------------------------------
    python scripts/poset_constraint_postprocessing.py \\
        --pred_dir  data/imaging_datasets/totalseg_output_amos_v157 \\
        --poset     data/posets/llm_sessions/llm_claude_v157.json \\
        --gt_dir    data/imaging_datasets/amos22_labelsTr \\
        --gt_format amos_multilabel \\
        --subject   amos_0102 \\
        --csv       results/amos_constraint_v157.csv

Usage (small v201 dataset with per-structure GT, for testing):
-------------------------------------------------------------
    python scripts/poset_constraint_postprocessing.py \\
        --pred_dir   data/imaging_datasets/totalseg_output_small_v201 \\
        --gt_dir     data/imaging_datasets/Totalsegmentator_dataset_small_v201 \\
        --gt_format  totalseg_per_subject \\
        --poset      data/posets/llm_sessions/llm_claude_v157.json \\
        --all_subjects \\
        --csv        results/v201_constraint_v157.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy.ndimage import label as cc_label

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from anatomy_poset.core.io import load_poset_from_json, PosetFromJson  # noqa: E402

# AMOS label index → TotalSegmentator v157 structure name
AMOS_LABEL_MAP: Dict[int, str] = {
    1:  "spleen",
    2:  "kidney_right",
    3:  "kidney_left",
    4:  "gallbladder",
    5:  "esophagus",
    6:  "liver",
    7:  "stomach",
    8:  "aorta",
    9:  "inferior_vena_cava",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "duodenum",
    14: "urinary_bladder",
    15: "prostate",
}

# FLARE22 label index → TotalSegmentator v157 structure name
FLARE22_LABEL_MAP: Dict[int, str] = {
    1:  "liver",
    2:  "kidney_right",
    3:  "spleen",
    4:  "pancreas",
    5:  "aorta",
    6:  "inferior_vena_cava",
    7:  "adrenal_gland_right",
    8:  "adrenal_gland_left",
    9:  "gallbladder",
    10: "esophagus",
    11: "stomach",
    12: "duodenum",
    13: "kidney_left",
}

DEFAULT_POSET = PROJECT_ROOT / "data" / "posets" / "llm_sessions" / "llm_claude_v157.json"
DEFAULT_PRED  = PROJECT_ROOT / "data" / "imaging_datasets" / "totalseg_output_amos_v157"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice(pred, gt) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return float(2.0 * (p & g).sum() / denom) if denom > 0 else 1.0


def volume_ml(mask, affine) -> float:
    vox_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    return float(mask.astype(bool).sum() * vox_mm3 / 1000.0)


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def axis_sign_map(affine: np.ndarray) -> Dict[str, Tuple[int, int]]:
    """Maps anatomy axis name → (voxel_axis_index, sign).
    sign=+1: increasing voxel index → more superior / left / anterior.
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


def axis_extent(mask: np.ndarray, vox_ax: int) -> Optional[Tuple[int, int]]:
    """Returns (min_idx, max_idx) along vox_ax where mask is True, or None if empty."""
    other = tuple(ax for ax in range(mask.ndim) if ax != vox_ax)
    proj  = mask.any(axis=other)
    idx   = np.where(proj)[0]
    if len(idx) == 0:
        return None
    return int(idx.min()), int(idx.max())


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Returns a boolean mask containing only the largest connected component."""
    if not mask.any():
        return mask.copy()
    labeled, n = cc_label(mask)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(bool)


# ---------------------------------------------------------------------------
# Core constraint application
# ---------------------------------------------------------------------------

def apply_constraints_gt_free(
    predictions: Dict[str, np.ndarray],
    poset:       PosetFromJson,
    asmap:       Dict[str, Tuple[int, int]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    GT-free constraint cleaning via connected component analysis.

    For each constraint "A is strictly above B":
      1. Extract B's largest connected component (LCC) as the trusted reference.
      2. Find all connected components of A.
      3. Remove any non-LCC component of A that lies *entirely* below B's LCC
         inferior boundary — these are disconnected false positives in the wrong zone.

    Conservative: A's own LCC is never removed, even if it violates the constraint.
    """
    constrained    = {name: mask.copy() for name, mask in predictions.items()}
    voxels_removed = {name: 0 for name in predictions}

    for axis_name, matrix in [('vertical', poset.matrix_vertical)]:
        if axis_name not in asmap:
            continue
        vox_ax, sign = asmap[axis_name]

        for i, si in enumerate(poset.structures):
            for j, sj in enumerate(poset.structures):
                if i == j:
                    continue
                cell = matrix[i][j]
                if cell is None or cell != 1:
                    continue

                name_i, name_j = si.name, sj.name
                if name_i not in constrained or name_j not in constrained:
                    continue

                mask_i = constrained[name_i]
                mask_j = constrained[name_j]
                if not mask_i.any() or not mask_j.any():
                    continue

                # Trusted inferior boundary of j
                ext_j = axis_extent(largest_connected_component(mask_j), vox_ax)
                if ext_j is None:
                    continue
                min_j, max_j = ext_j

                # Label all components of i
                labeled_i, n_i = cc_label(mask_i)
                if n_i == 0:
                    continue
                sizes_i = np.bincount(labeled_i.ravel())
                sizes_i[0] = 0
                lcc_label_i = int(sizes_i.argmax())

                for comp_label in range(1, n_i + 1):
                    if comp_label == lcc_label_i:
                        continue  # never remove i's main body

                    comp = labeled_i == comp_label
                    ext_c = axis_extent(comp, vox_ax)
                    if ext_c is None:
                        continue
                    c_min, c_max = ext_c

                    # Entire component must be in the violation zone to be removed
                    if sign == +1:
                        entirely_violated = c_max < min_j  # all of comp below j's bottom
                    else:
                        entirely_violated = c_min > max_j

                    if entirely_violated:
                        removed = int(comp.sum())
                        voxels_removed[name_i] += removed
                        constrained[name_i][comp] = False

    return constrained, voxels_removed


# ---------------------------------------------------------------------------
# GT loaders (used for Dice evaluation only, not for cleaning)
# ---------------------------------------------------------------------------

def load_gt_amos_multilabel(
    gt_dir: Path,
    subject: str,
    structures: List[str],
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """Load AMOS multi-label GT file and split into per-structure binary masks."""
    candidates = [
        gt_dir / f"{subject}.nii.gz",
        gt_dir / f"labels{subject.split('_')[-1]}.nii.gz",
    ]
    gt_path = next((p for p in candidates if p.exists()), None)
    if gt_path is None:
        return None, None

    img  = nib.load(str(gt_path))
    data = np.asarray(img.dataobj, dtype=np.int16)

    masks: Dict[str, np.ndarray] = {}
    for label_idx, name in AMOS_LABEL_MAP.items():
        if name in structures:
            masks[name] = (data == label_idx)

    return masks, img.affine


def load_gt_flare22_multilabel(
    gt_dir: Path,
    subject: str,
    structures: List[str],
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """Load FLARE22 multi-label GT file and split into per-structure binary masks."""
    gt_path = gt_dir / f"{subject}.nii.gz"
    if not gt_path.exists():
        return None, None

    img  = nib.load(str(gt_path))
    data = np.asarray(img.dataobj, dtype=np.int16)

    masks: Dict[str, np.ndarray] = {}
    for label_idx, name in FLARE22_LABEL_MAP.items():
        if name in structures:
            masks[name] = (data == label_idx)

    return masks, img.affine


def load_gt_per_subject(
    gt_dir: Path,
    subject: str,
    structures: List[str],
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """Load per-structure GT masks from Totalsegmentator_dataset layout."""
    seg_dir = gt_dir / subject / "segmentations"
    ct_path = gt_dir / subject / "ct.nii.gz"
    if not seg_dir.exists():
        return None, None

    affine = None
    if ct_path.exists():
        affine = nib.load(str(ct_path)).affine

    masks: Dict[str, np.ndarray] = {}
    for name in structures:
        p = seg_dir / f"{name}.nii.gz"
        if p.exists():
            img = nib.load(str(p))
            masks[name] = np.asarray(img.dataobj).astype(bool)
            if affine is None:
                affine = img.affine

    return (masks, affine) if masks else (None, None)


# ---------------------------------------------------------------------------
# Single-subject runner
# ---------------------------------------------------------------------------

def run_subject(
    subject:    str,
    pred_dir:   Path,
    poset:      PosetFromJson,
    gt_dir:     Optional[Path],
    gt_format:  str,
    out_dir:    Optional[Path],
    structures: Optional[List[str]] = None,
) -> List[dict]:

    subj_pred_dir = pred_dir / subject
    if not subj_pred_dir.exists():
        print(f"  [skip] pred dir not found: {subj_pred_dir}")
        return []

    # Load all predictions (support both flat and segmentations/ subdirectory layouts)
    pred_search_dir = subj_pred_dir
    if not list(subj_pred_dir.glob("*.nii.gz")) and (subj_pred_dir / "segmentations").exists():
        pred_search_dir = subj_pred_dir / "segmentations"
    pred_names = {p.stem.replace(".nii", ""): p
                  for p in pred_search_dir.glob("*.nii.gz")}
    poset_names = {s.name for s in poset.structures}
    common = poset_names & set(pred_names)
    if structures:
        common = common & set(structures)
    if not common:
        print(f"  [skip] no structures overlap between poset and predictions")
        return []

    # Load predictions + get affine from first file
    predictions: Dict[str, np.ndarray] = {}
    affine = None
    header = None
    first_img = None
    for name in sorted(common):
        img = nib.load(str(pred_names[name]))
        predictions[name] = np.asarray(img.dataobj).astype(bool)
        if affine is None:
            affine = img.affine
            header = img.header
            first_img = img

    asmap = axis_sign_map(affine)
    print(f"  Orientation: {aff2axcodes(affine)}  |  {len(common)} structures")

    # Load GT first (needed as boundary reference for cleaning)
    gt_masks, gt_affine = None, None
    if gt_dir is not None:
        if gt_format == "amos_multilabel":
            gt_masks, gt_affine = load_gt_amos_multilabel(gt_dir, subject, list(common))
        elif gt_format == "flare22_multilabel":
            gt_masks, gt_affine = load_gt_flare22_multilabel(gt_dir, subject, list(common))
        else:
            gt_masks, gt_affine = load_gt_per_subject(gt_dir, subject, list(common))

    # Apply GT-free constraints (connected component analysis)
    constrained, vox_removed = apply_constraints_gt_free(predictions, poset, asmap)

    # Save cleaned predictions if requested
    if out_dir is not None:
        subj_out = out_dir / subject
        subj_out.mkdir(parents=True, exist_ok=True)
        for name, mask in constrained.items():
            out_img = nib.Nifti1Image(mask.astype(np.uint8), affine, header)
            nib.save(out_img, str(subj_out / f"{name}.nii.gz"))

    # Evaluate Dice if GT is available
    rows = []
    if gt_dir is not None:

        if gt_masks:
            gt_aff = gt_affine if gt_affine is not None else affine
            eval_structures = sorted(common & set(gt_masks))
            for name in eval_structures:
                if not gt_masks[name].any():
                    continue
                d_before = dice(predictions[name],  gt_masks[name])
                d_after  = dice(constrained[name],  gt_masks[name])
                rows.append({
                    "subject":        subject,
                    "structure":      name,
                    "dice_before":    round(d_before, 4),
                    "dice_after":     round(d_after,  4),
                    "delta":          round(d_after - d_before, 4),
                    "voxels_removed": vox_removed.get(name, 0),
                    "vol_pred_ml":    round(volume_ml(predictions[name], affine), 2),
                    "vol_gt_ml":      round(volume_ml(gt_masks[name],   gt_aff),  2),
                })
        else:
            print(f"  [warn] GT not found for {subject}, skipping Dice")
    else:
        # No GT: just report how many voxels were removed
        for name in sorted(common):
            removed = vox_removed.get(name, 0)
            if removed > 0:
                rows.append({
                    "subject":        subject,
                    "structure":      name,
                    "voxels_removed": removed,
                    "vol_pred_ml":    round(volume_ml(predictions[name], affine), 2),
                })

    return rows


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_table(rows: List[dict], with_dice: bool) -> None:
    if not rows:
        return
    col_w = max(len(r["structure"]) for r in rows) + 2

    if with_dice:
        hdr = f"{'structure':<{col_w}} {'dice_before':>11} {'dice_after':>10} {'delta':>8} {'vox_removed':>12}"
        print(hdr)
        print("-" * len(hdr))
        improved = degraded = unchanged = 0
        for r in rows:
            d = r["delta"]
            marker = "▲" if d > 0 else ("▼" if d < 0 else " ")
            print(f"{r['structure']:<{col_w}} {r['dice_before']:>11.4f} {r['dice_after']:>10.4f} "
                  f"{d:>+8.4f} {marker}  {r['voxels_removed']:>10,}")
            if d > 0:   improved += 1
            elif d < 0: degraded += 1
            else:       unchanged += 1
        print("-" * len(hdr))
        mb = sum(r["dice_before"] for r in rows) / len(rows)
        ma = sum(r["dice_after"]  for r in rows) / len(rows)
        print(f"Improved: {improved}  Degraded: {degraded}  Unchanged: {unchanged}")
        print(f"Mean Dice  before={mb:.4f}  after={ma:.4f}  delta={ma - mb:+.4f}")
    else:
        hdr = f"{'structure':<{col_w}} {'vox_removed':>12} {'vol_pred_ml':>12}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{r['structure']:<{col_w}} {r['voxels_removed']:>12,} {r.get('vol_pred_ml', 0):>12.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_dir",   default=str(DEFAULT_PRED),
                   help="Directory containing per-subject prediction folders")
    p.add_argument("--poset",      default=str(DEFAULT_POSET),
                   help="Path to poset JSON (default: llm_claude_v157.json)")
    p.add_argument("--gt_dir",     default=None,
                   help="GT directory (optional). If absent, only cleaning is done.")
    p.add_argument("--gt_format",  default="amos_multilabel",
                   choices=["amos_multilabel", "flare22_multilabel", "totalseg_per_subject"],
                   help="GT format (default: amos_multilabel)")
    p.add_argument("--out_dir",    default=None,
                   help="Save cleaned predictions here (optional)")
    p.add_argument("--subject",    default=None,
                   help="Single subject ID, e.g. amos_0102")
    p.add_argument("--all_subjects", action="store_true",
                   help="Run on all subjects found in pred_dir")
    p.add_argument("--structures", nargs="+", default=None,
                   help="Limit to these structure names only")
    p.add_argument("--csv",        default=None,
                   help="Save results table to this CSV file")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    pred_dir = Path(args.pred_dir)
    gt_dir   = Path(args.gt_dir) if args.gt_dir else None
    out_dir  = Path(args.out_dir) if args.out_dir else None
    poset    = load_poset_from_json(args.poset)
    with_dice = gt_dir is not None

    if args.all_subjects:
        subjects = sorted(d.name for d in pred_dir.iterdir() if d.is_dir())
    elif args.subject:
        subjects = [args.subject]
    else:
        # default: first subject found
        subjects = [next(d.name for d in sorted(pred_dir.iterdir()) if d.is_dir())]

    all_rows: List[dict] = []
    for subject in subjects:
        print(f"\n{'='*60}\n  Subject: {subject}\n{'='*60}")
        rows = run_subject(subject, pred_dir, poset, gt_dir, args.gt_format, out_dir, args.structures)
        print_table(rows, with_dice)
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
