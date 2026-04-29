"""
Visualize Poset Constraint Cleaning
=====================================
Shows axial slices before/after cleaning for each structure.
Green = kept prediction, Red = removed by constraint, Blue = GT.

Usage:
    python scripts/cleaning/visualize_cleaning.py --subject amos_0102
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from anatomy_poset.core.io import load_poset_from_json
from anatomy_poset.core.io import PosetFromJson

# reuse functions from postprocessing script (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from poset_constraint_postprocessing import (
    apply_constraints, axis_sign_map, load_gt_amos_multilabel
)

AMOS_STRUCTURES = [
    "adrenal_gland_left", "adrenal_gland_right", "aorta", "duodenum",
    "esophagus", "gallbladder", "inferior_vena_cava", "kidney_left",
    "kidney_right", "liver", "pancreas", "spleen", "stomach", "urinary_bladder",
]


def best_slice(mask: np.ndarray, axis: int = 2) -> int:
    """Return the slice index along `axis` with the most True voxels."""
    sums = mask.any(axis=tuple(ax for ax in range(mask.ndim) if ax != axis))
    # actually sum per slice
    slices = np.moveaxis(mask, axis, 0)
    counts = np.array([s.sum() for s in slices])
    return int(counts.argmax())


def make_rgb(shape2d):
    return np.zeros((*shape2d, 3), dtype=np.float32)


def draw_structure(rgb, mask2d, color):
    for c, v in enumerate(color):
        rgb[..., c] = np.where(mask2d, v, rgb[..., c])


def visualize_subject(subject, pred_dir, poset_path, gt_dir, out_dir):
    poset = load_poset_from_json(str(poset_path))
    subj_pred = pred_dir / subject

    # Load predictions
    predictions = {}
    affine = None
    for name in AMOS_STRUCTURES:
        p = subj_pred / f"{name}.nii.gz"
        if p.exists():
            img = nib.load(str(p))
            predictions[name] = np.asarray(img.dataobj).astype(bool)
            if affine is None:
                affine = img.affine

    if not predictions:
        print("No predictions found.")
        return

    asmap = axis_sign_map(affine)

    # Load GT first — used both as boundary for cleaning and for display
    gt_masks, _ = load_gt_amos_multilabel(gt_dir, subject, AMOS_STRUCTURES)

    # Run cleaning (GT extents used as cut boundaries)
    constrained, vox_removed = apply_constraints(predictions, poset, asmap, gt_masks)

    # Determine axial axis (the one labelled S or I)
    from nibabel.orientations import aff2axcodes
    codes = aff2axcodes(affine)
    axial_ax = next((i for i, c in enumerate(codes) if c in ('S', 'I')), 2)

    shape = next(iter(predictions.values())).shape

    # Colour palette per structure
    cmap = plt.get_cmap("tab20")
    colours = {name: cmap(i / len(AMOS_STRUCTURES))[:3] for i, name in enumerate(AMOS_STRUCTURES)}

    # Find structures that actually had voxels removed
    changed = [n for n in AMOS_STRUCTURES if vox_removed.get(n, 0) > 0 and n in predictions]
    print(f"Structures with removed voxels ({len(changed)}): {changed}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Overview — all structures on their best slice ──────────────
    ncols = 3
    nrows = (len(AMOS_STRUCTURES) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 2 * 4, nrows * 3))
    fig.suptitle(f"{subject} — Before (left) / After (right) constraint cleaning\n"
                 f"Green=kept  Red=removed  Blue=GT", fontsize=13)

    for ax_row in axes.flatten():
        ax_row.axis("off")

    for idx, name in enumerate(AMOS_STRUCTURES):
        row = idx // ncols
        col_base = (idx % ncols) * 2

        pred_mask = predictions.get(name)
        cons_mask = constrained.get(name)
        gt_mask   = gt_masks.get(name) if gt_masks else None

        if pred_mask is None:
            continue

        # Best slice: where pred has most voxels
        sl_idx = best_slice(pred_mask, axis=axial_ax)
        sl = [slice(None)] * 3
        sl[axial_ax] = sl_idx
        sl = tuple(sl)

        pred2d = pred_mask[sl]
        cons2d = cons_mask[sl] if cons_mask is not None else pred2d
        gt2d   = gt_mask[sl]   if gt_mask   is not None else None
        removed2d = pred2d & ~cons2d

        h, w = pred2d.shape[0], pred2d.shape[1] if pred2d.ndim > 1 else pred2d.shape[0]

        def make_slice_img(pred, removed, gt):
            """Black background, green=kept, red=removed, blue=GT outline."""
            rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
            # GT in dark blue
            if gt is not None:
                rgb[gt, 2] = 0.4
            # kept pred in green
            kept = pred & ~removed
            rgb[kept, 1] = 0.8
            # removed in red
            rgb[removed, 0] = 1.0
            return rgb

        img_before = make_slice_img(pred2d, np.zeros_like(pred2d), gt2d)
        img_after  = make_slice_img(cons2d, removed2d, gt2d)

        ax_b = axes[row, col_base]
        ax_a = axes[row, col_base + 1]

        ax_b.imshow(img_before, origin="lower", interpolation="nearest")
        ax_b.set_title(f"{name}\nbefore  (z={sl_idx})", fontsize=7)
        ax_b.axis("off")

        ax_a.imshow(img_after, origin="lower", interpolation="nearest")
        removed_n = int(removed2d.sum())
        ax_a.set_title(f"after\n-{removed_n} vox on slice", fontsize=7)
        ax_a.axis("off")

    patches = [
        mpatches.Patch(color=(0, 0.8, 0), label="kept prediction"),
        mpatches.Patch(color=(1, 0,   0), label="removed by constraint"),
        mpatches.Patch(color=(0, 0,  0.4), label="GT"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = out_dir / f"{subject}_cleaning_overview.png"
    plt.savefig(str(out_path), dpi=120)
    plt.close()
    print(f"Saved overview → {out_path}")

    # ── Figure 2: Zoomed-in on changed structures across multiple slices ──────
    if changed:
        n_slices = 5
        fig2, axes2 = plt.subplots(len(changed), n_slices * 2,
                                   figsize=(n_slices * 2 * 2, len(changed) * 2.5))
        if len(changed) == 1:
            axes2 = axes2[np.newaxis, :]
        fig2.suptitle(f"{subject} — Structures with removed voxels (5 slices each)\n"
                      f"Green=kept  Red=removed  Blue=GT", fontsize=12)

        for ridx, name in enumerate(changed):
            pred_mask = predictions[name]
            cons_mask = constrained[name]
            removed_total = (pred_mask & ~cons_mask)
            gt_mask = gt_masks.get(name) if gt_masks else None

            # Find slices that have removed voxels
            removed_per_slice = np.array([
                removed_total.take(s, axis=axial_ax).sum()
                for s in range(pred_mask.shape[axial_ax])
            ])
            top_slices = np.argsort(removed_per_slice)[::-1][:n_slices]
            top_slices = sorted(top_slices)

            for sidx, sl_idx in enumerate(top_slices):
                sl = [slice(None)] * 3
                sl[axial_ax] = sl_idx
                sl = tuple(sl)

                pred2d    = pred_mask[sl]
                cons2d    = cons_mask[sl]
                removed2d = removed_total[sl]
                gt2d      = gt_mask[sl] if gt_mask is not None else None

                def img(pred, removed, gt):
                    rgb = np.zeros((*pred.shape, 3), dtype=np.float32)
                    if gt is not None: rgb[gt, 2] = 0.4
                    rgb[pred & ~removed, 1] = 0.8
                    rgb[removed, 0] = 1.0
                    return rgb

                ax_b = axes2[ridx, sidx * 2]
                ax_a = axes2[ridx, sidx * 2 + 1]

                ax_b.imshow(img(pred2d, np.zeros_like(pred2d), gt2d),
                            origin="lower", interpolation="nearest")
                ax_b.set_title(f"z={sl_idx}\nbefore", fontsize=6)
                ax_b.axis("off")
                if sidx == 0:
                    ax_b.set_ylabel(name, fontsize=7, rotation=45, ha="right")

                ax_a.imshow(img(cons2d, removed2d, gt2d),
                            origin="lower", interpolation="nearest")
                ax_a.set_title(f"z={sl_idx}\nafter -{int(removed2d.sum())}", fontsize=6)
                ax_a.axis("off")

        plt.tight_layout()
        out_path2 = out_dir / f"{subject}_cleaning_detail.png"
        plt.savefig(str(out_path2), dpi=140)
        plt.close()
        print(f"Saved detail → {out_path2}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subject",  default="amos_0102")
    p.add_argument("--pred_dir", default="data/imaging_datasets/totalseg_output_amos_v157")
    p.add_argument("--poset",    default="data/posets/llm_sessions/llm_claude_v157.json")
    p.add_argument("--gt_dir",   default="data/imaging_datasets/amos22_labelsTr")
    p.add_argument("--out_dir",  default="results/cleaning_viz")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_subject(
        subject=args.subject,
        pred_dir=Path(args.pred_dir),
        poset_path=Path(args.poset),
        gt_dir=Path(args.gt_dir),
        out_dir=Path(args.out_dir),
    )
