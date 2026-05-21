"""
Erosion-based baseline cleaning for MRI wrap-around artifact segmentations.

Method
------
Morphological opening (binary erosion followed by binary dilation with the same
ball-shaped structuring element) and largest-connected-component (LCC) retention.
Two sub-methods are available via --method:

  opening_lcc  (default)
      1. Erode mask with a ball of --radius voxels.
      2. Keep only the largest connected component of the eroded result.
      3. Dilate back with the same ball, clamped to the original mask so no new
         voxels are ever added — only spurious voxels are removed.
      Matches the pipeline described in Furtado (2021): "preceding the calculation
      of the largest region by morphological erosion, then calculating and isolating
      the largest region, subsequently applying dilation … to reverse the previous
      erosion operation."

  lcc_only
      Keep only the largest connected component of the raw prediction mask —
      no erosion or dilation.  Acts as the simplest possible post-processing
      baseline and allows isolating the contribution of morphological filtering.

Why erosion works for artifact cleaning
---------------------------------------
Erosion shrinks every foreground region by --radius voxels.  Small, spatially
isolated spurious predictions (wrap-around ghost components, floating false
positives) disappear entirely after erosion while the dominant true-anatomy
region survives.  The subsequent LCC step discards any remaining small fragments,
and the dilation step restores the true region to approximately its original
extent.

Key limitation: erosion does not use any anatomical knowledge.  If the ghost
component is large (high-d, high-r artifact conditions) it may survive erosion,
making this baseline weaker than the poset-based cleaner in those regimes — which
is the intended comparison.

References
----------
Furtado, P.N. (2021). Improving Deep Segmentation of Abdominal Organs MRI by
    Post-Processing. BioMedInformatics 1(3), 88–105.
    https://doi.org/10.3390/biomedinformatics1030007

Isensee, F. et al. (2021). nnU-Net: A Self-Configuring Method for Deep
    Learning-Based Biomedical Image Segmentation. Nature Methods 18, 203–211.
    https://doi.org/10.1038/s41592-020-01008-z  (automated LCC post-processing
    as a default pipeline step)

Fu, Y. et al. (2021). A Review of Deep Learning Based Methods for Medical Image
    Multi-Organ Segmentation. Physica Medica 85, 107–122.
    https://doi.org/10.1016/j.ejmp.2021.05.003  (morphological ops "widely used
    to remove small erroneous labels")

Neeteson, N.J. et al. (2023). Automatic Segmentation of Trabecular and Cortical
    Compartments in HR-pQCT Images Using an Embedding-Predicting U-Net and
    Morphological Post-processing. Scientific Reports 13, 252.
    https://doi.org/10.1038/s41598-022-27350-0

Guzzi, L. et al. (2024). Differentiable Soft Morphological Filters for Medical
    Image Segmentation. MICCAI 2024, LNCS 15009, 174–183.
    https://doi.org/10.1007/978-3-031-72111-3_17

Usage
-----
    python scripts/cleaning/evaluate_erosion_baseline.py \\
        --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \\
        --exp_dir   data/wraparound_experiments/wraparound_v4 \\
        --subject   s0175 \\
        --out_dir   data/wraparound_experiments/wraparound_v4_eval/erosion_baseline \\
        --radius    2 \\
        --method    opening_lcc

    # Multiple subjects:
    python scripts/cleaning/evaluate_erosion_baseline.py \\
        --subjects  s0175 s0236 s0219 \\
        --out_dir   data/wraparound_experiments/wraparound_v4_eval/erosion_baseline
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import label as cc_label


# ---------------------------------------------------------------------------
# Geometry helpers  (copied from evaluate_cleaning_methods.py for self-
# containedness — keeps this script runnable without importing the main module)
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


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum())
    pos = int(g.sum())
    return tp / pos if pos > 0 else 1.0


def tp_fp_fn(pred: np.ndarray, gt: np.ndarray):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum())
    fp = int(p.sum()) - tp
    fn = int(g.sum()) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Morphological helpers
# ---------------------------------------------------------------------------

def make_ball(radius: int) -> np.ndarray:
    """Return a 3D spherical binary structuring element of given voxel radius."""
    r = max(1, int(radius))
    cz, cy, cx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (cz ** 2 + cy ** 2 + cx ** 2) <= r ** 2


def select_lcc(mask: np.ndarray) -> np.ndarray:
    """Return a mask containing only the largest connected component."""
    if not mask.any():
        return mask.copy()
    labeled, n = cc_label(mask)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    lcc = int(sizes.argmax())
    return (labeled == lcc).astype(bool)


# ---------------------------------------------------------------------------
# Cleaning methods
# ---------------------------------------------------------------------------

def method_opening_lcc(
    predictions: Dict[str, np.ndarray],
    radius: int = 2,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Morphological opening (erode → LCC → dilate) clamped to original mask.

    Pipeline per structure:
      1. Erode with a ball of *radius* voxels — disconnects/removes small components.
      2. Keep only the LCC of the eroded result.
      3. Dilate with the same ball, then AND with the original mask so no new
         voxels are ever introduced — this is a purely subtractive operation.

    If erosion empties a mask entirely (structure smaller than the ball), the
    original mask is returned unchanged to avoid spurious zero-Dice outcomes.
    """
    ball = make_ball(radius)
    cleaned: Dict[str, np.ndarray] = {}
    removed: Dict[str, int] = {}

    for name, mask in predictions.items():
        orig = mask.astype(bool)
        n_orig = int(orig.sum())

        if n_orig == 0:
            cleaned[name] = orig.copy()
            removed[name] = 0
            continue

        eroded = binary_erosion(orig, structure=ball)

        if not eroded.any():
            # Erosion removed everything — structure is smaller than the kernel.
            # Fall back to the original mask so we don't accidentally zero it out.
            cleaned[name] = orig.copy()
            removed[name] = 0
            continue

        # Keep largest connected component of the eroded mask
        lcc_eroded = select_lcc(eroded)

        # Dilate back and clamp — no new voxels beyond original extent
        dilated = binary_dilation(lcc_eroded, structure=ball) & orig

        cleaned[name] = dilated
        removed[name] = n_orig - int(dilated.sum())

    return cleaned, removed


def method_lcc_only(
    predictions: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Keep only the largest connected component — no erosion or dilation.

    This is the simplest possible post-processing baseline: no morphological
    parameters to tune, and directly comparable to nnU-Net's built-in
    post-processing step (Isensee et al., 2021).
    """
    cleaned: Dict[str, np.ndarray] = {}
    removed: Dict[str, int] = {}

    for name, mask in predictions.items():
        orig = mask.astype(bool)
        result = select_lcc(orig)
        cleaned[name] = result
        removed[name] = int(orig.sum()) - int(result.sum())

    return cleaned, removed


# ---------------------------------------------------------------------------
# Crop helpers
# ---------------------------------------------------------------------------

CROPS = [
    ("brain_to_heart",  "brain",        "heart"),
    ("heart_to_kidney", "heart",        "kidney_left"),
    ("kidney_to_hip",   "kidney_left",  "hip_left"),
]
FALLBACKS = {"kidney_left": "kidney_right", "hip_left": "hip_right"}

DEFAULT_D_FRACS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DEFAULT_R_VALS  = (0.25, 0.50, 0.75, 1.00)
MARGIN = 5


def build_tags(d_fracs, r_vals) -> List[str]:
    return [f"d{int(d * 100):03d}_r{int(r * 100):03d}"
            for d in d_fracs for r in r_vals]


def crop_gt(gt_full: np.ndarray, si_ax: int, crop_lo: int, crop_hi: int) -> np.ndarray:
    slices = [slice(None)] * gt_full.ndim
    slices[si_ax] = slice(crop_lo, crop_hi)
    return gt_full[tuple(slices)]


def compute_crop_window(
    gt_dir: Path, anchors: Tuple[str, str], si_ax: int, full_height: int
) -> Tuple[int, int]:
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
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args, tags: Optional[List[str]] = None) -> List[dict]:
    data_dir = Path(args.data_dir)
    exp_dir  = Path(args.exp_dir)
    subjects = getattr(args, "subjects", None) or [args.subject]
    method   = getattr(args, "method", "opening_lcc")
    radius   = getattr(args, "radius", 2)

    all_rows = []
    for subj in subjects:
        print(f"\n{'#' * 70}")
        print(f"  SUBJECT: {subj}  method={method}  radius={radius}")
        print(f"{'#' * 70}")
        rows = _run_one_subject(args, subj, data_dir, exp_dir, method, radius,
                                tags=tags)
        all_rows.extend(rows)
    return all_rows


def _run_one_subject(
    args,
    subj: str,
    data_dir: Path,
    exp_dir: Path,
    method: str,
    radius: int,
    tags: Optional[List[str]] = None,
) -> List[dict]:
    gt_dir = data_dir / subj / "segmentations"

    mri_full = nib.load(str(data_dir / subj / "mri.nii.gz"))
    si_ax, _si_sign = get_si_info(mri_full.affine)
    full_height = mri_full.shape[si_ax]

    gt_names = {p.stem.replace(".nii", "") for p in gt_dir.glob("*.nii.gz")}

    all_rows = []

    for crop_name, anc_a, anc_b in CROPS:
        crop_lo, crop_hi = compute_crop_window(
            gt_dir, (anc_a, anc_b), si_ax, full_height
        )
        N = crop_hi - crop_lo
        print(f"\n{'=' * 70}")
        print(f"  Crop: {crop_name}  lo={crop_lo} hi={crop_hi} N={N}")

        # Pre-load and crop GT masks (constant across tags)
        gt_masks: Dict[str, np.ndarray] = {}
        for name in sorted(gt_names):
            g = np.asarray(
                nib.load(str(gt_dir / f"{name}.nii.gz")).dataobj
            ).astype(bool)
            g_crop = crop_gt(g, si_ax, crop_lo, crop_hi)
            if g_crop.any():
                gt_masks[name] = g_crop

        print(f"  GT structures in crop: {len(gt_masks)}")

        for tag in (tags or build_tags(DEFAULT_D_FRACS, DEFAULT_R_VALS)):
            d_str, r_str = tag.split("_")
            d_frac = int(d_str[1:]) / 100.0
            r_val  = int(r_str[1:]) / 100.0

            pred_dir = exp_dir / subj / crop_name / tag / "segmentations"
            if not pred_dir.exists():
                print(f"    [skip] {tag}: no segmentations directory")
                continue

            # Load predictions (full volume — not pre-cropped)
            all_preds: Dict[str, np.ndarray] = {}
            affine = None
            for pred_file in sorted(pred_dir.glob("*.nii.gz")):
                name = pred_file.stem.replace(".nii", "")
                img  = nib.load(str(pred_file))
                data = np.asarray(img.dataobj).astype(bool)
                if data.any():
                    all_preds[name] = data
                    if affine is None:
                        affine = img.affine

            if not all_preds:
                print(f"    [skip] {tag}: no predictions")
                continue

            # Apply erosion-based cleaning
            if method == "opening_lcc":
                cleaned, vox_removed = method_opening_lcc(all_preds, radius=radius)
            else:  # lcc_only
                cleaned, vox_removed = method_lcc_only(all_preds)

            tag_rows = []
            improved = degraded = 0

            for name in sorted(all_preds):
                # Predictions are already at crop-window size — no re-cropping needed.
                # Shape match against gt_masks confirms the structure is evaluable here.
                has_gt = (name in gt_masks and
                          all_preds[name].shape == gt_masks[name].shape)

                if has_gt:
                    gt  = gt_masks[name]
                    d0  = dice(all_preds[name], gt)
                    d_e = dice(cleaned[name], gt)
                    p0  = precision(all_preds[name], gt)
                    p_e = precision(cleaned[name], gt)
                    rc  = recall(all_preds[name], gt)
                    tp0, fp0, fn0 = tp_fp_fn(all_preds[name], gt)

                    delta = d_e - d0
                    if delta >  0.0001: improved += 1
                    if delta < -0.0001: degraded += 1

                    tag_rows.append({
                        "subject":           subj,
                        "crop":              crop_name,
                        "d_frac":            d_frac,
                        "r_val":             r_val,
                        "tag":               tag,
                        "structure":         name,
                        "has_gt":            True,
                        "method":            method,
                        "radius":            radius,
                        "dice_before":       round(d0,  5),
                        "dice_erosion":      round(d_e, 5),
                        "delta_erosion":     round(delta, 5),
                        "precision_before":  round(p0,  5),
                        "precision_erosion": round(p_e, 5),
                        "delta_prec_erosion":round(p_e - p0, 5),
                        "recall_before":     round(rc,  5),
                        "tp_before":         tp0,
                        "fp_before":         fp0,
                        "fn_before":         fn0,
                        "vox_before":        int(all_preds[name].sum()),
                        "vox_removed_erosion": vox_removed.get(name, 0),
                    })

            if tag_rows:
                mb = np.mean([r["dice_before"]  for r in tag_rows])
                me = np.mean([r["dice_erosion"] for r in tag_rows])
                print(f"    {tag}  n={len(tag_rows):2d} | "
                      f"before={mb:.4f} | erosion={me:.4f} ({me - mb:+.4f}) | "
                      f"improved={improved} degraded={degraded}")
            all_rows.extend(tag_rows)

    return all_rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

ER_COLOR = "#4878CF"   # blue for erosion method
R_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
D_COLORS = [plt.cm.viridis(i / 9) for i in range(10)]   # 10 d values


def make_plots(rows: List[dict], out_dir: Path, method: str, radius: int) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available — skipping plots")
        return

    df = pd.DataFrame(rows)
    if df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    method_label = f"{method} (r={radius})" if method == "opening_lcc" else method

    def grouped_bar(ax, pivot, colors, xlabel, ylabel, title, col_prefix="r"):
        x = np.arange(len(pivot.index))
        n_cols = len(pivot.columns)
        width  = 0.8 / n_cols
        for k, col in enumerate(pivot.columns):
            offset = (k - n_cols / 2 + 0.5) * width
            ax.bar(x + offset, pivot[col].values,
                   width=width * 0.9, color=colors[k % len(colors)],
                   alpha=0.85, label=f"{col_prefix}={col}")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pivot.index], fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)

    # 1. Mean Δ Dice by d, grouped by r
    pivot_d_r = df.groupby(["d_frac", "r_val"])["delta_erosion"].mean().unstack("r_val")
    fig, ax = plt.subplots(figsize=(12, 4))
    grouped_bar(ax, pivot_d_r, R_COLORS,
                "d (shift fraction)", f"Mean Δ Dice ({method_label})",
                f"Erosion baseline: Δ Dice by shift fraction, split by ghost intensity r")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_delta_by_d.png", dpi=150)
    plt.close()

    # 2. Mean Δ Dice by r, grouped by d
    pivot_r_d = df.groupby(["r_val", "d_frac"])["delta_erosion"].mean().unstack("d_frac")
    fig, ax = plt.subplots(figsize=(10, 4))
    grouped_bar(ax, pivot_r_d, D_COLORS,
                "r (ghost intensity)", f"Mean Δ Dice ({method_label})",
                f"Erosion baseline: Δ Dice by ghost intensity, split by shift fraction d",
                col_prefix="d")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_delta_by_r.png", dpi=150)
    plt.close()

    # 3. Heatmap: Δ Dice over (d, r)
    pivot_heat = df.groupby(["d_frac", "r_val"])["delta_erosion"].mean().unstack("r_val")
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(pivot_heat.values).max(), 1e-6)
    im = ax.imshow(pivot_heat.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot_heat.columns)))
    ax.set_xticklabels([f"r={v}" for v in pivot_heat.columns], fontsize=8)
    ax.set_yticks(range(len(pivot_heat.index)))
    ax.set_yticklabels([f"d={v}" for v in pivot_heat.index], fontsize=8)
    ax.set_title(f"Erosion baseline: Mean Δ Dice per (d, r)\n{method_label}", fontsize=10)
    for i in range(len(pivot_heat.index)):
        for j in range(len(pivot_heat.columns)):
            val = pivot_heat.values[i, j]
            ax.text(j, i, f"{val:+.4f}", ha="center", va="center", fontsize=6,
                    color="black" if abs(val) < 0.6 * vmax else "white")
    plt.colorbar(im, ax=ax, label="Mean Δ Dice")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_delta_d_r.png", dpi=150)
    plt.close()

    # 4. Mean Δ Precision heatmap
    pivot_prec = df.groupby(["d_frac", "r_val"])["delta_prec_erosion"].mean().unstack("r_val")
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(abs(pivot_prec.values).max(), 1e-6)
    im = ax.imshow(pivot_prec.values, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot_prec.columns)))
    ax.set_xticklabels([f"r={v}" for v in pivot_prec.columns], fontsize=8)
    ax.set_yticks(range(len(pivot_prec.index)))
    ax.set_yticklabels([f"d={v}" for v in pivot_prec.index], fontsize=8)
    ax.set_title(f"Erosion baseline: Mean Δ Precision per (d, r)\n{method_label}", fontsize=10)
    for i in range(len(pivot_prec.index)):
        for j in range(len(pivot_prec.columns)):
            val = pivot_prec.values[i, j]
            ax.text(j, i, f"{val:+.4f}", ha="center", va="center", fontsize=6,
                    color="black" if abs(val) < 0.6 * vmax else "white")
    plt.colorbar(im, ax=ax, label="Mean Δ Precision")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_prec_d_r.png", dpi=150)
    plt.close()

    # 5. Per-structure diverging bar (net improvement count)
    THRESH = 0.0001
    struct_counts = df.groupby("structure").apply(
        lambda g: pd.Series({
            "improved_dice": (g["delta_erosion"]      >  THRESH).sum(),
            "degraded_dice": (g["delta_erosion"]      < -THRESH).sum(),
            "improved_prec": (g["delta_prec_erosion"] >  THRESH).sum(),
            "degraded_prec": (g["delta_prec_erosion"] < -THRESH).sum(),
        })
    ).reset_index()
    struct_counts["net_dice"] = (struct_counts["improved_dice"]
                                 - struct_counts["degraded_dice"])
    struct_counts = struct_counts.sort_values("net_dice")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, imp_col, deg_col, title in [
        (axes[0], "improved_dice", "degraded_dice", "Dice"),
        (axes[1], "improved_prec", "degraded_prec", "Precision"),
    ]:
        y = np.arange(len(struct_counts))
        ax.barh(y,  struct_counts[imp_col].values, color="#55A868", alpha=0.85,
                label="improved")
        ax.barh(y, -struct_counts[deg_col].values, color="#C44E52", alpha=0.85,
                label="degraded")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(struct_counts["structure"].values, fontsize=5)
        ax.set_xlabel("# conditions", fontsize=9)
        ax.set_title(
            f"Erosion baseline: per-structure Δ{title} counts ({method_label})\n"
            f"sorted by net Dice", fontsize=9
        )
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "counts_per_structure.png", dpi=150)
    plt.close()

    # 6. Box plot distribution of Δ Dice
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(df["delta_erosion"].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor=ER_COLOR, alpha=0.6))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Δ Dice (erosion baseline)", fontsize=10)
    ax.set_title(f"Δ Dice distribution — erosion baseline ({method_label})\n"
                 f"all conditions & structures", fontsize=10)
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_delta_dice.png", dpi=150)
    plt.close()

    print(f"Plots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def make_report(rows: List[dict], out_dir: Path, method: str, radius: int) -> None:
    try:
        import pandas as pd
    except ImportError:
        return

    df = pd.DataFrame(rows)
    if df.empty:
        return

    THRESH = 0.0001
    n = len(df)
    method_label = f"{method} (radius={radius})" if method == "opening_lcc" else method

    def pct(k):
        return f"{100 * k / n:.1f}%"

    def section_table(grp_col, label):
        rows_md = []
        groups = sorted(df[grp_col].unique())
        rows_md.append(
            f"| {label} | Mean ΔDice | Median ΔDice | Imp↑ | Deg↓ | Net |"
            " Mean ΔPrec | Prec Imp↑ | Prec Deg↓ |"
        )
        rows_md.append("|---|---|---|---|---|---|---|---|---|")
        for g in groups:
            sub = df[df[grp_col] == g]
            k = len(sub)
            imp  = (sub["delta_erosion"]      >  THRESH).sum()
            deg  = (sub["delta_erosion"]      < -THRESH).sum()
            pimp = (sub["delta_prec_erosion"] >  THRESH).sum()
            pdeg = (sub["delta_prec_erosion"] < -THRESH).sum()
            rows_md.append(
                f"| {g}"
                f" | {sub['delta_erosion'].mean():+.5f}"
                f" | {sub['delta_erosion'].median():+.5f}"
                f" | {imp} ({100*imp/k:.1f}%)"
                f" | {deg} ({100*deg/k:.1f}%)"
                f" | {imp-deg:+d}"
                f" | {sub['delta_prec_erosion'].mean():+.5f}"
                f" | {pimp} ({100*pimp/k:.1f}%)"
                f" | {pdeg} ({100*pdeg/k:.1f}%) |"
            )
        return "\n".join(rows_md)

    imp_d  = (df["delta_erosion"]      >  THRESH).sum()
    deg_d  = (df["delta_erosion"]      < -THRESH).sum()
    imp_p  = (df["delta_prec_erosion"] >  THRESH).sum()
    deg_p  = (df["delta_prec_erosion"] < -THRESH).sum()

    lines = [
        "# Erosion Baseline Evaluation Report",
        "",
        f"**Method:** {method_label}  ",
        f"**Total structure×condition pairs:** {n}  ",
        "",
        "---",
        "## Overall",
        "",
        "| Metric | Mean | Median | Std | Improved | Degraded | Net |",
        "|--------|------|--------|-----|----------|----------|-----|",
        f"| Δ Dice | {df['delta_erosion'].mean():+.5f} | {df['delta_erosion'].median():+.5f}"
        f" | {df['delta_erosion'].std():.5f} | {imp_d} ({pct(imp_d)}) | {deg_d} ({pct(deg_d)}) | {imp_d-deg_d:+d} |",
        f"| Δ Precision | {df['delta_prec_erosion'].mean():+.5f} | {df['delta_prec_erosion'].median():+.5f}"
        f" | {df['delta_prec_erosion'].std():.5f} | {imp_p} ({pct(imp_p)}) | {deg_p} ({pct(deg_p)}) | {imp_p-deg_p:+d} |",
        "",
        "---",
        "## By shift fraction (d)",
        "",
        section_table("d_frac", "d"),
        "",
        "---",
        "## By ghost intensity (r)",
        "",
        section_table("r_val", "r"),
        "",
        "---",
        "## By crop region",
        "",
        section_table("crop", "crop"),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"Report saved to {out_dir / 'report.md'}")


# ---------------------------------------------------------------------------
# Radius sweep
# ---------------------------------------------------------------------------

SWEEP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]
CROP_MARKERS = {
    "brain_to_heart":  "o",
    "heart_to_kidney": "s",
    "kidney_to_hip":   "^",
    "overall":         "D",
}
CROP_LABELS = {
    "brain_to_heart":  "brain→heart",
    "heart_to_kidney": "heart→kidney",
    "kidney_to_hip":   "kidney→hip",
    "overall":         "overall",
}


def make_sweep_plots(sweep_results: dict, out_dir: Path) -> None:
    """Plot mean ΔDice and mean ΔPrecision vs erosion radius.

    sweep_results: {label → list[dict]}  where label is e.g. "lcc_only" or
    "opening_lcc r=2".  The key "radius_int" in each label maps to the x-axis.
    Actually we pass sweep_results as OrderedDict keyed by (method, radius_int).
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available — skipping sweep plots")
        return

    # Build a combined dataframe with a radius_val column (0 = lcc_only)
    frames = []
    for (method, radius_int), rows in sweep_results.items():
        df = pd.DataFrame(rows)
        df["radius_val"] = 0 if method == "lcc_only" else radius_int
        df["sweep_label"] = "LCC only" if method == "lcc_only" else f"r={radius_int}"
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    radii_sorted = sorted(all_df["radius_val"].unique())
    x = np.arange(len(radii_sorted))
    x_labels = ["LCC\nonly" if r == 0 else str(r) for r in radii_sorted]

    # ---- 1. Overall + per-crop line plots: ΔDice and ΔPrecision side by side ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_col, metric_name in [
        (axes[0], "delta_erosion",      "Mean Δ Dice"),
        (axes[1], "delta_prec_erosion", "Mean Δ Precision"),
    ]:
        # overall line
        overall_means = [
            all_df[all_df["radius_val"] == r][metric_col].mean()
            for r in radii_sorted
        ]
        ax.plot(x, overall_means, marker=CROP_MARKERS["overall"],
                color="black", linewidth=2, markersize=7,
                label=CROP_LABELS["overall"], zorder=5)

        # per-crop lines
        for i, crop in enumerate(["brain_to_heart", "heart_to_kidney", "kidney_to_hip"]):
            crop_df = all_df[all_df["crop"] == crop]
            means = [
                crop_df[crop_df["radius_val"] == r][metric_col].mean()
                for r in radii_sorted
            ]
            ax.plot(x, means, marker=CROP_MARKERS[crop],
                    color=SWEEP_COLORS[i], linewidth=1.4, markersize=6,
                    alpha=0.85, label=CROP_LABELS[crop])

        ax.axhline(0, color="gray", lw=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel("Erosion radius (voxels)", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(f"Erosion baseline radius sweep\n{metric_name} vs radius",
                     fontsize=10)
        ax.legend(fontsize=8)

    plt.suptitle("Erosion baseline: effect of ball radius on Dice and Precision\n"
                 "(radius=0 → LCC-only, no erosion)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "radius_sweep.png", dpi=150)
    plt.close()

    # ---- 2. Summary table printed to stdout ----
    print("\n" + "=" * 70)
    print("  RADIUS SWEEP SUMMARY")
    print("=" * 70)
    print(f"  {'Label':<18} {'Mean ΔDice':>12} {'Mean ΔPrec':>12} "
          f"{'Imp↑':>8} {'Deg↓':>8} {'Net':>6}")
    print("  " + "-" * 68)
    THRESH = 0.0001
    for r in radii_sorted:
        sub  = all_df[all_df["radius_val"] == r]
        lbl  = "LCC only" if r == 0 else f"opening r={r}"
        imp  = (sub["delta_erosion"]      >  THRESH).sum()
        deg  = (sub["delta_erosion"]      < -THRESH).sum()
        print(f"  {lbl:<18} {sub['delta_erosion'].mean():>+12.5f} "
              f"{sub['delta_prec_erosion'].mean():>+12.5f} "
              f"{imp:>8d} {deg:>8d} {imp-deg:>+6d}")
    print("=" * 70 + "\n")

    print(f"Sweep plot saved to {out_dir / 'radius_sweep.png'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir",  default="data/datasets/TotalsegmentatorMRI_dataset_v200",
                   help="Root of the TotalSegmentator MRI dataset")
    p.add_argument("--exp_dir",   default="data/wraparound_experiments/wraparound_v4",
                   help="Root of the wraparound experiment (contains subject subdirs)")
    p.add_argument("--subject",   default="s0175",
                   help="Single subject (ignored if --subjects is given)")
    p.add_argument("--subjects",  nargs="+", default=None,
                   help="One or more subjects, e.g. s0175 s0236 s0219")
    p.add_argument("--out_dir",   default="data/wraparound_experiments/wraparound_v4_eval/erosion_baseline",
                   help="Output directory for CSV, plots, and report")
    p.add_argument("--method",    default="opening_lcc",
                   choices=["opening_lcc", "lcc_only"],
                   help="opening_lcc: erode→LCC→dilate (default);  "
                        "lcc_only: just keep the largest connected component")
    p.add_argument("--radius",    type=int, default=2,
                   help="Ball radius in voxels for erosion/dilation (default: 2). "
                        "Ignored for lcc_only.")
    p.add_argument("--radii",     type=int, nargs="+", default=None,
                   help="Sweep mode: evaluate multiple radii and produce a comparison "
                        "plot. E.g. --radii 1 2 3 4 5. lcc_only is always included "
                        "as the radius=0 reference point. Overrides --radius and "
                        "--method when given.")
    p.add_argument("--d_fracs",   type=float, nargs="+", default=None,
                   help="Shift fractions to evaluate, e.g. 0.05 0.25 0.50 (default: all 10)")
    p.add_argument("--r_vals",    type=float, nargs="+", default=None,
                   help="Ghost intensities to evaluate, e.g. 0.25 1.00 (default: all 4)")
    args = p.parse_args()

    d_fracs = args.d_fracs or DEFAULT_D_FRACS
    r_vals  = args.r_vals  or DEFAULT_R_VALS
    tags    = build_tags(d_fracs, r_vals)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sweep mode: evaluate multiple radii in one run
    # ------------------------------------------------------------------
    if args.radii:
        radii = sorted(set(args.radii))
        print(f"Sweep mode: radii={radii}  (+lcc_only as reference)")
        print(f"Evaluating {len(tags)} conditions per radius: "
              f"d={[f'{d:.2f}' for d in d_fracs]}  r={list(r_vals)}")

        # Use an ordered dict keyed by (method, radius_int)
        sweep_results = {}

        # Always include lcc_only as the radius=0 baseline
        args.method = "lcc_only"
        args.radius = 0
        print(f"\n{'#' * 70}")
        print(f"  Running lcc_only (radius=0 reference)")
        lcc_rows = run_evaluation(args, tags=tags)
        sweep_results[("lcc_only", 0)] = lcc_rows
        lcc_dir = out_dir / "lcc_only"
        lcc_dir.mkdir(parents=True, exist_ok=True)
        if lcc_rows:
            with open(lcc_dir / "results.csv", "w", newline="") as f:
                csv.DictWriter(f, fieldnames=list(lcc_rows[0].keys())).writeheader()
                csv.DictWriter(f, fieldnames=list(lcc_rows[0].keys())).writerows(lcc_rows)
            make_plots(lcc_rows, lcc_dir, method="lcc_only", radius=0)
            make_report(lcc_rows, lcc_dir, method="lcc_only", radius=0)

        # One run per radius
        for r in radii:
            args.method = "opening_lcc"
            args.radius = r
            print(f"\n{'#' * 70}")
            print(f"  Running opening_lcc  radius={r}")
            rows = run_evaluation(args, tags=tags)
            sweep_results[("opening_lcc", r)] = rows
            r_dir = out_dir / f"radius_{r}"
            r_dir.mkdir(parents=True, exist_ok=True)
            if rows:
                with open(r_dir / "results.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
                make_plots(rows, r_dir, method="opening_lcc", radius=r)
                make_report(rows, r_dir, method="opening_lcc", radius=r)

        make_sweep_plots(sweep_results, out_dir)

    # ------------------------------------------------------------------
    # Single-radius mode (default)
    # ------------------------------------------------------------------
    else:
        print(f"Method: {args.method}  radius: {args.radius}")
        print(f"Evaluating {len(tags)} conditions: "
              f"d={[f'{d:.2f}' for d in d_fracs]}  r={list(r_vals)}")

        rows = run_evaluation(args, tags=tags)

        if rows:
            csv_path = out_dir / "results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nResults CSV → {csv_path}")
            make_plots(rows, out_dir, method=args.method, radius=args.radius)
            make_report(rows, out_dir, method=args.method, radius=args.radius)
        else:
            print("No rows produced — check paths and subject names.")


if __name__ == "__main__":
    main()
