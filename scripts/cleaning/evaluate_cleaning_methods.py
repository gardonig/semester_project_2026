"""
Evaluate three poset cleaning methods on all 18 MRI wrap-around artifact conditions.

Methods
-------
1. Unidirectional (current): for each pair (i above j), remove non-LCC components
   of i that sit entirely below j's LCC inferior boundary.

2. Symmetric: same as (1), plus remove non-LCC components of j that sit entirely
   above i's LCC superior boundary.  Catches both artifact directions.

3. Middle-out + spatial prior: like (2), but selects the "real" component using the
   atlas CoM prior (closest centroid to expected atlas position) rather than pure LCC.
   Processes constraint pairs ordered from most-central structures outward, so
   already-cleaned central anchors inform the periphery.

Usage
-----
    python scripts/cleaning/evaluate_cleaning_methods.py \
        --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
        --exp_dir   data/experiments/wraparound \
        --poset     data/structures/totalseg_v2_empirical_poset.json \
        --com       data/structures/totalseg_v2_com.json \
        --subject   s0175 \
        --out_dir   data/experiments/wraparound_cleaning_eval \
        --threshold 0.95
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy.ndimage import label as cc_label

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from anatomy_poset.core.io import load_poset_from_json, PosetFromJson

# ---------------------------------------------------------------------------
# Geometry helpers
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


def centroid_1d(mask: np.ndarray, si_ax: int) -> Optional[float]:
    idx = np.where(mask.any(axis=tuple(ax for ax in range(mask.ndim) if ax != si_ax)))[0]
    return float(idx.mean()) if len(idx) > 0 else None


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return float(2.0 * (p & g).sum() / denom) if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Connected component selection
# ---------------------------------------------------------------------------

def get_components(mask: np.ndarray):
    """Returns (labeled_array, n_components, sizes_array, lcc_label)."""
    if not mask.any():
        return None, 0, None, None
    labeled, n = cc_label(mask)
    if n == 0:
        return labeled, 0, None, None
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    lcc_label = int(sizes.argmax())
    return labeled, n, sizes, lcc_label


def select_by_lcc(mask: np.ndarray) -> np.ndarray:
    labeled, n, sizes, lcc = get_components(mask)
    if n == 0:
        return mask.copy()
    return (labeled == lcc).astype(bool)


def select_by_prior(mask: np.ndarray, si_ax: int, expected_local: float,
                    size_dominance: float = 5.0) -> np.ndarray:
    """Pick the component closest to expected_local, but only override the LCC
    when a competing component is within size_dominance× of the LCC.
    This prevents the prior from accidentally picking a tiny noise CC and
    marking the true large component for removal."""
    labeled, n, sizes, lcc = get_components(mask)
    if n <= 1:
        return mask.copy()

    # Sort labels by size descending (excluding background=0)
    sorted_labels = np.argsort(-sizes[1:]) + 1   # labels in size order
    top2_sizes = [sizes[sorted_labels[0]], sizes[sorted_labels[1]]]

    # If the LCC is dominant (>5× the 2nd largest), trust it unconditionally
    if top2_sizes[0] >= size_dominance * top2_sizes[1]:
        return (labeled == lcc).astype(bool)

    # Multiple similarly-sized CCs: use prior to pick the most plausible one
    best_label, best_dist = lcc, float("inf")
    for lab in range(1, n + 1):
        c = centroid_1d((labeled == lab), si_ax)
        if c is not None:
            dist = abs(c - expected_local)
            if dist < best_dist:
                best_dist, best_label = dist, lab
    return (labeled == best_label).astype(bool)


# ---------------------------------------------------------------------------
# Method 1: unidirectional (current)
# ---------------------------------------------------------------------------

def method1_unidirectional(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    cleaned = {n: m.copy() for n, m in predictions.items()}
    removed = {n: 0 for n in predictions}
    cc_cache: Dict[str, tuple] = {}

    pairs = _get_pairs(poset, threshold)
    for name_i, name_j in pairs:
        if name_i not in cleaned or name_j not in cleaned:
            continue
        if name_j not in cc_cache:
            labeled, n, sizes, lcc = get_components(cleaned[name_j])
            cc_cache[name_j] = (labeled, n, sizes, lcc)
        _, _, _, lcc = cc_cache[name_j]
        if lcc is None:
            continue
        labeled_j = cc_cache[name_j][0]
        anchor = (labeled_j == lcc).astype(bool)
        ext_j = axis_extent(anchor, si_ax)
        if ext_j is None:
            continue
        _remove_violated_components(cleaned, removed, name_i, si_ax, si_sign,
                                    below_limit=ext_j[0], above_limit=None,
                                    cc_cache=cc_cache)
    return cleaned, removed


# ---------------------------------------------------------------------------
# Method 2: symmetric
# ---------------------------------------------------------------------------

def method2_symmetric(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    cleaned = {n: m.copy() for n, m in predictions.items()}
    removed = {n: 0 for n in predictions}
    cc_cache: Dict[str, tuple] = {}

    pairs = _get_pairs(poset, threshold)
    for name_i, name_j in pairs:
        if name_i not in cleaned or name_j not in cleaned:
            continue

        def lcc_extent(name):
            if name not in cc_cache:
                cc_cache[name] = get_components(cleaned[name])
            labeled, n, sizes, lcc = cc_cache[name]
            if lcc is None:
                return None
            return axis_extent((labeled == lcc).astype(bool), si_ax)

        ext_j = lcc_extent(name_j)
        ext_i = lcc_extent(name_i)

        if ext_j is not None:
            _remove_violated_components(cleaned, removed, name_i, si_ax, si_sign,
                                        below_limit=ext_j[0], above_limit=None,
                                        cc_cache=cc_cache)
        if ext_i is not None:
            _remove_violated_components(cleaned, removed, name_j, si_ax, si_sign,
                                        below_limit=None, above_limit=ext_i[1],
                                        cc_cache=cc_cache)
    return cleaned, removed


# ---------------------------------------------------------------------------
# Method 3: middle-out + spatial prior
# ---------------------------------------------------------------------------

def method3_middle_out_prior(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
    com_lookup: Dict[str, float],   # name → com_vertical (0–100, 100=most superior)
    crop_lo: int,
    crop_hi: int,
    full_height: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    cleaned = {n: m.copy() for n, m in predictions.items()}
    removed = {n: 0 for n in predictions}
    N = crop_hi - crop_lo

    def expected_local(name: str) -> Optional[float]:
        if name not in com_lookup:
            return None
        com_v = com_lookup[name]
        # com_vertical is 0–100, 100 = most superior
        # For si_sign=+1 ('S'): high index = superior → expected_vox_global ≈ com_v/100 * full_height
        # For si_sign=-1 ('I'): low index = superior → expected_vox_global ≈ (1-com_v/100) * full_height
        if si_sign == +1:
            vox_global = (com_v / 100.0) * full_height
        else:
            vox_global = (1.0 - com_v / 100.0) * full_height
        local = vox_global - crop_lo
        return local  # may be outside [0, N) → structure not expected in this crop

    # Compute crop centre in atlas-normalized coordinates
    crop_center_global = crop_lo + N / 2.0

    # Order pairs from most-central pair (both structures near crop centre) to peripheral
    pairs = _get_pairs(poset, threshold)

    def pair_centrality(pair):
        ni, nj = pair
        ei = expected_local(ni)
        ej = expected_local(nj)
        ei = ei if ei is not None else -9999
        ej = ej if ej is not None else -9999
        return abs((ei + ej) / 2.0 - N / 2.0)

    pairs_sorted = sorted(pairs, key=pair_centrality)

    cc_cache: Dict[str, tuple] = {}

    for name_i, name_j in pairs_sorted:
        if name_i not in cleaned or name_j not in cleaned:
            continue

        exp_i = expected_local(name_i)
        exp_j = expected_local(name_j)

        def get_anchor(name, exp):
            if not cleaned[name].any():
                return None, None
            if exp is not None and 0 <= exp < N:
                anchor = select_by_prior(cleaned[name], si_ax, exp)
            else:
                if name not in cc_cache:
                    cc_cache[name] = get_components(cleaned[name])
                labeled, _, sizes, lcc = cc_cache[name]
                anchor = (labeled == lcc).astype(bool) if lcc is not None else None
            if anchor is None:
                return None, None
            return anchor, axis_extent(anchor, si_ax)

        anchor_i, ext_i = get_anchor(name_i, exp_i)
        anchor_j, ext_j = get_anchor(name_j, exp_j)

        if ext_j is not None:
            _remove_violated_components(cleaned, removed, name_i, si_ax, si_sign,
                                        below_limit=ext_j[0], above_limit=None,
                                        cc_cache=cc_cache, anchor_override=anchor_i)
        if ext_i is not None:
            _remove_violated_components(cleaned, removed, name_j, si_ax, si_sign,
                                        below_limit=None, above_limit=ext_i[1],
                                        cc_cache=cc_cache, anchor_override=anchor_j)

    return cleaned, removed


# ---------------------------------------------------------------------------
# Shared removal helper
# ---------------------------------------------------------------------------

def _get_pairs(poset: PosetFromJson, threshold: float) -> List[Tuple[str, str]]:
    n = len(poset.structures)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cell = poset.matrix_vertical[i][j]
            if cell is not None and cell >= threshold:
                pairs.append((poset.structures[i].name, poset.structures[j].name))
    return pairs


def _remove_violated_components(
    cleaned: Dict[str, np.ndarray],
    removed: Dict[str, int],
    name: str,
    si_ax: int,
    si_sign: int,
    below_limit: Optional[int],
    above_limit: Optional[int],
    cc_cache: Dict[str, tuple],
    anchor_override: Optional[np.ndarray] = None,
) -> bool:
    """Returns True if any voxels were removed (caller should invalidate cc_cache[name])."""
    mask = cleaned[name]
    if not mask.any():
        return False

    if name not in cc_cache:
        labeled, n, sizes, lcc_label = get_components(mask)
        cc_cache[name] = (labeled, n, sizes, lcc_label)
    labeled, n, sizes, lcc_label = cc_cache[name]
    if n == 0:
        return False

    if anchor_override is not None:
        overlap = np.bincount(labeled[anchor_override].ravel(), minlength=n + 1)
        overlap[0] = 0
        anchor_label = int(overlap.argmax()) if overlap.max() > 0 else lcc_label
    else:
        anchor_label = lcc_label

    changed = False
    for comp_label in range(1, n + 1):
        if comp_label == anchor_label:
            continue
        comp = labeled == comp_label
        ext = axis_extent(comp, si_ax)
        if ext is None:
            continue
        c_min, c_max = ext

        violated = False
        if below_limit is not None:
            if si_sign == +1:
                violated = violated or (c_max < below_limit)
            else:
                violated = violated or (c_min > below_limit)
        if above_limit is not None:
            if si_sign == +1:
                violated = violated or (c_min > above_limit)
            else:
                violated = violated or (c_max < above_limit)

        if violated:
            removed[name] += int(comp.sum())
            cleaned[name][comp] = False
            changed = True

    if changed:
        cc_cache.pop(name, None)   # invalidate so next call re-labels
    return changed


# ---------------------------------------------------------------------------
# Crop GT to prediction space
# ---------------------------------------------------------------------------

def crop_gt(gt_full: np.ndarray, si_ax: int, si_sign: int,
            crop_lo: int, crop_hi: int) -> np.ndarray:
    slices = [slice(None)] * gt_full.ndim
    slices[si_ax] = slice(crop_lo, crop_hi)
    return gt_full[tuple(slices)]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

CROPS = [
    ("brain_to_heart",  "brain",       "heart"),
    ("heart_to_kidney", "heart",       "kidney_left"),
    ("kidney_to_hip",   "kidney_left", "hip_left"),
]
FALLBACKS = {"kidney_left": "kidney_right", "hip_left": "hip_right"}
TAGS = [f"d{int(d*100):03d}_r{int(r*100):03d}"
        for d in (0.10, 0.25, 0.50) for r in (0.50, 1.00)]

MARGIN = 5


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
    lo = max(0, min(lo_vals) - MARGIN)
    hi = min(full_height, max(hi_vals) + MARGIN)
    return lo, hi


def run_evaluation(args) -> List[dict]:
    poset = load_poset_from_json(args.poset)
    with open(args.com) as f:
        com_data = json.load(f)
    com_lookup = {s["name"]: s["com_vertical"] for s in com_data["structures"]}

    data_dir = Path(args.data_dir)
    exp_dir  = Path(args.exp_dir)
    subj     = args.subject
    gt_dir   = data_dir / subj / "segmentations"

    # Load full MRI for geometry
    mri_full = nib.load(str(data_dir / subj / "mri.nii.gz"))
    si_ax, si_sign = get_si_info(mri_full.affine)
    full_height = mri_full.shape[si_ax]

    # Evaluable structure names (GT ∩ pred ∩ poset)
    poset_names  = {s.name for s in poset.structures}
    gt_names     = {p.stem.replace(".nii", "") for p in gt_dir.glob("*.nii.gz")}
    eval_names   = sorted(poset_names & gt_names)

    all_rows = []

    for crop_name, anc_a, anc_b in CROPS:
        crop_lo, crop_hi = compute_crop_window(gt_dir, (anc_a, anc_b), si_ax, full_height)
        N = crop_hi - crop_lo
        print(f"\n{'='*70}")
        print(f"  Crop: {crop_name}  lo={crop_lo} hi={crop_hi} N={N}")

        # Pre-load and crop GT masks (constant across tags)
        gt_masks = {}
        for name in eval_names:
            g = np.asarray(nib.load(str(gt_dir / f"{name}.nii.gz")).dataobj).astype(bool)
            gt_crop = crop_gt(g, si_ax, si_sign, crop_lo, crop_hi)
            if gt_crop.any():
                gt_masks[name] = gt_crop
        eval_here = sorted(gt_masks)
        print(f"  Evaluable structures in crop: {len(eval_here)}")

        for tag in TAGS:
            d_str, r_str = tag.split("_")
            d_frac = int(d_str[1:]) / 100.0
            r_val  = int(r_str[1:]) / 100.0
            pred_dir = exp_dir / subj / crop_name / tag / "segmentations"
            if not pred_dir.exists():
                print(f"    [skip] {tag}: no segmentations directory")
                continue

            # Load ALL predictions (needed so cleaning can use full poset context)
            all_preds: Dict[str, np.ndarray] = {}
            affine = None
            for pred_file in sorted(pred_dir.glob("*.nii.gz")):
                name = pred_file.stem.replace(".nii", "")
                img = nib.load(str(pred_file))
                data = np.asarray(img.dataobj).astype(bool)
                if data.any():
                    all_preds[name] = data
                    if affine is None:
                        affine = img.affine

            if not all_preds:
                print(f"    [skip] {tag}: no predictions")
                continue

            pred_si_ax, pred_si_sign = get_si_info(affine)

            # Apply all three methods on ALL predictions
            c1, r1 = method1_unidirectional(all_preds, poset, pred_si_ax, pred_si_sign, args.threshold)
            c2, r2 = method2_symmetric(all_preds, poset, pred_si_ax, pred_si_sign, args.threshold)
            c3, r3 = method3_middle_out_prior(all_preds, poset, pred_si_ax, pred_si_sign, args.threshold,
                                              com_lookup, crop_lo, crop_hi, full_height)

            # Evaluate Dice for GT-evaluable structures
            improved = [0, 0, 0]
            degraded = [0, 0, 0]
            fp_removed = [0, 0, 0]   # false positive voxels removed (no GT in crop)
            tag_rows = []

            for name in sorted(all_preds):
                has_gt = name in gt_masks and all_preds[name].shape == gt_masks[name].shape

                if has_gt:
                    gt = gt_masks[name]
                    d0 = dice(all_preds[name], gt)
                    d1 = dice(c1[name], gt)
                    d2 = dice(c2[name], gt)
                    d3 = dice(c3[name], gt)

                    for k, dk in enumerate([d1, d2, d3]):
                        delta = dk - d0
                        if delta > 0.0001:   improved[k] += 1
                        elif delta < -0.0001: degraded[k] += 1

                    tag_rows.append({
                        "subject": subj,
                        "crop": crop_name,
                        "d_frac": d_frac,
                        "r_val": r_val,
                        "tag": tag,
                        "structure": name,
                        "has_gt": True,
                        "dice_before": round(d0, 5),
                        "dice_m1":     round(d1, 5),
                        "dice_m2":     round(d2, 5),
                        "dice_m3":     round(d3, 5),
                        "delta_m1":    round(d1 - d0, 5),
                        "delta_m2":    round(d2 - d0, 5),
                        "delta_m3":    round(d3 - d0, 5),
                        "vox_before":  int(all_preds[name].sum()),
                        "vox_removed_m1": r1.get(name, 0),
                        "vox_removed_m2": r2.get(name, 0),
                        "vox_removed_m3": r3.get(name, 0),
                    })
                else:
                    # No GT → any prediction is a false positive; measure reduction
                    for k, (removed_dict, cln) in enumerate(
                            [(r1, c1), (r2, c2), (r3, c3)]):
                        fp_removed[k] += removed_dict.get(name, 0)

            if tag_rows:
                mb  = np.mean([r["dice_before"] for r in tag_rows])
                m1  = np.mean([r["dice_m1"]     for r in tag_rows])
                m2  = np.mean([r["dice_m2"]     for r in tag_rows])
                m3  = np.mean([r["dice_m3"]     for r in tag_rows])
                print(f"    {tag}  n={len(tag_rows):2d} | "
                      f"before={mb:.4f} | "
                      f"M1={m1:.4f}({m1-mb:+.4f}) "
                      f"M2={m2:.4f}({m2-mb:+.4f}) "
                      f"M3={m3:.4f}({m3-mb:+.4f})")
                print(f"           improved: M1={improved[0]} M2={improved[1]} M3={improved[2]} | "
                      f"degraded: M1={degraded[0]} M2={degraded[1]} M3={degraded[2]} | "
                      f"FP removed: M1={fp_removed[0]} M2={fp_removed[1]} M3={fp_removed[2]}")
            all_rows.extend(tag_rows)

    return all_rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(rows: List[dict], out_dir: Path) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods  = ["M1 (unidirectional)", "M2 (symmetric)", "M3 (middle-out+prior)"]
    delta_cols = ["delta_m1", "delta_m2", "delta_m3"]
    colors   = ["#4C72B0", "#DD8452", "#55A868"]

    # --- 1. Box plot: Δ Dice per method ---
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df[col].values for col in delta_cols]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Δ Dice (after − before)")
    ax.set_title("Dice improvement distribution across all conditions & structures")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_delta_dice.png", dpi=150)
    plt.close()

    # --- 2. Bar chart: mean Δ Dice grouped by d_frac ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, col, method, color in zip(axes, delta_cols, methods, colors):
        grp = df.groupby("d_frac")[col].mean()
        ax.bar(grp.index.astype(str), grp.values, color=color, alpha=0.8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(method, fontsize=9)
        ax.set_xlabel("d (shift fraction)")
        ax.set_ylabel("Mean Δ Dice")
    plt.suptitle("Effect of shift fraction on mean Dice improvement", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "bar_delta_by_d.png", dpi=150)
    plt.close()

    # --- 3. Bar chart: mean Δ Dice grouped by crop ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, col, method, color in zip(axes, delta_cols, methods, colors):
        grp = df.groupby("crop")[col].mean()
        ax.bar(range(len(grp)), grp.values, color=color, alpha=0.8)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels([c.replace("_to_", "→") for c in grp.index], fontsize=7, rotation=15)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(method, fontsize=9)
        ax.set_ylabel("Mean Δ Dice")
    plt.suptitle("Effect of crop region on mean Dice improvement", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "bar_delta_by_crop.png", dpi=150)
    plt.close()

    # --- 4. Scatter: M1 vs M2 vs M3 per-structure mean delta ---
    mean_per_struct = df.groupby("structure")[delta_cols].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (xa, xm), (ya, ym) in zip(axes,
            [("delta_m1", "M1"), ("delta_m1", "M1")],
            [("delta_m2", "M2"), ("delta_m3", "M3")]):
        ax.scatter(mean_per_struct[xa], mean_per_struct[ya], s=18, alpha=0.6)
        lim = max(abs(mean_per_struct[[xa, ya]].values).max() * 1.1, 0.01)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.6, alpha=0.4)
        ax.set_xlabel(f"Mean Δ Dice {xm}")
        ax.set_ylabel(f"Mean Δ Dice {ym}")
        ax.set_title(f"{xm} vs {ym} per structure")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_method_comparison.png", dpi=150)
    plt.close()

    # --- 5. Summary table printed ---
    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL 18 CONDITIONS")
    print("="*70)
    n_total = len(df)
    for col, name in zip(delta_cols, methods):
        improved = (df[col] > 0.0001).sum()
        degraded = (df[col] < -0.0001).sum()
        print(f"\n{name}")
        print(f"  Mean Δ Dice : {df[col].mean():+.5f}  ± {df[col].std():.5f}")
        print(f"  Median Δ   : {df[col].median():+.5f}")
        print(f"  Improved   : {improved}/{n_total} ({100*improved/n_total:.1f}%)")
        print(f"  Degraded   : {degraded}/{n_total} ({100*degraded/n_total:.1f}%)")

    print(f"\nPlots saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir",  default="data/datasets/TotalsegmentatorMRI_dataset_v200")
    p.add_argument("--exp_dir",   default="data/experiments/wraparound")
    p.add_argument("--poset",     default="data/structures/totalseg_v2_empirical_poset.json")
    p.add_argument("--com",       default="data/structures/totalseg_v2_com.json")
    p.add_argument("--subject",   default="s0175")
    p.add_argument("--out_dir",   default="data/experiments/wraparound_cleaning_eval")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Min poset probability to count as hard constraint (default: 0.95)")
    args = p.parse_args()

    rows = run_evaluation(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults CSV → {csv_path}")
        make_plots(rows, out_dir)


if __name__ == "__main__":
    main()
