"""
Visualize real CM3 success/failure modes from wraparound_v4 data.

This script:
1) mines partial CSVs to select one failure and one good case,
2) reruns CM3 (method3) with event-level logging,
3) plots coronal views: **MRI = one slice**; **segmentation masks = A/P silhouette**
   (logical OR along A/P) onto the LR×SI plane so full 3D extent is visible.

Outputs:
  results/cm3_real_cases/cm3_real_good_bad_examples.png
  results/cm3_real_cases/cm3_real_case_details.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import aff2axcodes

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "cleaning"))

from anatomy_poset.core.io import load_poset_from_json, PosetFromJson
from evaluate_cleaning_methods import (
    get_si_info,
    get_components,
    axis_extent,
    _get_pairs,
    compute_crop_window,
    crop_gt,
    dice,
    precision,
)


DATA_DIR = PROJECT_ROOT / "data" / "datasets" / "TotalsegmentatorMRI_dataset_v200"
EXP_DIR = PROJECT_ROOT / "data" / "experiments" / "wraparound_v4"
EVAL_T100_DIR = PROJECT_ROOT / "data" / "experiments" / "wraparound_v4_eval" / "t100"
POSET_PATH = PROJECT_ROOT / "data" / "posets" / "empirical" / "totalseg_mri_empirical_poset.json"
OUT_DIR = PROJECT_ROOT / "results" / "cm3_real_cases"
THRESHOLD = 1.00

CROP_ANCHORS = {
    "brain_to_heart": ("brain", "heart"),
    "heart_to_kidney": ("heart", "kidney_left"),
    "kidney_to_hip": ("kidney_left", "hip_left"),
}


@dataclass
class CaseRow:
    subject: str
    crop: str
    tag: str
    structure: str
    delta_pc: float
    precision_before: float
    precision_pc: float
    vox_removed_pc: int


def choose_cases(df: pd.DataFrame) -> Tuple[CaseRow, CaseRow]:
    # Failure mode from discussion: humerus_right in brain_to_heart with strong TP loss.
    fail = (
        df[
            (df["structure"] == "humerus_right")
            & (df["crop"] == "brain_to_heart")
            & (df["vox_removed_pc"] > 0)
            & (df["delta_pc"] < -0.2)
        ]
        .sort_values(["r_val", "delta_pc"], ascending=[True, True])
        .iloc[0]
    )

    # Good mode from discussion: kidney_to_hip pure-ghost cleanup.
    good = (
        df[
            (df["structure"] == "femur_right")
            & (df["crop"] == "kidney_to_hip")
            & (df["vox_removed_pc"] > 0)
            & (df["delta_pc"] > 0)
        ]
        .sort_values("delta_pc", ascending=False)
        .iloc[0]
    )

    return (
        CaseRow(
            subject=str(fail["subject"]),
            crop=str(fail["crop"]),
            tag=str(fail["tag"]),
            structure=str(fail["structure"]),
            delta_pc=float(fail["delta_pc"]),
            precision_before=float(fail["precision_before"]),
            precision_pc=float(fail["precision_pc"]),
            vox_removed_pc=int(fail["vox_removed_pc"]),
        ),
        CaseRow(
            subject=str(good["subject"]),
            crop=str(good["crop"]),
            tag=str(good["tag"]),
            structure=str(good["structure"]),
            delta_pc=float(good["delta_pc"]),
            precision_before=float(good["precision_before"]),
            precision_pc=float(good["precision_pc"]),
            vox_removed_pc=int(good["vox_removed_pc"]),
        ),
    )


def choose_cases_d20(df: pd.DataFrame) -> Tuple[CaseRow, CaseRow]:
    d20 = df[(df["d_frac"] == 0.20) & (df["has_gt"] == True) & (df["vox_removed_pc"] > 0)]
    fail = d20.sort_values("delta_pc", ascending=True).iloc[0]
    good = d20.sort_values("delta_pc", ascending=False).iloc[0]
    return (
        CaseRow(
            subject=str(fail["subject"]),
            crop=str(fail["crop"]),
            tag=str(fail["tag"]),
            structure=str(fail["structure"]),
            delta_pc=float(fail["delta_pc"]),
            precision_before=float(fail["precision_before"]),
            precision_pc=float(fail["precision_pc"]),
            vox_removed_pc=int(fail["vox_removed_pc"]),
        ),
        CaseRow(
            subject=str(good["subject"]),
            crop=str(good["crop"]),
            tag=str(good["tag"]),
            structure=str(good["structure"]),
            delta_pc=float(good["delta_pc"]),
            precision_before=float(good["precision_before"]),
            precision_pc=float(good["precision_pc"]),
            vox_removed_pc=int(good["vox_removed_pc"]),
        ),
    )


def choose_more_cases_d20(df: pd.DataFrame, n_fail: int = 2, n_good: int = 3) -> Tuple[List[CaseRow], List[CaseRow]]:
    d20 = df[(df["d_frac"] == 0.20) & (df["has_gt"] == True) & (df["vox_removed_pc"] > 0)].copy()
    d20 = d20.sort_values("delta_pc")

    def pick_unique(rows: pd.DataFrame, n: int, ascending: bool) -> List[CaseRow]:
        out: List[CaseRow] = []
        used_structs = set()
        used_subj_crop = set()
        rows_it = rows.sort_values("delta_pc", ascending=ascending).itertuples(index=False)
        for r in rows_it:
            key_sc = (str(r.subject), str(r.crop))
            sname = str(r.structure)
            if sname in used_structs:
                continue
            if key_sc in used_subj_crop and len(out) < max(1, n - 1):
                continue
            out.append(
                CaseRow(
                    subject=str(r.subject),
                    crop=str(r.crop),
                    tag=str(r.tag),
                    structure=sname,
                    delta_pc=float(r.delta_pc),
                    precision_before=float(r.precision_before),
                    precision_pc=float(r.precision_pc),
                    vox_removed_pc=int(r.vox_removed_pc),
                )
            )
            used_structs.add(sname)
            used_subj_crop.add(key_sc)
            if len(out) >= n:
                break
        return out

    fails = pick_unique(d20[d20["delta_pc"] < 0], n_fail, ascending=True)
    goods = pick_unique(d20[d20["delta_pc"] > 0], n_good, ascending=False)
    return fails, goods


def load_predictions(pred_dir: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    affine = None
    for p in sorted(pred_dir.glob("*.nii.gz")):
        arr = np.asarray(nib.load(str(p)).dataobj).astype(bool)
        if arr.any():
            preds[p.stem.replace(".nii", "")] = arr
            if affine is None:
                affine = nib.load(str(p)).affine
    if not preds or affine is None:
        raise RuntimeError(f"No predictions found in {pred_dir}")
    return preds, affine


def load_gt_crop(
    subject: str, crop: str, pred_shape: Tuple[int, int, int], si_ax: int, si_sign: int
):
    gt_dir = DATA_DIR / subject / "segmentations"
    mri_full = nib.load(str(DATA_DIR / subject / "mri.nii.gz"))
    full_height = mri_full.shape[si_ax]
    anc_a, anc_b = CROP_ANCHORS[crop]
    crop_lo, crop_hi = compute_crop_window(gt_dir, (anc_a, anc_b), si_ax, full_height)

    gt_masks: Dict[str, np.ndarray] = {}
    for p in gt_dir.glob("*.nii.gz"):
        name = p.stem.replace(".nii", "")
        g = np.asarray(nib.load(str(p)).dataobj).astype(bool)
        g_crop = crop_gt(g, si_ax, si_sign, crop_lo, crop_hi)
        if g_crop.shape == pred_shape:
            gt_masks[name] = g_crop
    return gt_masks


def load_artifact_mri(case: CaseRow, pred_shape: Tuple[int, int, int]) -> np.ndarray:
    mri_path = EXP_DIR / case.subject / case.crop / case.tag / "mri_artifact.nii.gz"
    if not mri_path.exists():
        raise RuntimeError(f"Missing artifact MRI: {mri_path}")
    mri = np.asarray(nib.load(str(mri_path)).dataobj)
    if mri.shape != pred_shape:
        raise RuntimeError(
            f"Artifact MRI shape {mri.shape} != prediction shape {pred_shape} "
            f"for {case.subject}/{case.crop}/{case.tag}"
        )
    return mri


def _bincount_overlap(labeled: np.ndarray, anchor_mask: Optional[np.ndarray], n: int, lcc_label: Optional[int]) -> int:
    if anchor_mask is None:
        return lcc_label if lcc_label is not None else -1
    overlap = np.bincount(labeled[anchor_mask].ravel(), minlength=n + 1)
    overlap[0] = 0
    if overlap.max() > 0:
        return int(overlap.argmax())
    return lcc_label if lcc_label is not None else -1


def run_cm3_with_logs(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
    gt_masks: Dict[str, np.ndarray],
):
    cleaned = {n: m.copy() for n, m in predictions.items()}
    removed_tot = {n: 0 for n in predictions}
    events: List[dict] = []
    cc_cache: Dict[str, tuple] = {}

    N = next(iter(predictions.values())).shape[si_ax]
    center = N / 2.0

    def _lcc_midpoint(name: str) -> float:
        if name not in cleaned or not cleaned[name].any():
            return center
        if name not in cc_cache:
            cc_cache[name] = get_components(cleaned[name])
        labeled, _, _, lcc = cc_cache[name]
        if lcc is None:
            return center
        ext = axis_extent((labeled == lcc).astype(bool), si_ax)
        return (ext[0] + ext[1]) / 2.0 if ext is not None else center

    pairs = _get_pairs(poset, threshold)
    pairs_sorted = sorted(
        pairs, key=lambda p: abs((_lcc_midpoint(p[0]) + _lcc_midpoint(p[1])) / 2.0 - center)
    )

    def _get_anchor(name: str):
        if not cleaned[name].any():
            return None, None
        if name not in cc_cache:
            cc_cache[name] = get_components(cleaned[name])
        labeled, _, _, lcc = cc_cache[name]
        if lcc is None:
            return None, None
        anchor = (labeled == lcc).astype(bool)
        return anchor, axis_extent(anchor, si_ax)

    def _is_entirely_below(ext_a, ext_b) -> bool:
        if si_sign == +1:
            return ext_a[1] < ext_b[0]
        return ext_a[0] > ext_b[1]

    def remove_logged(
        *,
        pair_i: str,
        pair_j: str,
        step_kind: str,
        target_name: str,
        below_limit: Optional[int],
        above_limit: Optional[int],
        anchor_override: Optional[np.ndarray],
        protect_anchor: bool,
    ) -> None:
        mask = cleaned[target_name]
        if not mask.any():
            return

        if target_name not in cc_cache:
            cc_cache[target_name] = get_components(mask)
        labeled, n, sizes, lcc_label = cc_cache[target_name]
        if n == 0:
            return

        if not protect_anchor:
            anchor_label = -1
        else:
            anchor_label = _bincount_overlap(labeled, anchor_override, n, lcc_label)

        anchor_mask = (labeled == anchor_label) if anchor_label > 0 else np.zeros_like(mask, dtype=bool)
        anchor_extent = axis_extent(anchor_mask, si_ax) if anchor_label > 0 else None
        lcc_extent = axis_extent((labeled == lcc_label), si_ax) if lcc_label is not None else None

        removed_mask = np.zeros_like(mask, dtype=bool)
        component_rows = []

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
                violated = violated or ((c_max < below_limit) if si_sign == +1 else (c_min > below_limit))
            if above_limit is not None:
                violated = violated or ((c_min > above_limit) if si_sign == +1 else (c_max < above_limit))
            if violated:
                removed_mask |= comp
                component_rows.append(
                    {
                        "label": int(comp_label),
                        "voxels": int(comp.sum()),
                        "extent": ext,
                    }
                )

        if not removed_mask.any():
            return

        gt = gt_masks.get(target_name)
        tp_removed = int((removed_mask & gt).sum()) if gt is not None else 0
        fp_removed = int((removed_mask & ~gt).sum()) if gt is not None else int(removed_mask.sum())

        cleaned[target_name][removed_mask] = False
        removed_tot[target_name] += int(removed_mask.sum())
        cc_cache.pop(target_name, None)

        events.append(
            {
                "pair_i": pair_i,
                "pair_j": pair_j,
                "step_kind": step_kind,
                "target_name": target_name,
                "below_limit": below_limit,
                "above_limit": above_limit,
                "protect_anchor": protect_anchor,
                "anchor_label": int(anchor_label),
                "lcc_label": int(lcc_label) if lcc_label is not None else -1,
                "anchor_extent": anchor_extent,
                "lcc_extent": lcc_extent,
                "anchor_voxels": int(anchor_mask.sum()),
                "removed_voxels": int(removed_mask.sum()),
                "removed_tp": tp_removed,
                "removed_fp": fp_removed,
                "components": component_rows,
                "removed_mask": removed_mask.copy(),
                "anchor_mask": anchor_mask.copy(),
            }
        )

    for name_i, name_j in pairs_sorted:
        if name_i not in cleaned or name_j not in cleaned:
            continue

        anchor_i, ext_i = _get_anchor(name_i)
        anchor_j, ext_j = _get_anchor(name_j)

        if ext_i is not None and ext_j is not None and _is_entirely_below(ext_i, ext_j):
            remove_logged(
                pair_i=name_i,
                pair_j=name_j,
                step_kind="guard_remove_i_below_j",
                target_name=name_i,
                below_limit=ext_j[0],
                above_limit=None,
                anchor_override=None,
                protect_anchor=False,
            )
            anchor_i, ext_i = None, None
            anchor_j, ext_j = None, None

        if ext_j is not None:
            remove_logged(
                pair_i=name_i,
                pair_j=name_j,
                step_kind="normal_remove_i_below_j",
                target_name=name_i,
                below_limit=ext_j[0],
                above_limit=None,
                anchor_override=anchor_i,
                protect_anchor=True,
            )
        if ext_i is not None:
            remove_logged(
                pair_i=name_i,
                pair_j=name_j,
                step_kind="normal_remove_j_above_i",
                target_name=name_j,
                below_limit=None,
                above_limit=ext_i[1],
                anchor_override=anchor_j,
                protect_anchor=True,
            )

    return cleaned, removed_tot, events


def best_slice(mask: np.ndarray, si_ax: int) -> int:
    counts = np.array([mask.take(i, axis=si_ax).sum() for i in range(mask.shape[si_ax])], dtype=int)
    return int(counts.argmax()) if counts.max() > 0 else int(mask.shape[si_ax] // 2)


def best_slice_union(a: np.ndarray, b: np.ndarray, axis: int) -> int:
    u = a | b
    return best_slice(u, axis)


def get_ap_axis(affine: np.ndarray) -> int:
    codes = aff2axcodes(affine)
    for i, c in enumerate(codes):
        if c in ("A", "P"):
            return i
    raise RuntimeError(f"Could not determine AP axis from affine codes={codes}")


def get_lr_axis(affine: np.ndarray) -> int:
    codes = aff2axcodes(affine)
    for i, c in enumerate(codes):
        if c in ("L", "R"):
            return i
    raise RuntimeError(f"Could not determine LR axis from affine codes={codes}")


def extract_oriented_footprint(
    vol: np.ndarray,
    projection_axis: int,
    x_axis: int,
    y_axis: int,
) -> np.ndarray:
    """OR-project along ``projection_axis`` onto the (x_axis × y_axis) plane (coronal silhouette)."""
    m = np.asarray(vol).astype(bool)
    if m.ndim != 3:
        raise ValueError(f"footprint expects 3D volume, got shape {m.shape}")
    proj = np.any(m, axis=projection_axis)
    rem_axes = [ax for ax in range(m.ndim) if ax != projection_axis]
    x_pos = rem_axes.index(x_axis)
    y_pos = rem_axes.index(y_axis)
    if (y_pos, x_pos) == (0, 1):
        return proj
    if (y_pos, x_pos) == (1, 0):
        return proj.T
    raise RuntimeError(
        f"Unexpected axis mapping: projection_axis={projection_axis}, x_axis={x_axis}, "
        f"y_axis={y_axis}, rem_axes={rem_axes}"
    )


def extract_oriented_plane(
    vol: np.ndarray,
    slice_axis: int,
    slice_index: int,
    x_axis: int,
    y_axis: int,
) -> np.ndarray:
    """
    Extract 2D plane with explicit orientation:
      horizontal image axis = x_axis
      vertical image axis   = y_axis
    Returns array shaped (len(y_axis), len(x_axis)).
    """
    sl = vol.take(slice_index, axis=slice_axis)
    rem_axes = [ax for ax in range(vol.ndim) if ax != slice_axis]
    x_pos = rem_axes.index(x_axis)
    y_pos = rem_axes.index(y_axis)
    if (y_pos, x_pos) == (0, 1):
        return sl
    if (y_pos, x_pos) == (1, 0):
        return sl.T
    raise RuntimeError(
        f"Unexpected axis mapping: slice_axis={slice_axis}, x_axis={x_axis}, "
        f"y_axis={y_axis}, rem_axes={rem_axes}"
    )


def event_for_structure(events: List[dict], structure: str, prefer_tp: bool) -> Optional[dict]:
    sub = [e for e in events if e["target_name"] == structure and e["removed_voxels"] > 0]
    if not sub:
        return None
    if prefer_tp:
        sub.sort(key=lambda e: (e["removed_tp"], e["removed_voxels"]), reverse=True)
    else:
        sub.sort(key=lambda e: (e["removed_fp"] - e["removed_tp"], e["removed_voxels"]), reverse=True)
    return sub[0]


def rgb_overlay(
    mri2d: np.ndarray,
    gt2d: np.ndarray,
    before2d: np.ndarray,
    after2d: np.ndarray,
    event_removed2d: np.ndarray,
    anchor2d: np.ndarray,
    partner2d: np.ndarray,
):
    base = mri2d.astype(float)
    lo, hi = np.percentile(base, [1, 99])
    base = np.clip((base - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.dstack([base, base, base]) * 0.90
    rgb[gt2d, 2] = np.maximum(rgb[gt2d, 2], 0.65)          # GT blue
    rgb[after2d, 1] = np.maximum(rgb[after2d, 1], 0.95)    # after (kept) green
    rgb[event_removed2d, 0] = np.maximum(rgb[event_removed2d, 0], 0.98)  # removed red
    rgb[anchor2d, 0] = np.maximum(rgb[anchor2d, 0], 0.85)  # anchor magenta
    rgb[anchor2d, 2] = np.maximum(rgb[anchor2d, 2], 0.85)
    rgb[partner2d, 1] = np.maximum(rgb[partner2d, 1], 0.90)  # partner cyan
    rgb[partner2d, 2] = np.maximum(rgb[partner2d, 2], 0.90)
    return rgb


def pair_overlay(
    mri2d: np.ndarray,
    target_before: np.ndarray,
    target_after: np.ndarray,
    partner_before: np.ndarray,
    partner_after: np.ndarray,
) -> np.ndarray:
    base = mri2d.astype(float)
    lo, hi = np.percentile(base, [1, 99])
    base = np.clip((base - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.dstack([base, base, base]) * 0.9
    # Always show both structures:
    # target before/after = magenta/orange
    # partner before/after = cyan/yellow
    rgb[target_before, 0] = np.maximum(rgb[target_before, 0], 0.85)
    rgb[target_before, 2] = np.maximum(rgb[target_before, 2], 0.85)
    rgb[target_after, 0] = np.maximum(rgb[target_after, 0], 0.95)
    rgb[target_after, 1] = np.maximum(rgb[target_after, 1], 0.60)
    rgb[partner_before, 1] = np.maximum(rgb[partner_before, 1], 0.9)
    rgb[partner_before, 2] = np.maximum(rgb[partner_before, 2], 0.9)
    rgb[partner_after, 0] = np.maximum(rgb[partner_after, 0], 0.9)
    rgb[partner_after, 1] = np.maximum(rgb[partner_after, 1], 0.9)
    return rgb


def _set_pixel_true_aspect(ax, shape2d: Tuple[int, int]) -> None:
    h, w = shape2d
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(h / max(w, 1))


def imshow_native(ax, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    ax.imshow(img, origin="lower", interpolation="none", aspect="equal")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)


def visualize_case_row(ax_row, case: CaseRow, poset: PosetFromJson, prefer_tp_event: bool):
    pred_dir = EXP_DIR / case.subject / case.crop / case.tag / "segmentations"
    preds, affine = load_predictions(pred_dir)
    si_ax, si_sign = get_si_info(affine)
    ap_ax = get_ap_axis(affine)
    lr_ax = get_lr_axis(affine)
    pred_shape = next(iter(preds.values())).shape
    gt_masks = load_gt_crop(case.subject, case.crop, pred_shape, si_ax, si_sign)
    mri_art = load_artifact_mri(case, pred_shape)

    cleaned, removed_tot, events = run_cm3_with_logs(preds, poset, si_ax, si_sign, THRESHOLD, gt_masks)
    evt = event_for_structure(events, case.structure, prefer_tp=prefer_tp_event)
    if evt is None:
        raise RuntimeError(f"No removal event found for {case.structure} in {case.subject}/{case.crop}/{case.tag}")

    before = preds[case.structure]
    after = cleaned[case.structure]
    gt = gt_masks.get(case.structure, np.zeros_like(before, dtype=bool))

    partner_name = evt["pair_j"] if evt["target_name"] == evt["pair_i"] else evt["pair_i"]
    partner_mask = preds.get(partner_name, np.zeros_like(before, dtype=bool))

    # Choose coronal slice where both removed target voxels and partner are visible.
    sl_cor = best_slice_union(evt["removed_mask"], partner_mask, ap_ax)
    # Coronal plane: slice normal to AP; display x=LR, y=SI (shows SI axis clearly)
    mri2d_cor = extract_oriented_plane(mri_art, ap_ax, sl_cor, x_axis=lr_ax, y_axis=si_ax)
    before2d_cor = extract_oriented_footprint(before, ap_ax, lr_ax, si_ax)
    after2d_cor = extract_oriented_footprint(after, ap_ax, lr_ax, si_ax)
    gt2d_cor = extract_oriented_footprint(gt, ap_ax, lr_ax, si_ax)
    rem2d_cor = extract_oriented_footprint(evt["removed_mask"], ap_ax, lr_ax, si_ax)
    anc2d_cor = extract_oriented_footprint(evt["anchor_mask"], ap_ax, lr_ax, si_ax)
    partner_after_mask = cleaned.get(partner_name, np.zeros_like(before, dtype=bool))
    partner2d_cor = extract_oriented_footprint(partner_mask, ap_ax, lr_ax, si_ax)
    partner2d_cor_after = extract_oriented_footprint(partner_after_mask, ap_ax, lr_ax, si_ax)

    d0 = dice(before, gt)
    d1 = dice(after, gt)
    p0 = precision(before, gt)
    p1 = precision(after, gt)

    # Panel 1: coronal before
    imshow_native(
        ax_row[0],
        rgb_overlay(
            mri2d_cor,
            gt2d_cor,
            before2d_cor,
            before2d_cor,
            np.zeros_like(rem2d_cor),
            anc2d_cor,
            partner2d_cor,
        ),
    )
    ax_row[0].set_title(
        f"{case.structure} before coronal\n{case.subject} {case.crop} {case.tag}  MRI AP={sl_cor}; segs A/P silhouette",
        fontsize=9,
    )
    _set_pixel_true_aspect(ax_row[0], before2d_cor.shape)
    ax_row[0].axis("off")

    # Panel 2: coronal after
    imshow_native(
        ax_row[1],
        rgb_overlay(
            mri2d_cor,
            gt2d_cor,
            before2d_cor,
            after2d_cor,
            rem2d_cor,
            anc2d_cor,
            partner2d_cor,
        ),
    )
    ax_row[1].set_title("after CM3 (A/P silhouette on coronal LR×SI)", fontsize=9)
    _set_pixel_true_aspect(ax_row[1], after2d_cor.shape)
    ax_row[1].axis("off")

    # Panel 3: explicit two-structure segmentation view (target + responsible partner).
    imshow_native(
        ax_row[2],
        pair_overlay(mri2d_cor, before2d_cor, after2d_cor, partner2d_cor, partner2d_cor_after),
    )
    ax_row[2].set_title(
        f"two-structure segs (A/P silhouette)\n"
        f"target `{case.structure}` (magenta/orange), partner `{partner_name}` (cyan/yellow)",
        fontsize=9,
    )
    _set_pixel_true_aspect(ax_row[2], partner2d_cor.shape)
    ax_row[2].axis("off")

    # Panel 4: text explanation for the exact event
    ax_row[3].axis("off")
    txt = (
        f"Constraint pair: {evt['pair_i']} above {evt['pair_j']}\n"
        f"Step: {evt['step_kind']}\n"
        f"Target cleaned: {evt['target_name']}\n"
        f"Constraint partner shown: {partner_name}\n"
        f"Anchor label chosen: {evt['anchor_label']} (LCC={evt['lcc_label']})\n"
        f"Anchor voxels: {evt['anchor_voxels']:,}\n"
        f"Limits: below<{evt['below_limit']}  above>{evt['above_limit']}\n"
        f"Removed voxels (event): {evt['removed_voxels']:,}\n"
        f"  TP removed: {evt['removed_tp']:,}\n"
        f"  FP removed: {evt['removed_fp']:,}\n\n"
        f"Dice: {d0:.4f} -> {d1:.4f}  (delta {d1-d0:+.4f})\n"
        f"Precision: {p0:.4f} -> {p1:.4f}  (delta {p1-p0:+.4f})\n"
        f"CSV delta_pc: {case.delta_pc:+.5f}, vox_removed_pc={case.vox_removed_pc:,}"
    )
    ax_row[3].text(0.0, 1.0, txt, va="top", ha="left", fontsize=9, family="monospace")

    return {
        "case": case,
        "event": evt,
        "dice_before": d0,
        "dice_after": d1,
        "precision_before": p0,
        "precision_after": p1,
        "removed_total_structure": removed_tot.get(case.structure, 0),
        "coronal_shape": before2d_cor.shape,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    partials = sorted(EVAL_T100_DIR.glob("partial_*/results.csv"))
    if not partials:
        raise RuntimeError(f"No partial CSVs under {EVAL_T100_DIR}")
    df = pd.concat([pd.read_csv(p) for p in partials], ignore_index=True)

    fail_case, good_case = choose_cases(df)
    poset = load_poset_from_json(str(POSET_PATH))

    # Dry-run once to determine panel geometry from true pixel sizes.
    fig_tmp, axes_tmp = plt.subplots(2, 4, figsize=(10, 6))
    summary_fail = visualize_case_row(axes_tmp[0], fail_case, poset, prefer_tp_event=True)
    summary_good = visualize_case_row(axes_tmp[1], good_case, poset, prefer_tp_event=False)
    plt.close(fig_tmp)

    cor_h = max(summary_fail["coronal_shape"][0], summary_good["coronal_shape"][0])
    cor_w = max(summary_fail["coronal_shape"][1], summary_good["coronal_shape"][1])

    # Coronal-only layout + text column.
    width_ratios = [cor_w, cor_w, cor_w, int(1.3 * cor_w)]
    row_h = cor_h
    scale = 0.035
    fig_w = scale * sum(width_ratios)
    fig_h = scale * (2.2 * row_h)

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=True,
    )
    summary_fail = visualize_case_row(axes[0], fail_case, poset, prefer_tp_event=True)
    summary_good = visualize_case_row(axes[1], good_case, poset, prefer_tp_event=False)

    fig.suptitle(
        "Real CM3 cases from partial CSVs: failure (top) vs success (bottom)\n"
        "MRI = one coronal slice · segmentations = A/P projection (LR×SI footprint); "
        "blue=GT, green=kept, red=removed, magenta=anchor, cyan=partner-before, yellow=partner-after",
        fontsize=12,
    )
    out_fig = OUT_DIR / "cm3_real_good_bad_examples.png"
    fig.savefig(out_fig, dpi=180)
    plt.close(fig)

    out_md = OUT_DIR / "cm3_real_case_details.md"
    out_md.write_text(
        "\n".join(
            [
                "# CM3 Real Case Details",
                "",
                "## Failure case",
                f"- subject/crop/tag: `{fail_case.subject}` / `{fail_case.crop}` / `{fail_case.tag}`",
                f"- structure: `{fail_case.structure}`",
                f"- selected event: `{summary_fail['event']['step_kind']}` on pair "
                f"`{summary_fail['event']['pair_i']} above {summary_fail['event']['pair_j']}`",
                f"- TP removed: `{summary_fail['event']['removed_tp']}`",
                f"- FP removed: `{summary_fail['event']['removed_fp']}`",
                "",
                "## Good case",
                f"- subject/crop/tag: `{good_case.subject}` / `{good_case.crop}` / `{good_case.tag}`",
                f"- structure: `{good_case.structure}`",
                f"- selected event: `{summary_good['event']['step_kind']}` on pair "
                f"`{summary_good['event']['pair_i']} above {summary_good['event']['pair_j']}`",
                f"- TP removed: `{summary_good['event']['removed_tp']}`",
                f"- FP removed: `{summary_good['event']['removed_fp']}`",
                "",
            ]
        )
    )

    print(f"Saved: {out_fig}")
    print(f"Saved: {out_md}")

    # Additional figure: d=0.20 cases (one failure, one success).
    fail20, good20 = choose_cases_d20(df)
    fig20, axes20 = plt.subplots(
        2,
        4,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=True,
    )
    _ = visualize_case_row(axes20[0], fail20, poset, prefer_tp_event=True)
    _ = visualize_case_row(axes20[1], good20, poset, prefer_tp_event=False)
    fig20.suptitle(
        "CM3 real cases at d=0.20: failure (top) and success (bottom)\n"
        "artifact MRI background; blue=GT, green=kept, red=removed, magenta=anchor, "
        "cyan=partner-before, yellow=partner-after",
        fontsize=12,
    )
    out_fig20 = OUT_DIR / "cm3_real_good_bad_examples_d020.png"
    fig20.savefig(out_fig20, dpi=180)
    plt.close(fig20)
    print(f"Saved: {out_fig20}")

    # More d=0.20 outcomes: multiple failures and successes.
    fails20, goods20 = choose_more_cases_d20(df, n_fail=2, n_good=3)
    cases20 = fails20 + goods20
    if cases20:
        nrows = len(cases20)
        fig_multi, axes_multi = plt.subplots(
            nrows,
        4,
            figsize=(fig_w, max(fig_h, 0.5 * nrows * fig_h)),
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=True,
        )
        if nrows == 1:
            axes_multi = np.array([axes_multi])

        details_lines = ["# d=0.20 Additional Cases", ""]
        for i, c in enumerate(cases20):
            pref_tp = i < len(fails20)
            summary = visualize_case_row(axes_multi[i], c, poset, prefer_tp_event=pref_tp)
            label = "failure" if i < len(fails20) else "success"
            details_lines.append(
                f"- {label}: `{c.subject}` / `{c.crop}` / `{c.tag}` / `{c.structure}` "
                f"(delta_pc={c.delta_pc:+.5f}) "
                f"pair=`{summary['event']['pair_i']} above {summary['event']['pair_j']}`"
            )

        fig_multi.suptitle(
            "CM3 additional real d=0.20 outcomes\n"
            "top rows: failures, bottom rows: successes; artifact MRI background",
            fontsize=12,
        )
        out_fig_multi = OUT_DIR / "cm3_real_d020_more_outcomes.png"
        fig_multi.savefig(out_fig_multi, dpi=180)
        plt.close(fig_multi)
        print(f"Saved: {out_fig_multi}")

        out_md_multi = OUT_DIR / "cm3_real_d020_more_outcomes.md"
        out_md_multi.write_text("\n".join(details_lines))
        print(f"Saved: {out_md_multi}")


if __name__ == "__main__":
    main()
