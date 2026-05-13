"""
Visualize real CM4 success/failure cases from wraparound_v4_eval_cm4 partial CSVs.

Final segmentations always come from ``evaluate_cleaning_methods.method4_center_conflict``
(the same routine as batch eval). ``run_cm4_with_logs`` is only for per-step captions;
**blue / red / green overlays use the full removal** ``pred & ~cleaned`` for the target
structure (all poset pairs), not a single logged sub-step — otherwise the figure could
disagree with CSV ΔDice / ``vox_removed_pc`` when multiple steps touch the same organ.

Case selection (merged CSV):
  - 3 **good** rows: d = 0.20, highest ΔDice among rows with cleaning applied
  - 3 **bad** rows: d = 0.50, lowest ΔDice
  - 1 **extra bad** row: lowest d (≤ 0.15) with negative ΔDice, if available

Coronal panels (**square pixels**): segmentation overlays use an **A/P silhouette**
(logical OR along the anterior–posterior axis) projected onto the coronal LR×SI plane, so
the full footprint of each 3D mask is visible even when the MRI slice only clips part of
the organ. **Only the MRI grayscale** stays a single coronal slice. **before**: blue
(removals) + purple (partner); **after**: red (removed TP) + green (removed FP).
Third “two-structure only” panel removed.

Legend colors:
  - Blue — target segmentation voxels removed by CM4 (scheduled removal)
  - Purple — partner structure whose constraint triggers cleaning
  - Red — removed true-positive voxels (overlap with GT)
  - Green — removed false-positive voxels

Outputs:
  results/cm4_real_cases/cm4_real_good_bad_examples.png
  results/cm4_real_cases/cm4_real_case_details.md
  results/cm4_real_cases/cm4_real_good_bad_examples_d010_020.png
  results/cm4_real_cases/cm4_real_case_details_d010_020.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import aff2axcodes

import sys
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "cleaning"))

from anatomy_poset.core.io import load_poset_from_json, PosetFromJson
from evaluate_cleaning_methods import (
    get_si_info,
    get_components,
    axis_extent,
    _get_pairs,
    _min_abs_si_distance_to_midplane,
    method4_center_conflict,
    compute_crop_window,
    crop_gt,
    dice,
    precision,
)

DATA_DIR = PROJECT_ROOT / "data" / "datasets" / "TotalsegmentatorMRI_dataset_v200"
EXP_DIR = PROJECT_ROOT / "data" / "experiments" / "wraparound_v4"
EVAL_CM4_T100 = PROJECT_ROOT / "data" / "experiments" / "wraparound_v4_eval_cm4" / "t100"
POSET_PATH = PROJECT_ROOT / "data" / "posets" / "empirical" / "totalseg_mri_empirical_poset.json"
OUT_DIR = PROJECT_ROOT / "results" / "cm4_real_cases"
THRESHOLD = 1.00


def strip_cm4_from_figure_text(s: str) -> str:
    """Strip ``CM4`` / ``cm4`` tokens from strings drawn on publication figures."""
    if not s:
        return s
    t = re.sub(r"(?i)cm4[_-]?", "", s)
    t = re.sub(r"(?i)\bcm4\b", "", t)
    return re.sub(r"\s+", " ", t).strip(" \t-_")


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


def load_merged_or_partials(eval_dir: Path) -> pd.DataFrame:
    merged = eval_dir / "results.csv"
    if merged.exists():
        return pd.read_csv(merged)
    partials = sorted(eval_dir.glob("partial_*/results.csv"))
    if not partials:
        raise RuntimeError(f"No results.csv or partial_*/results.csv under {eval_dir}")
    return pd.concat([pd.read_csv(p) for p in partials], ignore_index=True)


def _has_gt_series(df: pd.DataFrame) -> pd.Series:
    hg = df["has_gt"]
    if hg.dtype == bool:
        return hg
    return hg.astype(str).str.lower().isin(("true", "1", "yes"))


def _to_case(r) -> CaseRow:
    return CaseRow(
        subject=str(r.subject),
        crop=str(r.crop),
        tag=str(r.tag),
        structure=str(r.structure),
        delta_pc=float(r.delta_pc),
        precision_before=float(r.precision_before),
        precision_pc=float(r.precision_pc),
        vox_removed_pc=int(r.vox_removed_pc),
    )


def _pick_sorted(
    rows: pd.DataFrame, n: int, ascending: bool, require_positive_delta: Optional[bool] = None
) -> List[CaseRow]:
    if require_positive_delta is True:
        rows = rows[rows["delta_pc"] > 0]
    elif require_positive_delta is False:
        rows = rows[rows["delta_pc"] < 0]
    out: List[CaseRow] = []
    used_struct = set()
    used_key = set()
    for r in rows.sort_values("delta_pc", ascending=ascending).itertuples(index=False):
        sname = str(r.structure)
        key = (str(r.subject), str(r.crop), str(r.tag))
        if sname in used_struct:
            continue
        if key in used_key and len(out) < max(1, n - 1):
            continue
        out.append(_to_case(r))
        used_struct.add(sname)
        used_key.add(key)
        if len(out) >= n:
            break
    return out


def pick_cases_stratified(
    df: pd.DataFrame,
    n_good_d20: int = 3,
    n_bad_d50: int = 3,
    include_low_d_bad: bool = True,
    low_d_max: float = 0.15,
) -> List[Tuple[CaseRow, bool, str]]:
    """
    Good: d≈0.20, ΔDice>0. Bad d≈0.50: ΔDice<0. Extra bad: d≤low_d_max, ΔDice<0 (if found).
    """
    base = df[_has_gt_series(df) & (df["vox_removed_pc"] > 0)].copy()

    def near(val: float) -> pd.Series:
        return (base["d_frac"] - val).abs() < 1e-5

    good_pool = base[near(0.20) & (base["delta_pc"] > 0)]
    bad50_pool = base[near(0.50) & (base["delta_pc"] < 0)]

    goods = _pick_sorted(good_pool, n_good_d20, ascending=False)
    bads50 = _pick_sorted(bad50_pool, n_bad_d50, ascending=True)

    if len(goods) < n_good_d20:
        raise RuntimeError(f"Need {n_good_d20} good cases at d=0.20; found {len(goods)}")
    if len(bads50) < n_bad_d50:
        raise RuntimeError(f"Need {n_bad_d50} bad cases at d=0.50; found {len(bads50)}")

    extra: Optional[CaseRow] = None
    extra_label = ""
    if include_low_d_bad:
        low = base[(base["d_frac"] <= low_d_max + 1e-6) & (base["delta_pc"] < 0)].copy()
        picked = _pick_sorted(low, 1, ascending=True)
        if picked:
            extra_label = f"bad (d≤{low_d_max:g})"
        else:
            neg = base[base["delta_pc"] < 0].copy()
            if len(neg):
                d_min_fail = float(neg["d_frac"].min())
                band = neg[(neg["d_frac"] - d_min_fail).abs() < 1e-5]
                picked = _pick_sorted(band, 1, ascending=True)
                if picked:
                    extra_label = f"bad (min d with ΔDice<0, d={d_min_fail:.2f})"
        if picked:
            cand = picked[0]
            chosen_keys = {(str(c.subject), str(c.crop), str(c.tag), str(c.structure)) for c in goods + bads50}
            key = (cand.subject, cand.crop, cand.tag, cand.structure)
            if key not in chosen_keys:
                extra = cand

    order: List[Tuple[CaseRow, bool, str]] = []
    for c in goods:
        order.append((c, False, "good d=0.20"))
    for c in bads50:
        order.append((c, True, "bad d=0.50"))
    if extra is not None and extra_label:
        order.append((extra, True, extra_label))

    return order


def pick_cases_d_band(
    df: pd.DataFrame,
    d_lo: float = 0.10,
    d_hi: float = 0.20,
    n_good: int = 3,
    n_bad: int = 3,
) -> List[Tuple[CaseRow, bool, str]]:
    """Good / bad rows with wraparound strength ``d_frac`` in ``[d_lo, d_hi]`` (inclusive).

    If fewer than ``n_bad`` negative-ΔDice rows exist in the band, all available bad rows
    are kept (at least one is required).
    """
    base = df[_has_gt_series(df) & (df["vox_removed_pc"] > 0)].copy()
    band = base[(base["d_frac"] >= d_lo - 1e-9) & (base["d_frac"] <= d_hi + 1e-9)]
    good_pool = band[band["delta_pc"] > 0]
    bad_pool = band[band["delta_pc"] < 0]

    goods = _pick_sorted(good_pool, n_good, ascending=False)
    bads = _pick_sorted(bad_pool, n_bad, ascending=True)

    if len(goods) < n_good:
        raise RuntimeError(
            f"Need {n_good} good cases with {d_lo:g}≤d≤{d_hi:g}; found {len(goods)} in band (rows={len(band)})"
        )
    if len(bads) < 1:
        raise RuntimeError(
            f"Need at least 1 bad case with {d_lo:g}≤d≤{d_hi:g}; found 0 in band (rows={len(band)})"
        )
    if len(bads) < n_bad:
        print(
            f"WARNING: only {len(bads)} bad case(s) with {d_lo:g}≤d≤{d_hi:g} (wanted {n_bad}); "
            f"figure will use {len(bads)}.",
            flush=True,
        )

    order: List[Tuple[CaseRow, bool, str]] = []
    for c in goods:
        order.append((c, False, f"good {d_lo:g}≤d≤{d_hi:g}"))
    for c in bads:
        order.append((c, True, f"bad {d_lo:g}≤d≤{d_hi:g}"))
    return order


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


def run_cm4_with_logs(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
    gt_masks: Dict[str, np.ndarray],
    *,
    collect_snapshots: bool = False,
):
    """CM4 with per-event logging (matches evaluate_cleaning_methods.method4_center_conflict).

    If collect_snapshots is True, also returns full-mask snapshots after each removal event
    (snapshots[0] = state before any pair processing — actually before loop we append initial;
    after each successful removal, append updated cleaned).
    """
    cleaned = {n: m.copy() for n, m in predictions.items()}
    snapshots: Optional[List[Dict[str, np.ndarray]]] = [] if collect_snapshots else None
    if collect_snapshots:
        snapshots.append({k: v.copy() for k, v in cleaned.items()})
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
                component_rows.append({"label": int(comp_label), "voxels": int(comp.sum()), "extent": ext})

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
        if collect_snapshots:
            assert snapshots is not None
            snapshots.append({k: v.copy() for k, v in cleaned.items()})

    for name_i, name_j in pairs_sorted:
        if name_i not in cleaned or name_j not in cleaned:
            continue

        anchor_i, ext_i = _get_anchor(name_i)
        anchor_j, ext_j = _get_anchor(name_j)

        if (
            ext_i is not None
            and ext_j is not None
            and anchor_i is not None
            and anchor_j is not None
            and _is_entirely_below(ext_i, ext_j)
        ):
            dist_i = _min_abs_si_distance_to_midplane(anchor_i, si_ax, center)
            dist_j = _min_abs_si_distance_to_midplane(anchor_j, si_ax, center)

            if dist_i <= dist_j:
                remove_logged(
                    pair_i=name_i,
                    pair_j=name_j,
                    step_kind="cm4_conflict_prefer_i_remove_j",
                    target_name=name_j,
                    below_limit=None,
                    above_limit=ext_i[1],
                    anchor_override=None,
                    protect_anchor=False,
                )
            else:
                remove_logged(
                    pair_i=name_i,
                    pair_j=name_j,
                    step_kind="cm4_conflict_prefer_j_remove_i",
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

    return cleaned, removed_tot, events, snapshots


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
    """Collapse ``projection_axis`` with logical OR — silhouette on the (x_axis × y_axis) plane.

    For a coronal MRI slice (normal ≈ A/P), pass ``projection_axis`` = AP so segmentations
    show full LR×SI extent even when the chosen slice only clips part of the volume.
    Same image layout as ``extract_oriented_plane``: shape (len(y), len(x)).
    """
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


def best_event_for_structure(
    events: List[dict], structure: str, full_removed: np.ndarray, prefer_tp_event: bool
) -> Optional[dict]:
    """Pick a logged event that best overlaps the *actual* total removal (multi-step safe)."""
    sub = [e for e in events if e["target_name"] == structure and e["removed_voxels"] > 0]
    if not sub:
        return None

    def overlap(e):
        return int(np.count_nonzero(e["removed_mask"] & full_removed))

    sub.sort(key=lambda e: (overlap(e), e["removed_tp"] if prefer_tp_event else e["removed_fp"]), reverse=True)
    return sub[0]


def _normalize_mri(mri2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(mri2d.astype(float), [1, 99])
    return np.clip((mri2d.astype(float) - lo) / max(hi - lo, 1e-6), 0, 1)


def overlay_before_cm4(
    mri2d: np.ndarray,
    target_removed_blue_2d: np.ndarray,
    partner_purple_2d: np.ndarray,
) -> np.ndarray:
    """Blue = target voxels scheduled for removal; purple = constraint-partner segmentation."""
    g = _normalize_mri(mri2d)
    rgb = np.stack([g, g, g], axis=-1)
    pp = partner_purple_2d.astype(bool)
    rb = target_removed_blue_2d.astype(bool)
    rgb[pp, 0] = np.maximum(rgb[pp, 0], 0.72)
    rgb[pp, 1] = np.minimum(rgb[pp, 1], 0.42)
    rgb[pp, 2] = np.maximum(rgb[pp, 2], 0.88)
    rgb[rb, 0] = np.minimum(rgb[rb, 0], 0.22)
    rgb[rb, 1] = np.maximum(rgb[rb, 1], 0.52)
    rgb[rb, 2] = np.maximum(rgb[rb, 2], 0.98)
    return np.clip(rgb, 0, 1)


def overlay_after_tp_fp(mri2d: np.ndarray, tp_red_2d: np.ndarray, fp_green_2d: np.ndarray) -> np.ndarray:
    """Red = removed TP; green = removed FP. Green drawn first so TP wins if silhouettes overlap."""
    g = _normalize_mri(mri2d)
    rgb = np.stack([g, g, g], axis=-1)
    tr = tp_red_2d.astype(bool)
    fg = fp_green_2d.astype(bool)
    rgb[fg, 0] *= 0.45
    rgb[fg, 1] = np.maximum(rgb[fg, 1], 0.94)
    rgb[fg, 2] *= 0.45
    rgb[tr, 0] = np.maximum(rgb[tr, 0], 0.96)
    rgb[tr, 1] *= 0.35
    rgb[tr, 2] *= 0.35
    return np.clip(rgb, 0, 1)


def _set_pixel_true_aspect(ax, shape2d: Tuple[int, int]) -> None:
    h, w = shape2d
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(h / max(w, 1))


def imshow_native(ax, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    ax.imshow(img, origin="lower", interpolation="none", aspect="equal")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)


def visualize_case_row(ax_row, case: CaseRow, poset: PosetFromJson, prefer_tp_event: bool, label: str):
    """ax_row: length 3 — before | after (TP/FP removal) | text."""
    pred_dir = EXP_DIR / case.subject / case.crop / case.tag / "segmentations"
    preds, affine = load_predictions(pred_dir)
    si_ax, si_sign = get_si_info(affine)
    ap_ax = get_ap_axis(affine)
    lr_ax = get_lr_axis(affine)
    pred_shape = next(iter(preds.values())).shape
    gt_masks = load_gt_crop(case.subject, case.crop, pred_shape, si_ax, si_sign)
    mri_art = load_artifact_mri(case, pred_shape)

    # Source of truth for final masks (must match evaluate_cleaning_methods CM4 eval).
    cleaned, _removed_eval = method4_center_conflict(preds, poset, si_ax, si_sign, THRESHOLD)
    cleaned_log, removed_tot_log, events, _snap = run_cm4_with_logs(
        preds, poset, si_ax, si_sign, THRESHOLD, gt_masks
    )
    for name in preds:
        if not np.array_equal(cleaned[name], cleaned_log[name]):
            raise RuntimeError(
                f"CM4 implementation drift: cleaned masks differ for {name!r} in "
                f"{case.subject}/{case.crop}/{case.tag}. Re-sync run_cm4_with_logs with method4_center_conflict."
            )

    before = preds[case.structure]
    after = cleaned[case.structure]
    gt = gt_masks.get(case.structure, np.zeros_like(before, dtype=bool))
    full_removed = before & ~after
    if int(full_removed.sum()) != int(removed_tot_log.get(case.structure, 0)):
        # Logging aggregates per event; allow small mismatch only if structure untouched
        if full_removed.any():
            raise RuntimeError(
                f"Removal count mismatch for {case.structure}: "
                f"diff={full_removed.sum()} vs logged={removed_tot_log.get(case.structure, 0)}"
            )

    evt = best_event_for_structure(events, case.structure, full_removed, prefer_tp_event)
    if evt is None and full_removed.any():
        raise RuntimeError(f"No removal event found for {case.structure} in {case.subject}/{case.crop}/{case.tag}")
    if evt is None:
        raise RuntimeError(f"No CM4 removal for {case.structure} in {case.subject}/{case.crop}/{case.tag}")

    partner_name = evt["pair_j"] if evt["target_name"] == evt["pair_i"] else evt["pair_i"]
    partner_mask = preds.get(partner_name, np.zeros_like(before, dtype=bool))

    sl_cor = best_slice_union(full_removed, partner_mask, ap_ax)
    mri2d_cor = extract_oriented_plane(mri_art, ap_ax, sl_cor, x_axis=lr_ax, y_axis=si_ax)
    rem2d_cor = extract_oriented_footprint(full_removed, ap_ax, lr_ax, si_ax)
    partner2d_cor = extract_oriented_footprint(partner_mask, ap_ax, lr_ax, si_ax)

    tp_removed_2d = extract_oriented_footprint(full_removed & gt, ap_ax, lr_ax, si_ax)
    fp_removed_2d = extract_oriented_footprint(full_removed & (~gt), ap_ax, lr_ax, si_ax)

    d0 = dice(before, gt)
    d1 = dice(after, gt)
    p0 = precision(before, gt)
    p1 = precision(after, gt)

    imshow_native(ax_row[0], overlay_before_cm4(mri2d_cor, rem2d_cor, partner2d_cor))
    ax_row[0].set_title(
        f"{label}: before cleaning (coronal)\n{case.structure} | {case.subject} {case.crop} {case.tag}  "
        f"MRI AP={sl_cor} · masks = A/P silhouette",
        fontsize=10,
    )
    _set_pixel_true_aspect(ax_row[0], rem2d_cor.shape)
    ax_row[0].axis("off")

    imshow_native(ax_row[1], overlay_after_tp_fp(mri2d_cor, tp_removed_2d, fp_removed_2d))
    ax_row[1].set_title(
        "after cleaning — removals\nred/green = TP/FP (A/P silhouette on this slice)",
        fontsize=10,
    )
    _set_pixel_true_aspect(ax_row[1], rem2d_cor.shape)
    ax_row[1].axis("off")

    ax_row[2].axis("off")
    sk = strip_cm4_from_figure_text(str(evt["step_kind"]))
    txt = (
        f"{label.upper()}  CSV ΔDice={case.delta_pc:+.5f}\n"
        f"Pair: {evt['pair_i']} above {evt['pair_j']}\n"
        f"Step: {sk}\n"
        f"Target (blue removal): {evt['target_name']}\n"
        f"Partner (purple): {partner_name}\n"
        f"Full removal (all steps): {int(full_removed.sum()):,} vox  "
        f"(TP {int((full_removed & gt).sum()):,}, FP {int((full_removed & ~gt).sum()):,})\n"
        f"Caption event ({sk}): {evt['removed_voxels']:,} vox  "
        f"(TP {evt['removed_tp']:,}, FP {evt['removed_fp']:,})\n\n"
        f"Dice: {d0:.4f} → {d1:.4f}  (Δ {d1 - d0:+.4f})\n"
        f"Precision: {p0:.4f} → {p1:.4f}  (Δ {p1 - p0:+.4f})\n"
        f"Total removed (structure): {int(full_removed.sum()):,}  "
        f"(logged primary event: {evt['removed_voxels']:,} voxels)\n"
        f"Note: blue/red/green use **full** removal (all steps); caption step is the best-matching log event."
    )
    ax_row[2].text(
        0.02,
        1.0,
        txt,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        linespacing=1.35,
    )

    return {"coronal_shape": rem2d_cor.shape, "event": evt}


def render_good_bad_grid(
    cases_order: List[Tuple[CaseRow, bool, str]],
    poset: PosetFromJson,
    *,
    out_png: Path,
    out_md: Path,
    suptitle: str,
    md_header: str,
) -> None:
    """Build the 3-column × N-row figure and companion markdown."""
    # Geometry dry-run
    fig_tmp, axes_tmp = plt.subplots(len(cases_order), 3, figsize=(12.5, 3.1 * len(cases_order)))
    if len(cases_order) == 1:
        axes_tmp = np.array([axes_tmp])
    shapes = []
    for i, (case, pref_tp, lab) in enumerate(cases_order):
        s = visualize_case_row(axes_tmp[i], case, poset, prefer_tp_event=pref_tp, label=lab)
        shapes.append(s["coronal_shape"])
    plt.close(fig_tmp)

    cor_h = max(s[0] for s in shapes)
    cor_w = max(s[1] for s in shapes)
    width_ratios = [cor_w, cor_w, int(2.6 * cor_w)]
    scale = 0.021
    fig_w = scale * sum(width_ratios)
    fig_h = scale * (2.05 * cor_h) * len(cases_order)

    fig, axes = plt.subplots(
        len(cases_order),
        3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if len(cases_order) == 1:
        axes = np.array([axes])

    md_lines = [md_header, ""]
    for i, (case, pref_tp, lab) in enumerate(cases_order):
        summ = visualize_case_row(axes[i], case, poset, prefer_tp_event=pref_tp, label=lab)
        md_lines.append(
            f"- **{lab}**: `{case.subject}` / `{case.crop}` / `{case.tag}` / `{case.structure}` "
            f"ΔDice={case.delta_pc:+.5f} — `{strip_cm4_from_figure_text(str(summ['event']['step_kind']))}`"
        )

    legend_handles = [
        Patch(
            facecolor=(0.25, 0.55, 0.98),
            edgecolor="navy",
            linewidth=0.7,
            label="Blue: target removals (A/P silhouette on coronal LR×SI plane)",
        ),
        Patch(
            facecolor=(0.72, 0.35, 0.88),
            edgecolor="purple",
            linewidth=0.7,
            label="Purple: partner (A/P silhouette)",
        ),
        Patch(
            facecolor=(0.96, 0.22, 0.22),
            edgecolor="darkred",
            linewidth=0.5,
            label="Red: removed TP (A/P silhouette)",
        ),
        Patch(
            facecolor=(0.22, 0.92, 0.35),
            edgecolor="darkgreen",
            linewidth=0.5,
            label="Green: removed FP (A/P silhouette)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_md}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_merged_or_partials(EVAL_CM4_T100)
    poset = load_poset_from_json(str(POSET_PATH))

    strat = pick_cases_stratified(df)
    render_good_bad_grid(
        strat,
        poset,
        out_png=OUT_DIR / "cm4_real_good_bad_examples.png",
        out_md=OUT_DIR / "cm4_real_case_details.md",
        suptitle=(
            "Cases — good at d=0.20, bad at d=0.50 (+ optional bad at low d)\n"
            "MRI = one coronal slice · colored masks = A/P projection (full LR×SI footprint)"
        ),
        md_header="# Real cleaning cases (coronal, square pixels)",
    )

    band = pick_cases_d_band(df, d_lo=0.10, d_hi=0.20, n_good=3, n_bad=3)
    render_good_bad_grid(
        band,
        poset,
        out_png=OUT_DIR / "cm4_real_good_bad_examples_d010_020.png",
        out_md=OUT_DIR / "cm4_real_case_details_d010_020.md",
        suptitle=(
            "Cases — good and bad with 0.10 ≤ d ≤ 0.20\n"
            "MRI = one coronal slice · colored masks = A/P projection (full LR×SI footprint)"
        ),
        md_header="# Real cleaning cases — d in [0.10, 0.20] (coronal, square pixels)",
    )


if __name__ == "__main__":
    main()
