"""
Visualize real CM4 success/failure cases from wraparound_v4_eval_cm4 partial CSVs.

Final segmentations always come from ``evaluate_cleaning_methods.method4_center_conflict``
(the same routine as batch eval). ``run_cm4_with_logs`` is only for per-step captions;
**blue / red / green overlays use the full removal** ``pred & ~cleaned`` for the target
structure (all poset pairs), not a single logged sub-step — otherwise the figure could
disagree with CSV metrics / ``vox_removed_pc`` when multiple steps touch the same organ.
Figure panels report **ΔF1** from the full 3D masks shown (before vs after cleaning).

Case selection (merged CSV, **one subject** — default ``FOCUS_SUBJECT``; override with ``--subject``):
  - **Good** rows: ``d_frac ≤ 0.20``, ``ΔF1 > 0`` when column ``delta_f1`` exists (else ``ΔDice > 0``),
    cleaning applied; sorted by **largest** ``vox_removed_pc`` first so larger removed components
    show clearly (correct TP removal).
  - **Bad** rows: ``d_frac ≤ 0.45``, ``ΔF1 < 0`` (else ``ΔDice < 0``); same large-voxel bias
    (large FP / harmful removals visible).

Coronal panels: overlays use an **A/P silhouette**
(logical OR along the anterior–posterior axis) projected onto the coronal LR×SI plane, so
the full footprint of each 3D mask is visible even when the MRI slice only clips part of
the organ. **Only the MRI grayscale** stays a single coronal slice. For subject ``s0175``,
the SI axis (image rows) is drawn **2× taller** on screen than LR for readability; other
subjects keep square data pixels. **before**: blue
(targeted structure removals) + purple (trusted anchor); **after**: red (removed TP) + green (removed FP).
MRI axes show short titles (crop + subject, before/after); the companion ``.md`` lists full case details.
Third “two-structure only” panel removed.

Legend colors:
  - Blue — targeted structure: voxels CM4 removes from the organ (scheduled removal)
  - Purple — trusted anchor: partner structure whose constraint drives the step
  - Red — removed true-positive voxels (overlap with GT)
  - Green — removed false-positive voxels

Performance:
  Loading all segmentations + GT for one volume and running **CM4 twice** (``method4_center_conflict`` +
  ``run_cm4_with_logs``) dominates runtime. The script **caches** that work per ``(subject, crop, tag)``
  and **precomputes** each selected row once, then draws **good-only** and **bad-only** figures from the
  same preps (no duplicate dry-run pass). Disk caching of bundles is not enabled (masks are large); re-run
  is still needed if data change.

Outputs:
  results/cm4_real_cases/cm4_real_case_details.md  (companion notes for all selected rows)
  results/cm4_real_cases/cm4_real_good_examples.png
  results/cm4_real_cases/cm4_real_bad_examples.png

  Other subjects get ``_*`` suffixes, e.g. ``cm4_real_good_examples_s0167.png``.

Usage:
  python scripts/cleaning/visualize_cm4_real_modes.py --subject s0167
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec as mgs
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
    f1,
)

DATA_DIR = PROJECT_ROOT / "data" / "datasets" / "TotalsegmentatorMRI_dataset_v200"
EXP_DIR = PROJECT_ROOT / "data" / "experiments" / "wraparound_v4"
EVAL_CM4_T100 = PROJECT_ROOT / "data" / "experiments" / "wraparound_v4_eval_cm4" / "t100"
POSET_PATH = PROJECT_ROOT / "data" / "posets" / "empirical" / "totalseg_mri_empirical_poset.json"
OUT_DIR = PROJECT_ROOT / "results" / "cm4_real_cases"
THRESHOLD = 1.00

# ``render_case_tile_grid`` / ``visualize_case_row`` layout & typography.
# Heavy CM4 work is cached per (subject, crop, tag); each figure reuses ``CaseRowPrep``.
FIG_TEXT_FONTSIZE = 12
FIG_PANEL_TITLE_FONTSIZE = 11  # crop + subject lines above MRI panels
FIG_COMBINED_ROW_HSPACE = 0.30  # vertical gap between case rows (tight; no title/text overlap)
FIG_ROW_HSPACE_SINGLE = 0.09  # single-row grids / dry-run layout (minimal gap)
# Lower scale => smaller coronal image panels (inches per data pixel unit) for balance with text.
FIG_LAYOUT_SCALE = 0.0178
# Shorter panel titles → less band above MRI; keep factor just above 1 so titles + image fit.
FIG_ROW_HEIGHT_FACTOR = 1.66
FIG_PER_ROW_HEIGHT_MULT = 1.02
FIG_TEXT_COL_WIDTH_MULT = 2.85  # text column width vs max coronal width (was 2.6)
# ``tight_layout`` top; legend is placed from title bboxes so this only needs modest headroom.
FIG_TIGHT_RECT_TOP_COMBINED = 0.94
FIG_TIGHT_RECT_TOP_TILE = 0.94
# Figure fraction: gap from top of MRI **title** text to bottom of legend.
FIG_LEGEND_GAP_ABOVE_TITLE = 0.012
# Tiled good-only / bad-only: one case strip per row (``SPLIT_FIG_NCOL=1`` → N rows × 1 column).
# Each row is still before | after | text. Use ``SPLIT_FIG_NCOL=2`` for two strips per row, etc.
SPLIT_FIG_NCOL = 1
SPLIT_FIG_SCALE_MULT = 0.90  # slightly smaller cells than the combined strip figure
SPLIT_FIG_HSPACE = 0.34  # vertical gap between stacked strips (good/bad-only figures)

# Single-subject publication figure (largest-voxel examples)
FOCUS_SUBJECT = "s0175"
# Coronal LR×SI panels: stretch SI (image rows) on screen for the focus subject only.
FOCUS_SUBJECT_VERTICAL_STRETCH = 2.0
GOOD_D_MAX = 0.35   # good: d ≤ 35%
BAD_D_MAX = 0.50    # bad: d ≤ 50%
MIN_VOX_REMOVED = 200  # skip tiny dust; prefer large components
# At most this many good / bad rows per figure; fewer are used if the CSV has fewer candidates.
MAX_CASES_PER_SIDE = 6


def _good_rank_indices(n_show: int) -> Tuple[int, ...]:
    """Ranks into largest-voxel good list: skip the 2nd candidate (index 1) when ``n_show`` ≥ 2."""
    if n_show <= 0:
        return ()
    if n_show == 1:
        return (0,)
    return (0,) + tuple(range(2, n_show + 1))


def _bad_rank_indices(n_show: int) -> Tuple[int, ...]:
    """Ranks into largest-voxel bad list: skip the top candidate (index 0) when ``n_show`` ≥ 2."""
    if n_show <= 0:
        return ()
    if n_show == 1:
        return (0,)
    return tuple(range(1, n_show + 1))


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
    delta_f1: Optional[float] = None


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
    d_f1 = getattr(r, "delta_f1", None)
    delta_f1: Optional[float] = None
    if d_f1 is not None:
        try:
            v = float(d_f1)
            if v == v:  # not NaN
                delta_f1 = v
        except (TypeError, ValueError):
            pass
    return CaseRow(
        subject=str(r.subject),
        crop=str(r.crop),
        tag=str(r.tag),
        structure=str(r.structure),
        delta_pc=float(r.delta_pc),
        precision_before=float(r.precision_before),
        precision_pc=float(r.precision_pc),
        vox_removed_pc=int(r.vox_removed_pc),
        delta_f1=delta_f1,
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


def _pick_sorted_largest_voxels(
    rows: pd.DataFrame,
    n: int,
    *,
    ascending_delta: bool,
    require_positive_delta: Optional[bool] = None,
    delta_column: str = "delta_pc",
) -> List[CaseRow]:
    """Like ``_pick_sorted`` but prefer rows with **largest** ``vox_removed_pc`` first."""
    if require_positive_delta is True:
        rows = rows[rows[delta_column] > 0]
    elif require_positive_delta is False:
        rows = rows[rows[delta_column] < 0]
    rows_sorted = rows.sort_values(
        ["vox_removed_pc", delta_column],
        ascending=[False, ascending_delta],
    )
    out: List[CaseRow] = []
    used_struct = set()
    used_key = set()
    for r in rows_sorted.itertuples(index=False):
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


def pick_cases_s0175_focus(
    df: pd.DataFrame,
    *,
    subject: str = FOCUS_SUBJECT,
    max_good: int = MAX_CASES_PER_SIDE,
    max_bad: int = MAX_CASES_PER_SIDE,
    good_d_max: float = GOOD_D_MAX,
    bad_d_max: float = BAD_D_MAX,
    min_vox_removed: int = MIN_VOX_REMOVED,
    delta_column: Optional[str] = None,
) -> List[Tuple[CaseRow, bool, str]]:
    """Single-subject panel: good vs bad rows ranked by largest ``vox_removed_pc``.

    Uses ``delta_f1`` when present in ``df``; otherwise ``delta_pc`` (ΔDice) with a warning.
    """
    delta_col = delta_column or ("delta_f1" if "delta_f1" in df.columns else "delta_pc")
    if delta_col not in df.columns:
        raise RuntimeError(f"pick_cases_s0175_focus: column {delta_col!r} not in eval CSV")
    if delta_col == "delta_pc" and "delta_f1" not in df.columns:
        print(
            "WARNING: eval CSV has no delta_f1; good/bad pools use ΔDice (delta_pc). "
            "Re-run evaluate_cleaning_methods (cm4) to write F1 columns.",
            flush=True,
        )

    base = df[
        _has_gt_series(df)
        & (df["subject"].astype(str) == subject)
        & (df["vox_removed_pc"] > 0)
    ].copy()

    good_pool = base[(base["d_frac"] <= good_d_max + 1e-9) & (base[delta_col] > 0)].copy()
    bad_pool = base[(base["d_frac"] <= bad_d_max + 1e-9) & (base[delta_col] < 0)].copy()

    def _prefer_min_vox(pool: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
        sub = pool[pool["vox_removed_pc"] >= min_vox_removed]
        if len(sub) >= n:
            return sub
        if len(pool) < n:
            print(
                f"WARNING: {subject} {label}: only {len(pool)} row(s) available "
                f"(need {n}); using all.",
                flush=True,
            )
            return pool
        print(
            f"WARNING: {subject} {label}: only {len(sub)} row(s) with "
            f"vox_removed_pc>={min_vox_removed}; using full pool.",
            flush=True,
        )
        return pool

    fetch_cap = max(max_good, max_bad) + 2

    goods_all = _pick_sorted_largest_voxels(
        _prefer_min_vox(good_pool, fetch_cap, "good"),
        fetch_cap,
        ascending_delta=False,
        require_positive_delta=True,
        delta_column=delta_col,
    )
    bads_all = _pick_sorted_largest_voxels(
        _prefer_min_vox(bad_pool, fetch_cap, "bad"),
        fetch_cap,
        ascending_delta=True,
        require_positive_delta=False,
        delta_column=delta_col,
    )

    n_good = min(max_good, len(goods_all))
    n_bad = min(max_bad, len(bads_all))

    metric = "ΔF1" if delta_col == "delta_f1" else "ΔDice"
    if n_good < 1:
        raise RuntimeError(
            f"{subject} good: need ≥1 row at d≤{good_d_max:g}, {metric}>0, vox_removed_pc>0; found 0"
        )
    if n_bad < 1:
        raise RuntimeError(
            f"{subject} bad: need ≥1 row at d≤{bad_d_max:g}, {metric}<0, vox_removed_pc>0; found 0"
        )

    good_ranks = _good_rank_indices(n_good)
    if n_good >= 2 and len(goods_all) > max(good_ranks):
        goods = [goods_all[i] for i in good_ranks]
    else:
        goods = goods_all[:n_good]

    bad_ranks = _bad_rank_indices(n_bad)
    if n_bad >= 2 and len(bads_all) > max(bad_ranks):
        bads = [bads_all[i] for i in bad_ranks]
    else:
        bads = bads_all[:n_bad]

    order: List[Tuple[CaseRow, bool, str]] = []
    for c in goods:
        order.append((c, False, f"good d≤{good_d_max:g} ({subject})"))
    for c in bads:
        order.append((c, True, f"bad d≤{bad_d_max:g} ({subject})"))
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


@dataclass
class VolumeCm4Bundle:
    """One (subject, crop, tag) volume: predictions + CM4 outputs + GT + artifact MRI."""

    preds: Dict[str, np.ndarray]
    cleaned: Dict[str, np.ndarray]
    removed_tot_log: Dict[str, int]
    events: List[dict]
    gt_masks: Dict[str, np.ndarray]
    mri_art: np.ndarray
    si_ax: int
    si_sign: int
    ap_ax: int
    lr_ax: int


# Cache across rows that share the same experiment folder (avoids duplicate CM4 work).
_volume_bundle_cache: Dict[Tuple[str, str, str], VolumeCm4Bundle] = {}


def clear_volume_bundle_cache() -> None:
    """Drop cached volume bundles (e.g. before a new ``main()`` run in a long-lived process)."""
    _volume_bundle_cache.clear()


def get_volume_cm4_bundle(case: CaseRow, poset: PosetFromJson) -> VolumeCm4Bundle:
    """Load one volume and run CM4 once; reused for every structure row from that volume."""
    key = (case.subject, case.crop, case.tag)
    if key in _volume_bundle_cache:
        return _volume_bundle_cache[key]

    pred_dir = EXP_DIR / case.subject / case.crop / case.tag / "segmentations"
    preds, affine = load_predictions(pred_dir)
    si_ax, si_sign = get_si_info(affine)
    ap_ax = get_ap_axis(affine)
    lr_ax = get_lr_axis(affine)
    pred_shape = next(iter(preds.values())).shape
    gt_masks = load_gt_crop(case.subject, case.crop, pred_shape, si_ax, si_sign)
    mri_art = load_artifact_mri(case, pred_shape)

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

    b = VolumeCm4Bundle(
        preds=preds,
        cleaned=cleaned,
        removed_tot_log=removed_tot_log,
        events=events,
        gt_masks=gt_masks,
        mri_art=mri_art,
        si_ax=si_ax,
        si_sign=si_sign,
        ap_ax=ap_ax,
        lr_ax=lr_ax,
    )
    _volume_bundle_cache[key] = b
    return b


@dataclass
class CaseRowPrep:
    """Heavy work done once; ``draw_case_row_from_prep`` only touches Matplotlib."""

    case: CaseRow
    coronal_shape: Tuple[int, int]
    mri2d_cor: np.ndarray
    rem2d_cor: np.ndarray
    partner2d_cor: np.ndarray
    tp_removed_2d: np.ndarray
    fp_removed_2d: np.ndarray
    summ: Dict[str, Any]


def _prep_lookup_key(case: CaseRow, prefer_tp_event: bool) -> Tuple[str, str, str, str, bool]:
    return (case.subject, case.crop, case.tag, case.structure, bool(prefer_tp_event))


def prepare_case_row_data(case: CaseRow, poset: PosetFromJson, prefer_tp_event: bool) -> CaseRowPrep:
    """Slice + metrics for one row (uses cached volume bundle when possible)."""
    b = get_volume_cm4_bundle(case, poset)
    before = b.preds[case.structure]
    after = b.cleaned[case.structure]
    gt = b.gt_masks.get(case.structure, np.zeros_like(before, dtype=bool))
    full_removed = before & ~after
    if int(full_removed.sum()) != int(b.removed_tot_log.get(case.structure, 0)):
        if full_removed.any():
            raise RuntimeError(
                f"Removal count mismatch for {case.structure}: "
                f"diff={full_removed.sum()} vs logged={b.removed_tot_log.get(case.structure, 0)}"
            )

    evt = best_event_for_structure(b.events, case.structure, full_removed, prefer_tp_event)
    if evt is None and full_removed.any():
        raise RuntimeError(f"No removal event found for {case.structure} in {case.subject}/{case.crop}/{case.tag}")
    if evt is None:
        raise RuntimeError(f"No CM4 removal for {case.structure} in {case.subject}/{case.crop}/{case.tag}")

    partner_name = evt["pair_j"] if evt["target_name"] == evt["pair_i"] else evt["pair_i"]
    partner_mask = b.preds.get(partner_name, np.zeros_like(before, dtype=bool))

    sl_cor = best_slice_union(full_removed, partner_mask, b.ap_ax)
    mri2d_cor = extract_oriented_plane(b.mri_art, b.ap_ax, sl_cor, x_axis=b.lr_ax, y_axis=b.si_ax)
    rem2d_cor = extract_oriented_footprint(full_removed, b.ap_ax, b.lr_ax, b.si_ax)
    partner2d_cor = extract_oriented_footprint(partner_mask, b.ap_ax, b.lr_ax, b.si_ax)

    tp_removed_2d = extract_oriented_footprint(full_removed & gt, b.ap_ax, b.lr_ax, b.si_ax)
    fp_removed_2d = extract_oriented_footprint(full_removed & (~gt), b.ap_ax, b.lr_ax, b.si_ax)

    f1_before = f1(before, gt)
    f1_after = f1(after, gt)
    delta_f1_live = f1_after - f1_before

    fr_tot = int(full_removed.sum())
    fr_tp = int((full_removed & gt).sum())
    fr_fp = int((full_removed & (~gt)).sum())

    slim_evt = {
        k: evt[k]
        for k in (
            "pair_i",
            "pair_j",
            "step_kind",
            "target_name",
            "removed_voxels",
            "removed_tp",
            "removed_fp",
            "below_limit",
            "above_limit",
            "protect_anchor",
            "anchor_label",
            "lcc_label",
            "anchor_voxels",
            "anchor_extent",
            "lcc_extent",
        )
        if k in evt
    }
    summ: Dict[str, Any] = {
        "coronal_shape": rem2d_cor.shape,
        "event": slim_evt,
        "delta_f1": delta_f1_live,
        "f1_before": f1_before,
        "f1_after": f1_after,
        "partner_structure": partner_name,
        "pair_i": str(evt["pair_i"]),
        "pair_j": str(evt["pair_j"]),
        "full_removed_total": fr_tot,
        "full_removed_tp": fr_tp,
        "full_removed_fp": fr_fp,
        "event_removed_voxels": int(evt["removed_voxels"]),
        "event_removed_tp": int(evt["removed_tp"]),
        "event_removed_fp": int(evt["removed_fp"]),
    }

    return CaseRowPrep(
        case=case,
        coronal_shape=(int(rem2d_cor.shape[0]), int(rem2d_cor.shape[1])),
        mri2d_cor=mri2d_cor,
        rem2d_cor=rem2d_cor,
        partner2d_cor=partner2d_cor,
        tp_removed_2d=tp_removed_2d,
        fp_removed_2d=fp_removed_2d,
        summ=summ,
    )


def draw_case_row_from_prep(
    ax_row,
    prep: CaseRowPrep,
    label: str,
    *,
    stretch_subject: Optional[str] = None,
    show_text_col: bool = True,
) -> None:
    """Render one strip from ``CaseRowPrep``.

    ``show_text_col=True``  (default): ax_row length 2 — combined MRI | text annotation.
    ``show_text_col=False``: ax_row length 1 — MRI panel only with compact title.
    """
    case = prep.case
    ss = stretch_subject if stretch_subject is not None else FOCUS_SUBJECT
    y_stretch = FOCUS_SUBJECT_VERTICAL_STRETCH if case.subject == ss else 1.0
    summ = prep.summ
    d1 = float(summ["delta_f1"])

    imshow_native(
        ax_row[0],
        overlay_combined_cm4(prep.mri2d_cor, prep.partner2d_cor, prep.tp_removed_2d, prep.fp_removed_2d),
    )
    if show_text_col:
        title = f"{case.crop}\n{case.subject}"
    else:
        title = f"{case.structure}  ΔF1={d1:+.3f}\n{case.tag}"
    ax_row[0].set_title(title, fontsize=FIG_PANEL_TITLE_FONTSIZE, pad=3)
    _set_pixel_true_aspect(ax_row[0], prep.rem2d_cor.shape, vertical_stretch=y_stretch)
    ax_row[0].axis("off")

    if not show_text_col:
        return

    ax_row[1].axis("off")
    sk = strip_cm4_from_figure_text(str(summ["event"]["step_kind"]))
    partner = str(summ["partner_structure"])
    frt = int(summ["full_removed_total"])
    frp = int(summ["full_removed_tp"])
    frf = int(summ["full_removed_fp"])
    txt = (
        f"{label}  ΔF1={d1:+.5f}\n"
        f"{case.tag}\n"
        f"step: {sk}\n"
        f"removed structure: {case.structure}\n"
        f"trusted anchor (purple): {partner}\n"
        f"removed voxels (3D): {frt:,}  (TP {frp:,}, FP {frf:,})\n"
        "(detail in .md)"
    )
    ax_row[1].text(
        0.02,
        0.5,
        txt,
        va="center",
        ha="left",
        fontsize=FIG_TEXT_FONTSIZE,
        family="monospace",
        linespacing=1.25,
        transform=ax_row[1].transAxes,
        clip_on=True,
    )


def precompute_row_preps(
    cases_order: List[Tuple[CaseRow, bool, str]], poset: PosetFromJson
) -> List[CaseRowPrep]:
    """Run the expensive path once per row (volume work is deduplicated)."""
    return [prepare_case_row_data(case, poset, pref_tp) for case, pref_tp, _lab in cases_order]


def precompute_row_preps_safe(
    cases_order: List[Tuple[CaseRow, bool, str]], poset: PosetFromJson
) -> Tuple[List[Tuple[CaseRow, bool, str]], List[CaseRowPrep]]:
    """Like ``precompute_row_preps`` but skips cases that raise RuntimeError (no CM4 event)."""
    valid_cases: List[Tuple[CaseRow, bool, str]] = []
    preps: List[CaseRowPrep] = []
    for case, pref_tp, lab in cases_order:
        try:
            preps.append(prepare_case_row_data(case, poset, pref_tp))
            valid_cases.append((case, pref_tp, lab))
        except RuntimeError as exc:
            print(f"WARNING: skipping {case.structure}/{case.subject}/{case.tag}: {exc}", flush=True)
    return valid_cases, preps


def _coronal_shapes_from_preps(preps: List[CaseRowPrep]) -> List[Tuple[int, int]]:
    return [p.coronal_shape for p in preps]


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
    """Blue = targeted-structure removals; purple = trusted-anchor (constraint) partner."""
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


def overlay_combined_cm4(
    mri2d: np.ndarray,
    partner_purple_2d: np.ndarray,
    tp_red_2d: np.ndarray,
    fp_green_2d: np.ndarray,
) -> np.ndarray:
    """Single-panel overlay: purple = trusted anchor; green = removed FP; red = removed TP (wins on overlap)."""
    g = _normalize_mri(mri2d)
    rgb = np.stack([g, g, g], axis=-1)
    pp = partner_purple_2d.astype(bool)
    fg = fp_green_2d.astype(bool)
    tr = tp_red_2d.astype(bool)
    rgb[pp, 0] = np.maximum(rgb[pp, 0], 0.72)
    rgb[pp, 1] = np.minimum(rgb[pp, 1], 0.42)
    rgb[pp, 2] = np.maximum(rgb[pp, 2], 0.88)
    rgb[fg, 0] *= 0.45
    rgb[fg, 1] = np.maximum(rgb[fg, 1], 0.94)
    rgb[fg, 2] *= 0.45
    rgb[tr, 0] = np.maximum(rgb[tr, 0], 0.96)
    rgb[tr, 1] *= 0.35
    rgb[tr, 2] *= 0.35
    return np.clip(rgb, 0, 1)


def _set_pixel_true_aspect(
    ax, shape2d: Tuple[int, int], *, vertical_stretch: float = 1.0
) -> None:
    """Square data pixels by default; ``vertical_stretch>1`` elongates the SI axis on screen."""
    h, w = int(shape2d[0]), int(shape2d[1])
    if abs(vertical_stretch - 1.0) < 1e-9:
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(h / max(w, 1))
        return
    # Numeric aspect: displayed y-unit is ``vertical_stretch`` × longer than x-unit (matplotlib).
    ax.set_aspect(vertical_stretch, adjustable="box")
    ax.set_box_aspect((h / max(w, 1)) * vertical_stretch)


def imshow_native(ax, img: np.ndarray) -> None:
    h, w = img.shape[:2]
    ax.imshow(img, origin="lower", interpolation="none", aspect="equal")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)


def visualize_case_row(
    ax_row,
    case: CaseRow,
    poset: PosetFromJson,
    prefer_tp_event: bool,
    label: str,
    *,
    stretch_subject: Optional[str] = None,
) -> Dict[str, Any]:
    """ax_row: length 3 — before | after | text. Prefer ``prepare_case_row_data`` + ``draw_case_row_from_prep`` when batching."""
    prep = prepare_case_row_data(case, poset, prefer_tp_event)
    draw_case_row_from_prep(ax_row, prep, label, stretch_subject=stretch_subject)
    return prep.summ


def _overlay_legend_patches() -> List[Patch]:
    return [
        Patch(
            facecolor=(0.72, 0.35, 0.88),
            edgecolor="purple",
            linewidth=0.7,
            label="Trusted anchor (purple): partner silhouette (A/P)",
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


def _add_overlay_legend_above_titles(
    fig: plt.Figure,
    title_axes: List,
    *,
    pad_frac: float = FIG_LEGEND_GAP_ABOVE_TITLE,
) -> None:
    """Legend horizontal center of figure, vertical gap above the MRI panel titles (no overlap)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    y_top = 0.0
    for ax in title_axes:
        t = ax.title
        if t.get_text():
            bb = t.get_window_extent(renderer=renderer).transformed(inv)
            y_top = max(y_top, float(bb.y1))
    if y_top <= 0.0:
        y_top = float(title_axes[0].get_position().y1)
    y = min(y_top + pad_frac, 0.999)
    fig.legend(
        handles=_overlay_legend_patches(),
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=2,
        fontsize=FIG_TEXT_FONTSIZE,
        frameon=True,
        borderaxespad=0.0,
    )


def render_good_bad_grid(
    cases_order: List[Tuple[CaseRow, bool, str]],
    poset: PosetFromJson,
    *,
    out_md: Path,
    md_header: str,
    stretch_subject: Optional[str] = None,
    precomputed_preps: Optional[List[CaseRowPrep]] = None,
    out_png: Optional[Path] = None,
) -> None:
    """Write companion markdown; optionally also build the combined good+bad PNG (``out_png``)."""
    if precomputed_preps is not None:
        if len(precomputed_preps) != len(cases_order):
            raise ValueError("precomputed_preps length must match cases_order")
        preps = precomputed_preps
    else:
        preps = precompute_row_preps(cases_order, poset)

    md_lines = [md_header, ""]
    for (case, pref_tp, lab), prep in zip(cases_order, preps):
        summ = prep.summ
        sk_md = strip_cm4_from_figure_text(str(summ["event"]["step_kind"]))
        md_lines.extend(
            [
                f"- **{lab}** — targeted structure `{case.structure}` @ `{case.subject}` / `{case.crop}` / `{case.tag}`",
                f"  - **Trusted anchor (purple silhouette)**: `{summ['partner_structure']}` · poset pair "
                f"`{summ['pair_i']}` / `{summ['pair_j']}`",
                f"  - **Step (caption)**: `{sk_md}`",
                f"  - **F1** (full 3D mask vs GT): {summ['f1_before']:.5f} → {summ['f1_after']:.5f} "
                f"(Δ {summ['delta_f1']:+.5f})",
                f"  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): "
                f"{summ['full_removed_total']:,} total; TP (red) {summ['full_removed_tp']:,}; "
                f"FP (green) {summ['full_removed_fp']:,}",
                f"  - **Caption event only** (one logged step; subset if multiple steps touched the organ): "
                f"{summ['event_removed_voxels']:,} removed; TP {summ['event_removed_tp']:,}; "
                f"FP {summ['event_removed_fp']:,}",
                "",
            ]
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Saved: {out_md}")

    if out_png is None:
        return

    shapes = _coronal_shapes_from_preps(preps)
    cor_h = max(s[0] for s in shapes)
    cor_w = max(s[1] for s in shapes)
    width_ratios = [cor_w, int(FIG_TEXT_COL_WIDTH_MULT * cor_w)]
    scale = FIG_LAYOUT_SCALE
    fig_w = scale * sum(width_ratios)
    fig_h = scale * (FIG_ROW_HEIGHT_FACTOR * cor_h) * FIG_PER_ROW_HEIGHT_MULT * len(cases_order)

    gs_kw = {"width_ratios": width_ratios}
    if len(cases_order) > 1:
        gs_kw["hspace"] = FIG_COMBINED_ROW_HSPACE
    fig, axes = plt.subplots(
        len(cases_order),
        2,
        figsize=(fig_w, fig_h),
        gridspec_kw=gs_kw,
    )
    if len(cases_order) == 1:
        axes = np.array([axes])

    for i, ((case, pref_tp, lab), prep) in enumerate(zip(cases_order, preps)):
        draw_case_row_from_prep(
            axes[i],
            prep,
            lab,
            stretch_subject=stretch_subject,
        )

    fig.tight_layout(rect=[0, 0.06, 1, FIG_TIGHT_RECT_TOP_COMBINED])
    _add_overlay_legend_above_titles(fig, [axes[0, 0]])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def render_case_tile_grid(
    cases_order: List[Tuple[CaseRow, bool, str]],
    poset: PosetFromJson,
    *,
    out_png: Path,
    outer_ncol: int = SPLIT_FIG_NCOL,
    stretch_subject: Optional[str] = None,
    precomputed_preps: Optional[List[CaseRowPrep]] = None,
    show_text_col: bool = True,
    text_col_width_mult: float = FIG_TEXT_COL_WIDTH_MULT,
) -> None:
    """Good-only or bad-only figure: ``outer_ncol`` case strips per row. No suptitle.

    ``show_text_col=False`` renders MRI-only panels (compact, suitable for multi-column layouts).
    """
    n = len(cases_order)
    if n == 0:
        print(f"Skipping empty tile grid: {out_png.name}", flush=True)
        return

    if precomputed_preps is not None:
        if len(precomputed_preps) != len(cases_order):
            raise ValueError("precomputed_preps length must match cases_order")
        preps = precomputed_preps
    else:
        preps = precompute_row_preps(cases_order, poset)
    shapes = _coronal_shapes_from_preps(preps)
    cor_h = max(s[0] for s in shapes)
    cor_w = max(s[1] for s in shapes)
    if show_text_col:
        width_ratios_inner = [cor_w, int(text_col_width_mult * cor_w)]
        inner_ncols = 2
    else:
        width_ratios_inner = [cor_w]
        inner_ncols = 1
    cell_w = sum(width_ratios_inner)
    ncol = max(1, outer_ncol)
    nrow = (n + ncol - 1) // ncol
    scale = FIG_LAYOUT_SCALE * SPLIT_FIG_SCALE_MULT
    fig_w = scale * cell_w * ncol
    fig_h = scale * (FIG_ROW_HEIGHT_FACTOR * cor_h) * FIG_PER_ROW_HEIGHT_MULT * nrow

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = mgs.GridSpec(nrow, ncol, figure=fig, hspace=SPLIT_FIG_HSPACE, wspace=0.02)
    first_title_axes: Optional[List] = None
    for i in range(n):
        r, c = divmod(i, ncol)
        inner = mgs.GridSpecFromSubplotSpec(
            1,
            inner_ncols,
            subplot_spec=outer[r, c],
            wspace=0.04,
            width_ratios=width_ratios_inner,
        )
        ax0 = fig.add_subplot(inner[0, 0])
        axes_row = [ax0]
        if show_text_col:
            ax1 = fig.add_subplot(inner[0, 1])
            axes_row.append(ax1)
        if first_title_axes is None:
            first_title_axes = [ax0]
        case, pref_tp, lab = cases_order[i]
        draw_case_row_from_prep(
            np.array(axes_row),
            preps[i],
            lab,
            stretch_subject=stretch_subject,
            show_text_col=show_text_col,
        )

    fig.tight_layout(rect=[0, 0.05, 1, FIG_TIGHT_RECT_TOP_TILE])
    assert first_title_axes is not None
    _add_overlay_legend_above_titles(fig, first_title_axes)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CM4 real-case publication figures from wraparound_v4_eval_cm4 CSVs.",
    )
    parser.add_argument(
        "--subject",
        default=FOCUS_SUBJECT,
        help=f"Totalsegmentator subject id (default: {FOCUS_SUBJECT})",
    )
    args = parser.parse_args()
    subject = str(args.subject).strip()
    if not subject:
        raise SystemExit("error: --subject must be non-empty")

    tag = "" if subject == FOCUS_SUBJECT else f"_{subject}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_volume_bundle_cache()
    df = load_merged_or_partials(EVAL_CM4_T100)
    poset = load_poset_from_json(str(POSET_PATH))

    use_f1 = "delta_f1" in df.columns
    m_pos = "ΔF1>0" if use_f1 else "ΔDice>0"
    m_neg = "ΔF1<0" if use_f1 else "ΔDice<0"
    pool_note = (
        "ΔF1 from eval CSV when present; figure panels recompute F1 from shown masks."
        if use_f1
        else "Good/bad pools use ΔDice (CSV has no delta_f1 — re-run evaluate_cleaning_methods)."
    )

    SECOND_SUBJECT = "s0167"
    EXTRA_SUBJECT_A = "s0250"   # stomach examples
    EXTRA_SUBJECT_B = "s0186"   # gluteus_medius examples
    subjects_for_tiles = [subject, SECOND_SUBJECT] if subject != SECOND_SUBJECT else [subject]
    extra_subjects = [EXTRA_SUBJECT_A, EXTRA_SUBJECT_B]

    # Collect safe (validated) cases + preps for both subjects
    goods_by_subj: Dict[str, Tuple[List, List]] = {}
    bads_by_subj: Dict[str, Tuple[List, List]] = {}

    for subj in subjects_for_tiles + extra_subjects:
        subj_tag = "" if subj == FOCUS_SUBJECT else f"_{subj}"
        extra_kw = {"max_bad": 2} if subj in extra_subjects else {}
        cases_raw = pick_cases_s0175_focus(df, subject=subj, **extra_kw)
        cases_subj, row_preps_subj = precompute_row_preps_safe(cases_raw, poset)
        prep_by_key_subj = {
            _prep_lookup_key(case, pref): prep
            for (case, pref, _), prep in zip(cases_subj, row_preps_subj)
        }
        render_good_bad_grid(
            cases_subj,
            poset,
            out_md=OUT_DIR / f"cm4_real_case_details{subj_tag}.md",
            md_header=(
                f"# Real cleaning cases — subject {subj} (coronal; SI axis 2× screen stretch)\n\n"
                f"Good rows: d≤{GOOD_D_MAX:g}, {m_pos}, ranked by largest `vox_removed_pc` "
                f"(up to {MAX_CASES_PER_SIDE}; editorial rank skip when enough candidates). "
                f"Bad rows: d≤{BAD_D_MAX:g}, {m_neg}, same rule (up to {MAX_CASES_PER_SIDE}). "
                f"{pool_note}\n\n"
                "Figure: one coronal MRI slice; mask overlays are **A/P silhouettes** on LR×SI. "
                + (
                    f"For this subject (`{subj}`), the SI axis is drawn **2× taller** on screen than LR. "
                    if subj == FOCUS_SUBJECT
                    else f"For this subject (`{subj}`), LR and SI use **equal** on-screen pixel scale. "
                )
                + "**Trusted anchor (purple)** = constraint partner; "
                "**red** = removed TP, **green** = removed FP."
            ),
            stretch_subject=subj,
            precomputed_preps=row_preps_subj,
        )
        goods_subj = [t for t in cases_subj if not t[1]]
        bads_subj = [t for t in cases_subj if t[1]]
        goods_preps_subj = [prep_by_key_subj[_prep_lookup_key(c, p)] for c, p, _ in goods_subj]
        bads_preps_subj = [prep_by_key_subj[_prep_lookup_key(c, p)] for c, p, _ in bads_subj]
        goods_by_subj[subj] = (goods_subj[:5], goods_preps_subj[:5])
        bads_by_subj[subj] = (bads_subj[:4], bads_preps_subj[:4])

    # Combined two-column figures: subjects interleaved so col 0 = subj_left, col 1 = subj_right.
    # Interleave: [L0, R0, L1, R1, ...] with ncol=2 maps to left/right columns.
    def _interleave(
        left: Tuple[List, List], right: Tuple[List, List]
    ) -> Tuple[List, List]:
        lc, lp = left
        rc, rp = right
        combined_cases: List = []
        combined_preps: List = []
        for i in range(max(len(lc), len(rc))):
            if i < len(lc):
                combined_cases.append(lc[i])
                combined_preps.append(lp[i])
            if i < len(rc):
                combined_cases.append(rc[i])
                combined_preps.append(rp[i])
        return combined_cases, combined_preps

    subj_left, subj_right = subjects_for_tiles[0], subjects_for_tiles[-1]
    good_cases_comb, good_preps_comb = _interleave(goods_by_subj[subj_left], goods_by_subj[subj_right])
    bad_cases_comb, bad_preps_comb = _interleave(bads_by_subj[subj_left], bads_by_subj[subj_right])

    render_case_tile_grid(
        good_cases_comb,
        poset,
        out_png=OUT_DIR / "cm4_real_good_examples.png",
        outer_ncol=2,
        precomputed_preps=good_preps_comb,
        show_text_col=True,
        text_col_width_mult=1.6,
    )
    render_case_tile_grid(
        bad_cases_comb,
        poset,
        out_png=OUT_DIR / "cm4_real_bad_examples.png",
        outer_ncol=2,
        precomputed_preps=bad_preps_comb,
        show_text_col=True,
        text_col_width_mult=1.6,
    )

    # Extra good-cases figure: s0250 (stomach) + s0186 (gluteus_medius), top 5 each
    extra_good_cases, extra_good_preps = _interleave(
        goods_by_subj[EXTRA_SUBJECT_A], goods_by_subj[EXTRA_SUBJECT_B]
    )
    if extra_good_cases:
        render_case_tile_grid(
            extra_good_cases,
            poset,
            out_png=OUT_DIR / "cm4_real_good_examples_extra.png",
            outer_ncol=2,
            precomputed_preps=extra_good_preps,
            show_text_col=True,
            text_col_width_mult=1.6,
        )


if __name__ == "__main__":
    main()
