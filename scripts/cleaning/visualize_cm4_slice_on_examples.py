"""
One figure per case: coronal MRI + **on-slice** masks only (no A/P silhouette projection).

Selects rows where the **A/P silhouette** of ``removal ∪ partner`` is not much larger than
the **union area on the best coronal slice** (ratio ≤ ``MAX_FOOTPRINT_TO_SLICE_RATIO``),
and both masks have a minimum on-slice footprint. Selection scans at most
``MAX_VOLUME_GROUPS_TO_SCAN`` artifact volumes per polarity (sorted by ΔDice), then
greedily picks diverse rows. **Tuning**: if you get fewer than 10 per side, raise
``MAX_VOLUME_GROUPS_TO_SCAN`` or relax ``MAX_FOOTPRINT_TO_SLICE_RATIO`` / ``MIN_SLICE_*``.

Outputs:
  results/cm4_real_cases/cm4_slice_on_good_{01..10}.png
  results/cm4_real_cases/cm4_slice_on_bad_{01..10}.png

``--large-area`` additionally writes:

  results/cm4_real_cases/cm4_slice_large_good_{01..10}.png
  results/cm4_real_cases/cm4_slice_large_bad_{01..10}.png

  (slice-only overlays chosen to maximize on-slice removal and partner area; no silhouette-ratio cap.)

Usage:
  python scripts/cleaning/visualize_cm4_slice_on_examples.py
  python scripts/cleaning/visualize_cm4_slice_on_examples.py --large-area
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "cleaning"))

from anatomy_poset.core.io import load_poset_from_json, PosetFromJson
from evaluate_cleaning_methods import dice, get_si_info, precision

from visualize_cm4_real_modes import (
    CaseRow,
    EVAL_CM4_T100,
    POSET_PATH,
    THRESHOLD,
    best_event_for_structure,
    best_slice_union,
    extract_oriented_plane,
    extract_oriented_footprint,
    get_ap_axis,
    get_lr_axis,
    load_artifact_mri,
    load_gt_crop,
    load_merged_or_partials,
    load_predictions,
    method4_center_conflict,
    overlay_after_tp_fp,
    overlay_before_cm4,
    run_cm4_with_logs,
    strip_cm4_from_figure_text,
    _has_gt_series,
    _set_pixel_true_aspect,
    imshow_native,
)

OUT_DIR = PROJECT_ROOT / "results" / "cm4_real_cases"
CaseGeom = Tuple[CaseRow, dict]

# Silhouette area / slice union area — close to 1 means the slice already carries the bulk
# of the 3D extent (organs not “thin sheets” along A/P on this view).
MAX_FOOTPRINT_TO_SLICE_RATIO = 2.85
# Require both partner and removal blob to occupy a noticeable area on the slice.
MIN_SLICE_VOX_PER_MASK = 200
MIN_SLICE_UNION = 420
# Candidate collection stops after this many passing rows (enough for greedy diversity).
MAX_PASSING_CANDIDATES = 80
# Hard cap on distinct volumes loaded while scanning (each load runs CM4 twice; ~15–25 s each).
MAX_VOLUME_GROUPS_TO_SCAN = 45

# --- “large both masks on slice” batch (``--large-area``) ---
# Scan more volumes because we sort globally by area after collection.
MAX_VOLUME_GROUPS_LARGE_AREA = 80
# Minimum on-slice voxels for removal and for partner (both must pass).
MIN_LARGE_EACH_SLICE = 270
# At most this many (case, geom) rows kept before sorting (safety cap).
MAX_LARGE_AREA_CANDIDATES = 400


def _wrap_to_width(text: str, width: int) -> str:
    text = text.strip()
    if not text:
        return text
    parts: List[str] = []
    for para in text.split("\n"):
        p = para.strip()
        if not p:
            parts.append("")
            continue
        parts.extend(
            textwrap.wrap(
                p,
                width=max(8, width),
                break_long_words=True,
                break_on_hyphens=True,
            )
        )
    return "\n".join(parts)


def _metrics_slice_vs_footprint(
    full_removed: np.ndarray,
    partner_mask: np.ndarray,
    ap_ax: int,
    lr_ax: int,
    si_ax: int,
) -> Optional[Tuple[int, float, int, int]]:
    """Returns (sl_cor, ratio, A_slice, A_foot) or None if slice filter fails."""
    if not full_removed.any():
        return None
    sl = best_slice_union(full_removed, partner_mask, ap_ax)
    union3d = full_removed | partner_mask
    slice_u = extract_oriented_plane(union3d, ap_ax, sl, lr_ax, si_ax)
    foot_u = extract_oriented_footprint(union3d, ap_ax, lr_ax, si_ax)
    A_slice = int(slice_u.sum())
    A_foot = int(foot_u.sum())
    if A_slice < MIN_SLICE_UNION:
        return None
    tgt_sl = extract_oriented_plane(full_removed, ap_ax, sl, lr_ax, si_ax)
    pr_sl = extract_oriented_plane(partner_mask, ap_ax, sl, lr_ax, si_ax)
    if int(tgt_sl.sum()) < MIN_SLICE_VOX_PER_MASK or int(pr_sl.sum()) < MIN_SLICE_VOX_PER_MASK:
        return None
    ratio = A_foot / max(A_slice, 1)
    if ratio > MAX_FOOTPRINT_TO_SLICE_RATIO:
        return None
    return sl, ratio, A_slice, A_foot


def _metrics_both_large_on_slice(
    full_removed: np.ndarray,
    partner_mask: np.ndarray,
    ap_ax: int,
    lr_ax: int,
    si_ax: int,
) -> Optional[Tuple[int, int, int, int, int, float]]:
    """Best coronal slice; require both masks large on-slice. No silhouette ratio cap.

    Returns ``(sl, area_tgt, area_partner, A_union_slice, A_foot_union, ratio)``.
    """
    if not full_removed.any():
        return None
    sl = best_slice_union(full_removed, partner_mask, ap_ax)
    tgt_sl = extract_oriented_plane(full_removed, ap_ax, sl, lr_ax, si_ax)
    pr_sl = extract_oriented_plane(partner_mask, ap_ax, sl, lr_ax, si_ax)
    at = int(tgt_sl.sum())
    ap = int(pr_sl.sum())
    if at < MIN_LARGE_EACH_SLICE or ap < MIN_LARGE_EACH_SLICE:
        return None
    union3d = full_removed | partner_mask
    slice_u = extract_oriented_plane(union3d, ap_ax, sl, lr_ax, si_ax)
    foot_u = extract_oriented_footprint(union3d, ap_ax, lr_ax, si_ax)
    A_slice = int(slice_u.sum())
    A_foot = int(foot_u.sum())
    ratio = A_foot / max(A_slice, 1)
    return sl, at, ap, A_slice, A_foot, ratio


def _volume_context(case: CaseRow, poset: PosetFromJson) -> dict:
    """Load one artifact volume + run CM4 once (shared across all structures in that volume)."""
    pred_dir = (
        PROJECT_ROOT
        / "data"
        / "experiments"
        / "wraparound_v4"
        / case.subject
        / case.crop
        / case.tag
        / "segmentations"
    )
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
                f"CM4 drift for {case.subject}/{case.crop}/{case.tag} mask {name!r}"
            )

    return {
        "preds": preds,
        "cleaned": cleaned,
        "affine": affine,
        "si_ax": si_ax,
        "si_sign": si_sign,
        "ap_ax": ap_ax,
        "lr_ax": lr_ax,
        "gt_masks": gt_masks,
        "mri_art": mri_art,
        "removed_tot_log": removed_tot_log,
        "events": events,
    }


def _geometry_for_structure(ctx: dict, case: CaseRow) -> Optional[dict]:
    """Slice-compactness filter + per-structure fields (no disk IO)."""
    preds = ctx["preds"]
    cleaned = ctx["cleaned"]
    ap_ax = ctx["ap_ax"]
    lr_ax = ctx["lr_ax"]
    si_ax = ctx["si_ax"]
    gt_masks = ctx["gt_masks"]
    mri_art = ctx["mri_art"]
    removed_tot_log = ctx["removed_tot_log"]
    events = ctx["events"]

    if case.structure not in preds:
        return None

    before = preds[case.structure]
    after = cleaned[case.structure]
    gt = gt_masks.get(case.structure, np.zeros_like(before, dtype=bool))
    full_removed = before & ~after
    if int(full_removed.sum()) != int(removed_tot_log.get(case.structure, 0)):
        if full_removed.any():
            raise RuntimeError(
                f"Removal count mismatch for {case.structure} in "
                f"{case.subject}/{case.crop}/{case.tag}"
            )

    prefer_tp = case.delta_pc >= 0
    evt = best_event_for_structure(events, case.structure, full_removed, prefer_tp)
    if evt is None:
        return None

    partner_name = evt["pair_j"] if evt["target_name"] == evt["pair_i"] else evt["pair_i"]
    partner_mask = preds.get(partner_name, np.zeros_like(before, dtype=bool))

    met = _metrics_slice_vs_footprint(full_removed, partner_mask, ap_ax, lr_ax, si_ax)
    if met is None:
        return None
    sl, ratio, A_slice, A_foot = met

    return {
        "preds": preds,
        "cleaned": cleaned,
        "affine": ctx["affine"],
        "si_ax": si_ax,
        "si_sign": ctx["si_sign"],
        "ap_ax": ap_ax,
        "lr_ax": lr_ax,
        "gt_masks": gt_masks,
        "mri_art": mri_art,
        "before": before,
        "after": after,
        "gt": gt,
        "full_removed": full_removed,
        "evt": evt,
        "partner_name": partner_name,
        "partner_mask": partner_mask,
        "sl_cor": sl,
        "ratio": ratio,
        "A_slice": A_slice,
        "A_foot": A_foot,
    }


def _geometry_for_structure_large_area(ctx: dict, case: CaseRow) -> Optional[dict]:
    """Same as compact path but score by large on-slice removal + partner (no ratio filter)."""
    preds = ctx["preds"]
    cleaned = ctx["cleaned"]
    ap_ax = ctx["ap_ax"]
    lr_ax = ctx["lr_ax"]
    si_ax = ctx["si_ax"]
    gt_masks = ctx["gt_masks"]
    mri_art = ctx["mri_art"]
    removed_tot_log = ctx["removed_tot_log"]
    events = ctx["events"]

    if case.structure not in preds:
        return None

    before = preds[case.structure]
    after = cleaned[case.structure]
    gt = gt_masks.get(case.structure, np.zeros_like(before, dtype=bool))
    full_removed = before & ~after
    if int(full_removed.sum()) != int(removed_tot_log.get(case.structure, 0)):
        if full_removed.any():
            raise RuntimeError(
                f"Removal count mismatch for {case.structure} in "
                f"{case.subject}/{case.crop}/{case.tag}"
            )

    prefer_tp = case.delta_pc >= 0
    evt = best_event_for_structure(events, case.structure, full_removed, prefer_tp)
    if evt is None:
        return None

    partner_name = evt["pair_j"] if evt["target_name"] == evt["pair_i"] else evt["pair_i"]
    partner_mask = preds.get(partner_name, np.zeros_like(before, dtype=bool))

    met = _metrics_both_large_on_slice(full_removed, partner_mask, ap_ax, lr_ax, si_ax)
    if met is None:
        return None
    sl, at, ap, A_slice, A_foot, ratio = met
    min_sl = min(at, ap)

    return {
        "preds": preds,
        "cleaned": cleaned,
        "affine": ctx["affine"],
        "si_ax": si_ax,
        "si_sign": ctx["si_sign"],
        "ap_ax": ap_ax,
        "lr_ax": lr_ax,
        "gt_masks": gt_masks,
        "mri_art": mri_art,
        "before": before,
        "after": after,
        "gt": gt,
        "full_removed": full_removed,
        "evt": evt,
        "partner_name": partner_name,
        "partner_mask": partner_mask,
        "sl_cor": sl,
        "ratio": ratio,
        "A_slice": A_slice,
        "A_foot": A_foot,
        "area_tgt_sl": at,
        "area_partner_sl": ap,
        "min_sl_area": min_sl,
    }


def _row_to_case(r) -> CaseRow:
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


def _collect_passing_cases(pool: pd.DataFrame, poset: PosetFromJson, ascending: bool) -> List[CaseGeom]:
    """One NIfTI load per (subject, crop, tag); keep rows that pass slice-compactness filters.

    Scans at most ``MAX_VOLUME_GROUPS_TO_SCAN`` volumes (in ΔDice order) and returns up to
    ``MAX_PASSING_CANDIDATES`` passing (case, geometry) pairs for greedy selection. Geometry
    is reused when plotting so volumes are not loaded twice.
    """
    pool_sorted = pool.sort_values("delta_pc", ascending=ascending)
    triple_order: List[Tuple[str, str, str]] = []
    seen_triple: set[Tuple[str, str, str]] = set()
    for r in pool_sorted.itertuples(index=False):
        t = (str(r.subject), str(r.crop), str(r.tag))
        if t in seen_triple:
            continue
        seen_triple.add(t)
        triple_order.append(t)
        if len(triple_order) >= MAX_VOLUME_GROUPS_TO_SCAN:
            break

    out: List[CaseGeom] = []
    for t in triple_order:
        if len(out) >= MAX_PASSING_CANDIDATES:
            break
        subdf = pool_sorted[
            (pool_sorted["subject"] == t[0])
            & (pool_sorted["crop"] == t[1])
            & (pool_sorted["tag"] == t[2])
        ].sort_values("delta_pc", ascending=ascending)
        try:
            ctx = _volume_context(_row_to_case(subdf.iloc[0]), poset)
        except (RuntimeError, OSError, FileNotFoundError, ValueError):
            continue
        for r in subdf.itertuples(index=False):
            if len(out) >= MAX_PASSING_CANDIDATES:
                break
            case = _row_to_case(r)
            geom = _geometry_for_structure(ctx, case)
            if geom is not None:
                out.append((case, geom))
    return out


def _collect_large_area_cases(pool: pd.DataFrame, poset: PosetFromJson, ascending: bool) -> List[CaseGeom]:
    pool_sorted = pool.sort_values("delta_pc", ascending=ascending)
    triple_order: List[Tuple[str, str, str]] = []
    seen_triple: set[Tuple[str, str, str]] = set()
    for r in pool_sorted.itertuples(index=False):
        t = (str(r.subject), str(r.crop), str(r.tag))
        if t in seen_triple:
            continue
        seen_triple.add(t)
        triple_order.append(t)
        if len(triple_order) >= MAX_VOLUME_GROUPS_LARGE_AREA:
            break

    out: List[CaseGeom] = []
    for t in triple_order:
        if len(out) >= MAX_LARGE_AREA_CANDIDATES:
            break
        subdf = pool_sorted[
            (pool_sorted["subject"] == t[0])
            & (pool_sorted["crop"] == t[1])
            & (pool_sorted["tag"] == t[2])
        ].sort_values("delta_pc", ascending=ascending)
        try:
            ctx = _volume_context(_row_to_case(subdf.iloc[0]), poset)
        except (RuntimeError, OSError, FileNotFoundError, ValueError):
            continue
        for r in subdf.itertuples(index=False):
            if len(out) >= MAX_LARGE_AREA_CANDIDATES:
                break
            case = _row_to_case(r)
            geom = _geometry_for_structure_large_area(ctx, case)
            if geom is not None:
                out.append((case, geom))
    return out


def _pick_top_diverse_by_min_slice_area(
    candidates: List[CaseGeom],
    n: int,
    max_per_subject: int = 2,
) -> List[CaseGeom]:
    """Sort by largest min(removal, partner) on slice, then union; greedy fill with diversity."""
    sorted_c = sorted(
        candidates,
        key=lambda cg: (-cg[1]["min_sl_area"], -(cg[1]["area_tgt_sl"] + cg[1]["area_partner_sl"])),
    )
    out: List[CaseGeom] = []
    used_case: set[Tuple[str, str, str, str]] = set()
    subj_n: Dict[str, int] = {}

    def try_fill(use_subj_cap: bool) -> None:
        for case, g in sorted_c:
            if len(out) >= n:
                return
            key = (case.subject, case.crop, case.tag, case.structure)
            if key in used_case:
                continue
            if use_subj_cap and subj_n.get(case.subject, 0) >= max_per_subject:
                continue
            out.append((case, g))
            used_case.add(key)
            subj_n[case.subject] = subj_n.get(case.subject, 0) + 1

    try_fill(use_subj_cap=True)
    if len(out) < n:
        try_fill(use_subj_cap=False)
    return out


def _greedy_diverse(candidates: List[CaseGeom], pool_df: pd.DataFrame, n: int) -> List[CaseGeom]:
    """Pick up to ``n`` cases from ``candidates`` with diversity; no extra disk IO."""
    # Map case key -> row for (d_frac, r_val)
    key_to_dr: Dict[Tuple[str, str, str, str], Tuple[float, float]] = {}
    for r in pool_df.itertuples(index=False):
        k = (str(r.subject), str(r.crop), str(r.tag), str(r.structure))
        key_to_dr[k] = (float(r.d_frac), float(r.r_val))

    out: List[CaseGeom] = []
    used_case = set()
    used_subj = set()
    used_dr = set()
    used_struct = set()
    remaining = list(candidates)

    while len(out) < n and remaining:
        best_i = None
        best_score = None
        for i, (case, _g) in enumerate(remaining):
            key = (case.subject, case.crop, case.tag, case.structure)
            if key in used_case:
                continue
            dr = key_to_dr.get(key, (0.0, 0.0))
            score = (
                1 if case.subject not in used_subj else 0,
                1 if dr not in used_dr else 0,
                1 if case.structure not in used_struct else 0,
                abs(case.delta_pc),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        chosen_case, chosen_g = remaining.pop(best_i)
        key = (chosen_case.subject, chosen_case.crop, chosen_case.tag, chosen_case.structure)
        dr = key_to_dr.get(key, (0.0, 0.0))
        out.append((chosen_case, chosen_g))
        used_case.add(key)
        used_subj.add(chosen_case.subject)
        used_dr.add(dr)
        used_struct.add(chosen_case.structure)
    return out


def select_cases_slice_on(
    df,
    poset: PosetFromJson,
    n_good: int = 10,
    n_bad: int = 10,
) -> Tuple[List[CaseGeom], List[CaseGeom]]:
    base = df[_has_gt_series(df) & (df["vox_removed_pc"] > 0)].copy()
    good_pool = base[base["delta_pc"] > 0]
    bad_pool = base[base["delta_pc"] < 0]

    good_cand = _collect_passing_cases(good_pool, poset, ascending=False)
    bad_cand = _collect_passing_cases(bad_pool, poset, ascending=True)
    print(
        f"Slice-compact candidate rows: {len(good_cand)} good, {len(bad_cand)} bad "
        f"(≤{MAX_VOLUME_GROUPS_TO_SCAN} volumes scanned per polarity).",
        flush=True,
    )

    return (
        _greedy_diverse(good_cand, good_pool, n_good),
        _greedy_diverse(bad_cand, bad_pool, n_bad),
    )


def select_cases_large_slice_area(
    df,
    poset: PosetFromJson,
    n_good: int = 10,
    n_bad: int = 10,
) -> Tuple[List[CaseGeom], List[CaseGeom]]:
    base = df[_has_gt_series(df) & (df["vox_removed_pc"] > 0)].copy()
    good_pool = base[base["delta_pc"] > 0]
    bad_pool = base[base["delta_pc"] < 0]

    good_cand = _collect_large_area_cases(good_pool, poset, ascending=False)
    bad_cand = _collect_large_area_cases(bad_pool, poset, ascending=True)
    print(
        f"Large-slice candidate rows: {len(good_cand)} good, {len(bad_cand)} bad "
        f"(≤{MAX_VOLUME_GROUPS_LARGE_AREA} volumes per polarity).",
        flush=True,
    )
    return (
        _pick_top_diverse_by_min_slice_area(good_cand, n_good),
        _pick_top_diverse_by_min_slice_area(bad_cand, n_bad),
    )


def plot_slice_on_case(case: CaseRow, g: dict, out_path: Path, label: str) -> None:
    ap_ax = g["ap_ax"]
    lr_ax = g["lr_ax"]
    si_ax = g["si_ax"]
    sl = g["sl_cor"]
    mri_art = g["mri_art"]
    full_removed = g["full_removed"]
    partner_mask = g["partner_mask"]
    gt = g["gt"]
    evt = g["evt"]
    before = g["before"]
    after = g["after"]

    mri2d = extract_oriented_plane(mri_art, ap_ax, sl, lr_ax, si_ax)
    rem2d = extract_oriented_plane(full_removed, ap_ax, sl, lr_ax, si_ax)
    partner2d = extract_oriented_plane(partner_mask, ap_ax, sl, lr_ax, si_ax)
    tp2d = extract_oriented_plane(full_removed & gt, ap_ax, sl, lr_ax, si_ax)
    fp2d = extract_oriented_plane(full_removed & (~gt), ap_ax, sl, lr_ax, si_ax)

    d0 = dice(before, gt)
    d1 = dice(after, gt)
    p0 = precision(before, gt)
    p1 = precision(after, gt)

    cor_h, cor_w = rem2d.shape
    width_ratios = [cor_w, cor_w, int(2.4 * cor_w)]
    scale = 0.02
    fig_w = scale * sum(width_ratios)
    fig_h = scale * (2.1 * cor_h)

    fig, ax_row = plt.subplots(
        1,
        3,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios},
    )

    imshow_native(ax_row[0], overlay_before_cm4(mri2d, rem2d, partner2d))
    area_note = ""
    if "min_sl_area" in g:
        area_note = (
            f" · on-slice px: rem={g['area_tgt_sl']:,} partner={g['area_partner_sl']:,} "
            f"(min={g['min_sl_area']:,})"
        )
    title0 = (
        f"{label}\n{case.structure} | {case.subject} {case.crop} {case.tag}\n"
        f"MRI coronal AP={sl} · masks on **this slice only** "
        f"(silhouette/slice area = {g['ratio']:.2f}){area_note}"
    )
    ax_row[0].set_title(_wrap_to_width(title0, 44), fontsize=9)
    _set_pixel_true_aspect(ax_row[0], rem2d.shape)
    ax_row[0].axis("off")

    imshow_native(ax_row[1], overlay_after_tp_fp(mri2d, tp2d, fp2d))
    ax_row[1].set_title(
        _wrap_to_width(
            "after cleaning — removals on slice\nred/green = TP/FP (same slice)", 36
        ),
        fontsize=9,
    )
    _set_pixel_true_aspect(ax_row[1], rem2d.shape)
    ax_row[1].axis("off")

    ax_row[2].axis("off")
    sk = strip_cm4_from_figure_text(str(evt["step_kind"]))
    txt = (
        f"{label.upper()}  CSV ΔDice={case.delta_pc:+.5f}\n"
        f"Pair: {evt['pair_i']} / {evt['pair_j']}\n"
        f"Step: {sk}\n"
        f"Target (blue): {evt['target_name']}  Partner (purple): {g['partner_name']}\n"
        f"Removal voxels (full run): {int(full_removed.sum()):,}  "
        f"(TP {int((full_removed & gt).sum()):,}, FP {int((full_removed & ~gt).sum()):,})\n"
        f"Slice union px: {g['A_slice']:,}  Silhouette union px: {g['A_foot']:,}  "
        f"ratio={g['ratio']:.3f}\n"
    )
    if "min_sl_area" in g:
        txt += (
            f"On-slice px: removal={g['area_tgt_sl']:,}  partner={g['area_partner_sl']:,}  "
            f"min={g['min_sl_area']:,}\n"
        )
    txt += (
        f"\nDice: {d0:.4f} → {d1:.4f}  Precision: {p0:.4f} → {p1:.4f}\n"
        f"Caption event voxels: {evt['removed_voxels']:,}"
    )
    ax_row[2].text(0.02, 1.0, txt, va="top", ha="left", fontsize=10, family="monospace", linespacing=1.3)

    supt = (
        "Slice-only overlays (no A/P projection) — compact slice vs silhouette"
        if "min_sl_area" not in g
        else "Slice-only — maximize on-slice removal & partner footprint (both large)"
    )
    fig.suptitle(_wrap_to_width(supt, 90), fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    print("Loading CSV and poset…", flush=True)
    poset = load_poset_from_json(str(POSET_PATH))
    df = load_merged_or_partials(EVAL_CM4_T100)
    print("Selecting slice-compact cases (this scans up to N volumes per polarity)…", flush=True)
    good_cases, bad_cases = select_cases_slice_on(df, poset, n_good=10, n_bad=10)
    print(
        f"Selected {len(good_cases)} good, {len(bad_cases)} bad → rendering PNGs…",
        flush=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (case, g) in enumerate(good_cases, start=1):
        plot_slice_on_case(
            case,
            g,
            OUT_DIR / f"cm4_slice_on_good_{i:02d}.png",
            f"GOOD {i}",
        )
    for i, (case, g) in enumerate(bad_cases, start=1):
        plot_slice_on_case(
            case,
            g,
            OUT_DIR / f"cm4_slice_on_bad_{i:02d}.png",
            f"BAD {i}",
        )

    if len(good_cases) < 10 or len(bad_cases) < 10:
        print(
            f"WARNING: only got {len(good_cases)} good and {len(bad_cases)} bad cases. "
            f"Relax MAX_FOOTPRINT_TO_SLICE_RATIO, MIN_SLICE_*, or MAX_VOLUME_GROUPS_TO_SCAN."
        )


def main_large_area() -> None:
    print("Loading CSV and poset (large-area slice batch)…", flush=True)
    poset = load_poset_from_json(str(POSET_PATH))
    df = load_merged_or_partials(EVAL_CM4_T100)
    print("Selecting cases with largest min(removal, partner) on-slice…", flush=True)
    good_cases, bad_cases = select_cases_large_slice_area(df, poset, n_good=10, n_bad=10)
    print(
        f"Picked {len(good_cases)} good, {len(bad_cases)} bad → rendering PNGs…",
        flush=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, (case, g) in enumerate(good_cases, start=1):
        plot_slice_on_case(
            case,
            g,
            OUT_DIR / f"cm4_slice_large_good_{i:02d}.png",
            f"GOOD-L {i}",
        )
    for i, (case, g) in enumerate(bad_cases, start=1):
        plot_slice_on_case(
            case,
            g,
            OUT_DIR / f"cm4_slice_large_bad_{i:02d}.png",
            f"BAD-L {i}",
        )
    if len(good_cases) < 10 or len(bad_cases) < 10:
        print(
            f"WARNING: only got {len(good_cases)} good and {len(bad_cases)} bad. "
            f"Raise MAX_VOLUME_GROUPS_LARGE_AREA or lower MIN_LARGE_EACH_SLICE.",
            flush=True,
        )


if __name__ == "__main__":
    if "--large-area" in sys.argv:
        main_large_area()
    else:
        main()
