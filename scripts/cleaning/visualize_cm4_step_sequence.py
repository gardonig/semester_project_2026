"""
Plot full CM4 removal sequences for **good** and **bad** cases (greedy diversity over
the eval CSV). The real-modes grid script ``visualize_cm4_real_modes`` uses a separate
``visualize_cm4_real_modes`` (CLI ``--subject``; same policy as ``pick_cases_s0175_focus``).

Each panel shows one coronal **MRI slice**; removal overlays use an **A/P silhouette**
(full LR×SI footprint via logical OR along the coronal normal) so 3D extent is visible.

Outputs (default):
  results/cm4_real_cases/cm4_step_sequence_good.png
  results/cm4_real_cases/cm4_step_sequence_bad.png

Usage:
  python scripts/cleaning/visualize_cm4_step_sequence.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "cleaning"))

from anatomy_poset.core.io import load_poset_from_json, PosetFromJson

from evaluate_cleaning_methods import get_si_info

from visualize_cm4_real_modes import (
    CaseRow,
    EVAL_CM4_T100,
    POSET_PATH,
    THRESHOLD,
    best_slice,
    extract_oriented_footprint,
    extract_oriented_plane,
    get_ap_axis,
    get_lr_axis,
    load_artifact_mri,
    load_gt_crop,
    load_merged_or_partials,
    load_predictions,
    overlay_after_tp_fp,
    strip_cm4_from_figure_text,
    _has_gt_series,
    run_cm4_with_logs,
)
OUT_DIR = PROJECT_ROOT / "results" / "cm4_real_cases"
NCOLS_MAX = 8


def _wrap_to_width(text: str, width: int) -> str:
    """Hard-wrap to at most ``width`` characters per line (newlines preserved between chunks)."""
    text = text.strip()
    if not text:
        return text
    parts = []
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


def _normalize_mri(mri2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(mri2d.astype(float), [1, 99])
    return np.clip((mri2d.astype(float) - lo) / max(hi - lo, 1e-6), 0, 1)


def imshow_mri_only(ax, mri2d: np.ndarray) -> None:
    g = _normalize_mri(mri2d)
    rgb = np.stack([g, g, g], axis=-1)
    ax.imshow(rgb, origin="lower", interpolation="none", aspect="equal")


def plot_sequence_for_case(
    case: CaseRow, title_prefix: str, out_path: Path, poset: PosetFromJson
) -> None:
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

    _cleaned, _removed, events, snapshots = run_cm4_with_logs(
        preds,
        poset,
        si_ax,
        si_sign,
        THRESHOLD,
        gt_masks,
        collect_snapshots=True,
    )
    assert snapshots is not None
    if len(snapshots) != len(events) + 1:
        raise RuntimeError(
            f"snapshot/event mismatch: snapshots={len(snapshots)} events={len(events)}"
        )

    cum_rm = np.zeros(pred_shape, dtype=bool)
    for e in events:
        cum_rm |= e["removed_mask"]
    if cum_rm.any():
        sl_cor = best_slice(cum_rm, ap_ax)
    else:
        sl_cor = pred_shape[ap_ax] // 2

    mri2d = extract_oriented_plane(mri_art, ap_ax, sl_cor, x_axis=lr_ax, y_axis=si_ax)

    n_panels = 1 + len(events)
    ncols = min(NCOLS_MAX, max(1, n_panels))
    nrows = (n_panels + ncols - 1) // ncols

    fig_w = 2.55 * ncols
    fig_h = 3.05 * nrows + 0.6
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    # Each image column is ~2.55 in wide; keep titles within that footprint (~26 chars @ 8–9 pt).
    panel_title_width = 26
    suptitle_width = max(36, min(120, int(fig_w * 14)))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    # Panel 0 — initial
    ax0 = axes_flat[0]
    imshow_mri_only(ax0, mri2d)
    ax0.set_title(
        _wrap_to_width("0 · initial\n(artifact predictions)", panel_title_width),
        fontsize=9,
    )
    ax0.axis("off")

    for k in range(len(events)):
        evt = events[k]
        ax = axes_flat[1 + k]
        tgt = evt["target_name"]
        gt = gt_masks.get(tgt)
        if gt is None:
            gt = np.zeros(pred_shape, dtype=bool)
        rem = evt["removed_mask"]
        tp = rem & gt
        fp = rem & (~gt)
        tp2 = extract_oriented_footprint(tp, ap_ax, lr_ax, si_ax)
        fp2 = extract_oriented_footprint(fp, ap_ax, lr_ax, si_ax)

        # Partner footprint (purple) to show constraint structure silhouette as well.
        partner_name = evt["pair_j"] if tgt == evt["pair_i"] else evt["pair_i"]
        partner_mask = preds.get(partner_name, np.zeros(pred_shape, dtype=bool))
        partner2 = extract_oriented_footprint(partner_mask, ap_ax, lr_ax, si_ax)

        rgb_overlay = overlay_after_tp_fp(mri2d, tp2, fp2)
        p = partner2.astype(bool)
        # Same purple as in visualize_cm4_real_modes.overlay_before_cm4
        rgb_overlay[p, 0] = np.maximum(rgb_overlay[p, 0], 0.72)
        rgb_overlay[p, 1] = np.minimum(rgb_overlay[p, 1], 0.42)
        rgb_overlay[p, 2] = np.maximum(rgb_overlay[p, 2], 0.88)

        ax.imshow(rgb_overlay, origin="lower", interpolation="none", aspect="equal")
        step_title = (
            f"{k + 1} · {strip_cm4_from_figure_text(str(evt['step_kind']))} · {tgt} vs {partner_name} "
            f"(−{evt['removed_voxels']} vox)"
        )
        ax.set_title(_wrap_to_width(step_title, panel_title_width), fontsize=8)
        ax.axis("off")

    csv_bits = f"CSV ΔDice={case.delta_pc:+.4f}  ΔPrec={case.precision_pc - case.precision_before:+.4f}"
    suptitle_raw = (
        f"{title_prefix}\n"
        f"{case.subject}  {case.crop}  {case.tag}  |  {csv_bits}\n"
        f"MRI coronal AP={sl_cor} · removals = A/P silhouette  |  {len(events)} step(s)"
    )
    fig.suptitle(_wrap_to_width(suptitle_raw, suptitle_width), fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def select_cases_for_sequences(df, n_good: int = 10, n_bad: int = 10):
    base = df[_has_gt_series(df) & (df["vox_removed_pc"] > 0)].copy()

    def greedy(pool, n, ascending: bool):
        rows = list(pool.sort_values("delta_pc", ascending=ascending).itertuples(index=False))
        out = []
        used_case = set()         # (subject,crop,tag,structure)
        used_subj_crop = set()    # hard constraint: no repeated crop for same subject
        used_subj = set()
        used_dr = set()
        used_struct = set()

        while len(out) < n:
            best_idx = None
            best_score = None
            for idx, r in enumerate(rows):
                subj = str(r.subject)
                crop = str(r.crop)
                key = (subj, crop, str(r.tag), str(r.structure))
                sc = (subj, crop)
                if key in used_case or sc in used_subj_crop:
                    continue

                dr = (float(r.d_frac), float(r.r_val))
                struct = str(r.structure)
                # Higher is better: prioritize new subjects first, then new (d,r), then structure diversity.
                score = (
                    1 if subj not in used_subj else 0,
                    1 if dr not in used_dr else 0,
                    1 if struct not in used_struct else 0,
                    abs(float(r.delta_pc)),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            r = rows.pop(best_idx)
            subj = str(r.subject)
            crop = str(r.crop)
            dr = (float(r.d_frac), float(r.r_val))
            struct = str(r.structure)
            key = (subj, crop, str(r.tag), struct)

            out.append(
                CaseRow(
                    subject=subj,
                    crop=crop,
                    tag=str(r.tag),
                    structure=struct,
                    delta_pc=float(r.delta_pc),
                    precision_before=float(r.precision_before),
                    precision_pc=float(r.precision_pc),
                    vox_removed_pc=int(r.vox_removed_pc),
                )
            )
            used_case.add(key)
            used_subj_crop.add((subj, crop))
            used_subj.add(subj)
            used_dr.add(dr)
            used_struct.add(struct)
        return out

    good_pool = base[base["delta_pc"] > 0]
    bad_pool = base[base["delta_pc"] < 0]
    good_cases = greedy(good_pool, n_good, ascending=False)
    bad_cases = greedy(bad_pool, n_bad, ascending=True)
    return good_cases, bad_cases


def main() -> None:
    poset = load_poset_from_json(str(POSET_PATH))
    df = load_merged_or_partials(EVAL_CM4_T100)
    good_cases, bad_cases = select_cases_for_sequences(df, n_good=10, n_bad=10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, case in enumerate(good_cases, start=1):
        plot_sequence_for_case(
            case,
            f"GOOD case {idx} (ΔDice>0, diverse d/r/crop/subject)",
            OUT_DIR / f"cm4_step_sequence_good_{idx:02d}.png",
            poset,
        )
    for idx, case in enumerate(bad_cases, start=1):
        plot_sequence_for_case(
            case,
            f"BAD case {idx} (ΔDice<0, diverse d/r/crop/subject)",
            OUT_DIR / f"cm4_step_sequence_bad_{idx:02d}.png",
            poset,
        )


if __name__ == "__main__":
    main()
