"""
Three-way comparison: raw predictions vs erosion baseline vs poset-based cleaning.

Produces a figure with two panels (Dice / Precision) showing mean metric vs
artifact shift fraction d.  Each panel has:
  - black dashed  : before cleaning (raw predictions)
  - blue solid    : erosion baseline  (opening_lcc at the given radius)
  - orange solid  : LCC-only baseline (if erosion CSV contains lcc_only rows)
  - red solid     : poset-based cleaning
  - green dashed  : no-artifact reference (constant horizontal line)

Usage
-----
    # Comparison using a single erosion radius CSV:
    python scripts/cleaning/plot_method_comparison.py \\
        --erosion_csv data/wraparound_experiments/wraparound_v4_eval/erosion_baseline/radius_2/results.csv \\
        --pc_csv      data/wraparound_experiments/wraparound_v4_eval/t100/results.csv \\
        --out_dir     data/wraparound_experiments/wraparound_v4_eval/erosion_baseline

    # If you also want to overlay the no-artifact reference line:
    python scripts/cleaning/plot_method_comparison.py \\
        --erosion_csv .../radius_2/results.csv \\
        --pc_csv      .../t100/results.csv \\
        --no_artifact_csv .../t100/results_with_no_artifact.csv \\
        --out_dir     .../erosion_baseline

    # To compare multiple erosion radii on the same figure pass a dir:
    python scripts/cleaning/plot_method_comparison.py \\
        --erosion_dir data/wraparound_experiments/wraparound_v4_eval/erosion_baseline \\
        --pc_csv      data/wraparound_experiments/wraparound_v4_eval/t100/results.csv \\
        --out_dir     data/wraparound_experiments/wraparound_v4_eval/erosion_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------
C_BEFORE    = "#555555"   # dark gray  — raw predictions
C_LCC       = "#ff7f0e"   # orange     — LCC only
C_EROSION   = "#1f77b4"   # blue       — opening_lcc
C_PC        = "#d62728"   # red        — poset-based cleaning
C_NOART     = "#1a9641"   # green      — no-artifact reference

CROP_LINESTYLES = {
    "brain_to_heart":  "-",
    "heart_to_kidney": "--",
    "kidney_to_hip":   ":",
    "overall":         "-",
}
CROP_LABELS = {
    "brain_to_heart":  "brain→heart",
    "heart_to_kidney": "heart→kidney",
    "kidney_to_hip":   "kidney→hip",
}
RADIUS_ALPHAS = {1: 0.40, 2: 0.70, 3: 1.00, 4: 0.70, 5: 0.40}   # highlight r=3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_by_d(df: pd.DataFrame, col: str) -> tuple[list[float], list[float]]:
    """Return (sorted d values, mean metric values) from df grouped by d_frac."""
    g = df.groupby("d_frac")[col].mean().sort_index()
    return list(g.index), list(g.values)


def mean_by_d_crop(df: pd.DataFrame, col: str, crop: str):
    sub = df[df["crop"] == crop]
    return mean_by_d(sub, col)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def make_comparison_figure(
    df_erosion: pd.DataFrame | None,
    df_lcc: pd.DataFrame | None,
    df_pc: pd.DataFrame,
    no_artifact_dice: float | None,
    no_artifact_prec: float | None,
    out_path: Path,
    title_suffix: str = "",
    per_crop: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, before_col, method_col_e, method_col_l, method_col_p, ylabel in [
        (axes[0],
         "dice_before", "dice_erosion", "dice_erosion", "dice_pc",
         "Mean Dice"),
        (axes[1],
         "precision_before", "precision_erosion", "precision_erosion", "precision_pc",
         "Mean Precision"),
    ]:
        d_raw, v_raw = mean_by_d(df_pc, before_col)
        ax.plot(d_raw, v_raw, color=C_BEFORE, linewidth=2.0, linestyle="--",
                marker="o", markersize=5, label="before cleaning", zorder=4)

        if df_lcc is not None and method_col_l in df_lcc.columns:
            d_l, v_l = mean_by_d(df_lcc, method_col_l)
            ax.plot(d_l, v_l, color=C_LCC, linewidth=1.8, linestyle="-",
                    marker="s", markersize=5, label="LCC only", zorder=3)

        if df_erosion is not None and method_col_e in df_erosion.columns:
            # Work out the radius from the data
            radius_val = int(df_erosion["radius"].iloc[0]) if "radius" in df_erosion.columns else "?"
            d_e, v_e = mean_by_d(df_erosion, method_col_e)
            ax.plot(d_e, v_e, color=C_EROSION, linewidth=1.8, linestyle="-",
                    marker="^", markersize=5, label=f"erosion baseline (r={radius_val})", zorder=3)

        d_pc, v_pc = mean_by_d(df_pc, method_col_p)
        ax.plot(d_pc, v_pc, color=C_PC, linewidth=2.0, linestyle="-",
                marker="D", markersize=5, label="poset-based cleaning", zorder=5)

        if no_artifact_dice is not None and ylabel == "Mean Dice":
            ax.axhline(no_artifact_dice, color=C_NOART, linewidth=1.5,
                       linestyle="--", label="no artifact (d=0)")
        if no_artifact_prec is not None and ylabel == "Mean Precision":
            ax.axhline(no_artifact_prec, color=C_NOART, linewidth=1.5,
                       linestyle="--", label="no artifact (d=0)")

        ax.set_xlabel("shift fraction d", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel + " vs artifact shift fraction" +
                     (f"\n{title_suffix}" if title_suffix else ""), fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def make_comparison_figure_by_r(
    df_erosion: pd.DataFrame | None,
    df_lcc: pd.DataFrame | None,
    df_pc: pd.DataFrame,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Repeat the comparison panel for each r value separately."""
    r_vals = sorted(df_pc["r_val"].unique())
    n = len(r_vals)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    for row_idx, r in enumerate(r_vals):
        pc_r = df_pc[df_pc["r_val"] == r]
        e_r  = df_erosion[df_erosion["r_val"] == r] if df_erosion is not None else None
        l_r  = df_lcc[df_lcc["r_val"] == r]         if df_lcc is not None else None

        for col_idx, (before_col, method_col_e, method_col_p, ylabel) in enumerate([
            ("dice_before",      "dice_erosion",      "dice_pc",      "Mean Dice"),
            ("precision_before", "precision_erosion", "precision_pc", "Mean Precision"),
        ]):
            ax = axes[row_idx][col_idx]

            d_raw, v_raw = mean_by_d(pc_r, before_col)
            ax.plot(d_raw, v_raw, color=C_BEFORE, linewidth=2.0, linestyle="--",
                    marker="o", markersize=4, label="before cleaning", zorder=4)

            if l_r is not None and method_col_e in (l_r.columns if l_r is not None else []):
                d_l, v_l = mean_by_d(l_r, method_col_e)
                ax.plot(d_l, v_l, color=C_LCC, linewidth=1.6, linestyle="-",
                        marker="s", markersize=4, label="LCC only")

            if e_r is not None and method_col_e in e_r.columns:
                radius_val = int(e_r["radius"].iloc[0]) if "radius" in e_r.columns else "?"
                d_e, v_e = mean_by_d(e_r, method_col_e)
                ax.plot(d_e, v_e, color=C_EROSION, linewidth=1.6, linestyle="-",
                        marker="^", markersize=4, label=f"erosion (r={radius_val})")

            d_pc, v_pc = mean_by_d(pc_r, method_col_p)
            ax.plot(d_pc, v_pc, color=C_PC, linewidth=2.0, linestyle="-",
                    marker="D", markersize=4, label="poset-based cleaning", zorder=5)

            ax.set_xlabel("d", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"r={r}  —  {ylabel}", fontsize=9)
            ax.legend(fontsize=7, loc="lower left")
            ax.set_ylim(bottom=0)

    plt.suptitle(f"Method comparison by ghost intensity r"
                 + (f"\n{title_suffix}" if title_suffix else ""), fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


def make_delta_comparison(
    df_erosion: pd.DataFrame | None,
    df_lcc: pd.DataFrame | None,
    df_pc: pd.DataFrame,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    """Show ΔDice and ΔPrecision vs d for both methods side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, e_col, l_col, p_col, ylabel in [
        (axes[0], "delta_erosion", "delta_erosion", "delta_pc",      "Mean Δ Dice"),
        (axes[1], "delta_prec_erosion", "delta_prec_erosion", "delta_prec_pc", "Mean Δ Precision"),
    ]:
        if df_lcc is not None and l_col in df_lcc.columns:
            d_l, v_l = mean_by_d(df_lcc, l_col)
            ax.plot(d_l, v_l, color=C_LCC, linewidth=1.8, linestyle="-",
                    marker="s", markersize=5, label="LCC only", zorder=3)

        if df_erosion is not None and e_col in df_erosion.columns:
            radius_val = int(df_erosion["radius"].iloc[0]) if "radius" in df_erosion.columns else "?"
            d_e, v_e = mean_by_d(df_erosion, e_col)
            ax.plot(d_e, v_e, color=C_EROSION, linewidth=1.8, linestyle="-",
                    marker="^", markersize=5, label=f"erosion baseline (r={radius_val})", zorder=3)

        if p_col in df_pc.columns:
            d_pc, v_pc = mean_by_d(df_pc, p_col)
            ax.plot(d_pc, v_pc, color=C_PC, linewidth=2.0, linestyle="-",
                    marker="D", markersize=5, label="poset-based cleaning", zorder=5)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("shift fraction d", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel + " vs artifact shift fraction" +
                     (f"\n{title_suffix}" if title_suffix else ""), fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--erosion_csv", type=Path, default=None,
                   help="results.csv from a single erosion radius variant")
    p.add_argument("--lcc_csv", type=Path, default=None,
                   help="results.csv from the lcc_only variant")
    p.add_argument("--erosion_dir", type=Path, default=None,
                   help="Base dir containing radius_N/ subdirs — uses the best "
                        "radius (highest mean ΔDice) automatically")
    p.add_argument("--pc_csv", type=Path, required=True,
                   help="results.csv from poset-based cleaning (dice_pc column)")
    p.add_argument("--no_artifact_csv", type=Path, default=None,
                   help="results_with_no_artifact.csv — used to draw the "
                        "no-artifact reference line")
    p.add_argument("--out_dir", type=Path,
                   default=Path("data/wraparound_experiments/wraparound_v4_eval/erosion_baseline"),
                   help="Directory to write comparison PNGs into")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load poset-based cleaning results
    # ------------------------------------------------------------------
    df_pc = pd.read_csv(args.pc_csv)
    print(f"Loaded PC results: {len(df_pc)} rows  subjects={df_pc['subject'].nunique()}")

    # ------------------------------------------------------------------
    # Load erosion results
    # ------------------------------------------------------------------
    df_erosion: pd.DataFrame | None = None
    df_lcc: pd.DataFrame | None = None

    if args.erosion_dir is not None:
        # Auto-select best radius by mean ΔDice across all conditions
        best_radius, best_delta = None, -999
        for r_dir in sorted(args.erosion_dir.glob("radius_*")):
            csv_path = r_dir / "results.csv"
            if not csv_path.exists():
                continue
            df_tmp = pd.read_csv(csv_path)
            mean_d = df_tmp["delta_erosion"].mean()
            r_int = int(r_dir.name.split("_")[1])
            print(f"  radius={r_int}  mean ΔDice={mean_d:+.5f}")
            if mean_d > best_delta:
                best_delta, best_radius, df_erosion = mean_d, r_int, df_tmp
        if df_erosion is not None:
            print(f"Auto-selected best erosion radius: r={best_radius}  (mean ΔDice={best_delta:+.5f})")

        lcc_path = args.erosion_dir / "lcc_only" / "results.csv"
        if lcc_path.exists():
            df_lcc = pd.read_csv(lcc_path)
            print(f"Loaded LCC-only results: {len(df_lcc)} rows")

    elif args.erosion_csv is not None:
        df_erosion = pd.read_csv(args.erosion_csv)
        print(f"Loaded erosion results: {len(df_erosion)} rows")

    if args.lcc_csv is not None:
        df_lcc = pd.read_csv(args.lcc_csv)
        print(f"Loaded LCC-only results: {len(df_lcc)} rows")

    # ------------------------------------------------------------------
    # No-artifact reference values
    # ------------------------------------------------------------------
    no_art_dice = no_art_prec = None
    if args.no_artifact_csv is not None and args.no_artifact_csv.exists():
        df_na = pd.read_csv(args.no_artifact_csv)
        if "dice_no_artifact" in df_na.columns:
            no_art_dice = df_na["dice_no_artifact"].mean()
            no_art_prec = df_na["prec_no_artifact"].mean() if "prec_no_artifact" in df_na.columns else None
            print(f"No-artifact reference: dice={no_art_dice:.4f}  prec={no_art_prec}")

    # ------------------------------------------------------------------
    # Title suffix — list subjects
    # ------------------------------------------------------------------
    n_subj_pc = df_pc["subject"].nunique()
    n_subj_e  = df_erosion["subject"].nunique() if df_erosion is not None else 0
    title = f"{max(n_subj_pc, n_subj_e)} subjects"

    # ------------------------------------------------------------------
    # Produce plots
    # ------------------------------------------------------------------
    make_comparison_figure(
        df_erosion, df_lcc, df_pc,
        no_art_dice, no_art_prec,
        args.out_dir / "method_comparison_dice_prec.png",
        title_suffix=title,
    )

    make_delta_comparison(
        df_erosion, df_lcc, df_pc,
        args.out_dir / "method_comparison_delta.png",
        title_suffix=title,
    )

    make_comparison_figure_by_r(
        df_erosion, df_lcc, df_pc,
        args.out_dir / "method_comparison_by_r.png",
        title_suffix=title,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
