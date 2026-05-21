"""
Two-panel ΔDice heatmap: morphological opening r=2 vs poset cleaning t=1.00.

Reads the same CSVs as the wraparound_v4_eval pipeline:
  - erosion:  .../erosion_baseline/radius_2/results.csv  (column delta_erosion)
  - poset:    .../t100/results.csv                        (column delta_pc)

Example:
    python scripts/cleaning/plot_heatmap_rad2_poset.py \\
        --erosion_csv data/wraparound_experiments/wraparound_v4_eval/erosion_baseline/radius_2/results.csv \\
        --poset_csv   data/wraparound_experiments/wraparound_v4_eval/t100/results.csv \\
        --out         data/wraparound_experiments/wraparound_v4_eval/heatmap_rad2_vs_poset_t1.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _mean_pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.groupby(["d_frac", "r_val"])[value_col].mean().unstack("r_val")


def _align_pivots(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = sorted(set(a.index) | set(b.index))
    cols = sorted(set(a.columns) | set(b.columns))
    return a.reindex(idx).reindex(columns=cols), b.reindex(idx).reindex(columns=cols)


def _plot_heatmap_panel(
    ax,
    pivot: pd.DataFrame,
    title: str,
    vmin: float,
    vmax: float,
):
    import numpy.ma as ma

    Z = ma.masked_invalid(pivot.values.astype(float))
    im = ax.imshow(Z, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"r={v}" for v in pivot.columns], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"d={v}" for v in pivot.index], fontsize=8)
    ax.set_title(title, fontsize=10)
    span = max(vmax, 1e-9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                ax.text(
                    j,
                    i,
                    f"{float(val):+.4f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black" if abs(float(val)) < 0.6 * span else "white",
                )
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="0.5")

    return im


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--erosion_csv",
        type=Path,
        default=PROJECT_ROOT
        / "data/wraparound_experiments/wraparound_v4_eval/erosion_baseline/radius_2/results.csv",
        help="Merged results for opening baseline at ball radius 2",
    )
    p.add_argument(
        "--poset_csv",
        type=Path,
        default=PROJECT_ROOT / "data/wraparound_experiments/wraparound_v4_eval/t100/results.csv",
        help="Merged poset eval at threshold t=1.00 (tag t100)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT
        / "data/wraparound_experiments/wraparound_v4_eval/heatmap_rad2_vs_poset_t1.png",
        help="Output PNG path",
    )
    args = p.parse_args()

    if not args.erosion_csv.is_file():
        print(f"Missing erosion CSV: {args.erosion_csv}", file=sys.stderr)
        sys.exit(1)
    if not args.poset_csv.is_file():
        print(f"Missing poset CSV: {args.poset_csv}", file=sys.stderr)
        sys.exit(1)

    df_e = pd.read_csv(args.erosion_csv)
    df_p = pd.read_csv(args.poset_csv)
    if "delta_erosion" not in df_e.columns:
        print("erosion CSV must contain column 'delta_erosion'", file=sys.stderr)
        sys.exit(1)
    if "delta_pc" not in df_p.columns:
        print("poset CSV must contain column 'delta_pc'", file=sys.stderr)
        sys.exit(1)

    pe = _mean_pivot(df_e, "delta_erosion")
    pp = _mean_pivot(df_p, "delta_pc")
    pe, pp = _align_pivots(pe, pp)

    abs_e = np.nanmax(np.abs(pe.values))
    abs_p = np.nanmax(np.abs(pp.values))
    vmax = float(max(abs_e, abs_p, 1e-6))
    vmin = -vmax

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_heatmap_panel(
        axes[0],
        pe,
        "Opening + LCC (r=2): mean Δ Dice per (d, r)",
        vmin,
        vmax,
    )
    im_last = _plot_heatmap_panel(
        axes[1],
        pp,
        "Poset cleaning (t=1.00): mean Δ Dice per (d, r)",
        vmin,
        vmax,
    )
    fig.colorbar(im_last, ax=axes, label="Mean Δ Dice", shrink=0.82, pad=0.02)
    fig.suptitle(
        "Mean Δ Dice (cleaned − artifact) — opening r=2 vs poset t=1.00\n"
        "Shared colour scale for direct comparison",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
