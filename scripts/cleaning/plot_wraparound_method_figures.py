"""
Reproduce the wraparound_v4_eval summary figures (tradeoff, efficiency, adaptivity,
heatmaps) using merged CSVs — extended with Poset CM4.

Requires:
  - results_with_baseline.csv (prec_clean / dice_clean for recovery metrics)
  - Poset CM3: wraparound_v4_eval/t100/results.csv
  - Poset CM4: wraparound_v4_eval_cm4/t100/results.csv
  - Erosion partials under erosion_baseline/partial_*/{lcc_only,radius_1,radius_2}/results.csv

Outputs (default --out_dir):
  fig_tradeoff_scatter.png
  fig_efficiency_bar.png
  fig_adaptivity_by_d.png
  fig_heatmap_comparison.png
  fig_heatmap_poset_vs_opening.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

KEYS = ["subject", "crop", "tag", "structure"]


def merge_erosion_partials(erosion_root: Path, method_subdir: str) -> pd.DataFrame:
    paths = sorted(erosion_root.glob(f"partial_*/{method_subdir}/results.csv"))
    if not paths:
        raise FileNotFoundError(f"No partial results under {erosion_root}/*/ {method_subdir}/")
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def standardize_poset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_dice"] = out["delta_pc"]
    out["delta_prec"] = out["delta_prec_pc"]
    out["precision_after"] = out["precision_pc"]
    n_gt = (out["tp_before"] + out["fn_before"]).clip(lower=1)
    tp_after = out["tp_before"] - out["tp_removed_pc"]
    out["delta_recall"] = tp_after / n_gt - out["recall_before"]
    return out


def standardize_erosion(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_dice"] = out["delta_erosion"]
    out["delta_prec"] = out["delta_prec_erosion"]
    out["precision_after"] = out["precision_erosion"]
    n_gt = (out["tp_before"] + out["fn_before"]).clip(lower=1)
    vox_after = out["vox_before"] - out["vox_removed_erosion"]
    tp_after = out["precision_erosion"] * vox_after
    out["delta_recall"] = tp_after / n_gt - out["recall_before"]
    return out


def attach_reference(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Add prec_clean (and dice_clean) from results_with_baseline."""
    cols = KEYS + [c for c in ("prec_clean", "dice_clean") if c in baseline.columns]
    ref = baseline[cols].drop_duplicates(KEYS)
    return df.merge(ref, on=KEYS, how="left")


def filter_gt(df: pd.DataFrame) -> pd.DataFrame:
    hg = df["has_gt"]
    if hg.dtype != bool:
        m = hg.astype(str).str.lower().isin(("true", "1", "yes"))
    else:
        m = hg
    return df[m].copy()


def mean_delta(df: pd.DataFrame) -> Tuple[float, float]:
    d = filter_gt(df)
    return float(d["delta_dice"].mean()), float(d["delta_prec"].mean())


def mean_prec_recovery(df: pd.DataFrame) -> float:
    """Mean of (Prec_cleaned - Prec_art)/(Prec_ref - Prec_art) over rows with valid denominator."""
    d = filter_gt(df).copy()
    den = d["prec_clean"] - d["precision_before"]
    num = d["precision_after"] - d["precision_before"]
    valid = den.abs() > 1e-8
    if not valid.any():
        return float("nan")
    return float((num[valid] / den[valid]).mean() * 100.0)


def efficiency_ratio(df: pd.DataFrame) -> float:
    """ΔPrec_mean / |ΔDice|_mean — matches summary tables when denominators are stable."""
    d = filter_gt(df)
    ad = d["delta_dice"].abs().mean()
    if ad < 1e-12:
        return float("inf")
    return float(d["delta_prec"].mean() / ad)


def mean_by_d(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return d_frac sorted, mean ΔDice by d, mean ΔPrec by d (r-averaged)."""
    d = filter_gt(df)
    g = d.groupby("d_frac").agg(delta_dice=("delta_dice", "mean"), delta_prec=("delta_prec", "mean")).sort_index()
    return g.index.values, g["delta_dice"].values, g["delta_prec"].values


def heatmap_pivot(df: pd.DataFrame, metric: str = "delta_dice") -> pd.DataFrame:
    d = filter_gt(df)
    return d.groupby(["d_frac", "r_val"])[metric].mean().unstack("r_val")


def plot_tradeoff(points: Dict[str, Tuple[float, float]], out: Path, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"LCC only": "#ff7f0e", "Opening r=1": "#1f77b4", "Opening r=2": "#9467bd", "Poset": "#2ca02c"}
    markers = {"LCC only": "o", "Opening r=1": "s", "Opening r=2": "^", "Poset": "P"}
    for name, (dx, dy) in points.items():
        ax.scatter(dx, dy, s=120, label=name, color=colors.get(name, "#333333"), marker=markers.get(name, "o"), zorder=4)
        ax.annotate(name, (dx, dy), textcoords="offset points", xytext=(6, 6), fontsize=8)
    lim = max(max(abs(x) for x, _ in points.values()), max(abs(y) for _, y in points.values()), 0.02) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("Mean Δ Dice", fontsize=11)
    ax.set_ylabel("Mean Δ Precision", fontsize=11)
    ax.set_title(f"Trade-off: mean ΔDice × mean ΔPrec\n{title_suffix}", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_efficiency(
    methods: List[str],
    recovery_pct: List[float],
    eff_ratio: List[float],
    out: Path,
    title_suffix: str,
) -> None:
    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    c1 = "#4393c3"
    bars = ax1.bar(x - 0.2, recovery_pct, width=0.4, color=c1, alpha=0.85, label="Prec recovery (%)")
    ax1.set_ylabel("Precision recovery (%)", color=c1, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9, rotation=15, ha="right")
    ax1.axhline(0, color="gray", lw=0.6)

    ax2 = ax1.twinx()
    c2 = "#d6604d"
    capped = [min(float(r), 12.0) if np.isfinite(r) else 12.0 for r in eff_ratio]
    ax2.bar(x + 0.2, capped, width=0.4, color=c2, alpha=0.85, label="ΔPrec / mean|ΔDice| (cap 12)")
    ax2.set_ylabel("Efficiency ratio (ΔPrec / mean |ΔDice|)", color=c2, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=c2)
    ax2.set_ylim(0, max(capped + [0.1]) * 1.15)

    fig.suptitle(f"Precision recovery vs efficiency ratio — {title_suffix}", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_adaptivity(
    series: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out: Path,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    styles = {"LCC only": "-", "Opening r=1": "-", "Opening r=2": "-", "Poset": "-"}
    widths = {"Poset": 2.2}
    for name, (d_vals, ddice, dprec) in series.items():
        dp = [0] + [float(v) * 100 for v in d_vals]
        axes[0].plot(dp, [0] + list(ddice), styles[name], linewidth=widths.get(name, 1.4), label=name, marker="o", ms=3)
        axes[1].plot(dp, [0] + list(dprec), styles[name], linewidth=widths.get(name, 1.4), label=name, marker="s", ms=3)
    for ax in axes:
        ax.axhline(0, color="gray", lw=0.6)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes[0].set_ylabel("Mean Δ Dice (r-averaged)", fontsize=10)
    axes[1].set_ylabel("Mean Δ Precision (r-averaged)", fontsize=10)
    axes[1].set_xlabel("Shift fraction d (%)", fontsize=10)
    axes[1].set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    fig.suptitle(f"Adaptivity vs shift d — {title_suffix}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _annotate_heatmap(ax, pivot: pd.DataFrame, span: float) -> None:
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color="0.5")
            else:
                ax.text(
                    j, i, f"{float(val):+.4f}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(float(val)) < 0.6 * max(span, 1e-9) else "white",
                )


METRICS = [
    ("delta_dice",   "F1"),
    ("delta_prec",   "Precision"),
]


def plot_heatmap_metrics(
    df_opening: pd.DataFrame,
    df_poset: pd.DataFrame,
    out: Path,
    suptitle: str,
) -> None:
    """3-row × 2-column heatmap grid: rows = metrics, columns = methods.
    Each row shares a colorscale; colorbar is placed outside the panels."""
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(n_metrics, 2, figsize=(13, 5 * n_metrics),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.15})

    for row, (col, label) in enumerate(METRICS):
        po = heatmap_pivot(df_opening, col)
        pp = heatmap_pivot(df_poset, col)
        idx = sorted(set(po.index) | set(pp.index))
        cols_ = sorted(set(po.columns) | set(pp.columns))
        po = po.reindex(idx).reindex(columns=cols_)
        pp = pp.reindex(idx).reindex(columns=cols_)

        vmax = float(max(np.nanmax(np.abs(po.values)), np.nanmax(np.abs(pp.values)), 1e-6))

        for col_idx, (pivot, method_label) in enumerate([
            (po, "Opening r=2"),
            (pp, "Poset"),
        ]):
            ax = axes[row, col_idx]
            im = ax.imshow(pivot.values.astype(float), aspect="auto",
                           cmap="RdYlGn", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"r={v}" for v in pivot.columns], fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"d={v}" for v in pivot.index], fontsize=8)
            ax.set_title(f"{method_label}: mean Δ {label} per (d, r)", fontsize=10)
            _annotate_heatmap(ax, pivot, vmax)

        fig.colorbar(im, ax=axes[row, :].tolist(), label=f"Mean Δ {label}",
                     shrink=0.75, pad=0.03, fraction=0.046)

    fig.suptitle(suptitle, fontsize=12, y=1.01)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diff_heatmap(p_a: pd.DataFrame, p_b: pd.DataFrame, out: Path, title: str) -> None:
    """Align indices/columns and plot difference (a - b)."""
    common_r = sorted(set(p_a.columns) & set(p_b.columns))
    common_d = sorted(set(p_a.index) & set(p_b.index))
    a = p_a.reindex(index=common_d, columns=common_r)
    b = p_b.reindex(index=common_d, columns=common_r)
    diff = a - b
    vmax = max(abs(diff.values).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(diff.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(diff.columns)))
    ax.set_xticklabels([f"r={v}" for v in diff.columns], fontsize=8)
    ax.set_yticks(range(len(diff.index)))
    ax.set_yticklabels([f"d={v}" for v in diff.index], fontsize=7)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="Δ Mean ΔDice (left − right)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_root", type=Path, default=PROJECT_ROOT / "data/wraparound_experiments/wraparound_v4_eval")
    p.add_argument("--cm4_eval_root", type=Path, default=PROJECT_ROOT / "data/wraparound_experiments/wraparound_v4_eval_cm4")
    p.add_argument("--baseline_csv", type=Path, default=None)
    p.add_argument("--out_dir", type=Path, default=None)
    args = p.parse_args()

    eval_root = args.eval_root
    cm4_root = args.cm4_eval_root
    baseline_csv = args.baseline_csv or (eval_root / "results_with_baseline.csv")
    out_dir = args.out_dir or cm4_root
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(baseline_csv)

    erosion_root = eval_root / "erosion_baseline"
    df_lcc = attach_reference(standardize_erosion(merge_erosion_partials(erosion_root, "lcc_only")), baseline)
    df_r1 = attach_reference(standardize_erosion(merge_erosion_partials(erosion_root, "radius_1")), baseline)
    df_r2 = attach_reference(standardize_erosion(merge_erosion_partials(erosion_root, "radius_2")), baseline)

    cm4 = attach_reference(standardize_poset(pd.read_csv(cm4_root / "t100" / "results.csv")), baseline)

    methods = ["LCC only", "Opening r=1", "Opening r=2", "Poset"]
    dfs = {"LCC only": df_lcc, "Opening r=1": df_r1, "Opening r=2": df_r2, "Poset": cm4}

    points = {m: mean_delta(dfs[m]) for m in methods}
    plot_tradeoff(points, out_dir / "fig_tradeoff_scatter.png", "Poset CM4 vs baselines")

    rec = [mean_prec_recovery(dfs[m]) for m in methods]
    eff = [efficiency_ratio(dfs[m]) for m in methods]
    plot_efficiency(methods, rec, eff, out_dir / "fig_efficiency_bar.png", "Poset CM4 vs baselines")

    series = {}
    for m in methods:
        d_vals, a, b = mean_by_d(dfs[m])
        series[m] = (d_vals, a, b)
    plot_adaptivity(series, out_dir / "fig_adaptivity_by_d.png", "Poset CM4 vs baselines")

    plot_heatmap_metrics(
        df_r2,
        cm4,
        out_dir / "fig_heatmap_comparison.png",
        "Mean Δ metric (cleaned − artifact) — Opening r=2 vs Poset\nShared colour scale per metric row",
    )

    plot_diff_heatmap(
        heatmap_pivot(cm4),
        heatmap_pivot(df_r2),
        out_dir / "fig_heatmap_poset_vs_opening.png",
        "Mean ΔDice: CM4 − Opening r=2",
    )

    print(f"Saved figures to {out_dir}/")


if __name__ == "__main__":
    main()
