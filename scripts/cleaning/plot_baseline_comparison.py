"""
Plot absolute Dice and Precision vs d and r, including the no-artifact (d=0) reference.

Produces:
  no_artifact_dice_prec_by_d.png   — mean Dice and Precision vs d, one line per r, + no-artifact reference
  no_artifact_dice_by_crop.png     — mean Dice vs d, one panel per crop, + no-artifact reference
  no_artifact_dice_by_r.png        — mean Dice vs r, one line per d, + no-artifact reference
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS_R = {0.25: "#4393c3", 0.50: "#92c5de", 0.75: "#f4a582", 1.00: "#d6604d"}
NO_ARTIFACT_COLOR = "#1a9641"
NO_ARTIFACT_LABEL = "no artifact (d=0)"


def load(results_with_no_artifact_csv: Path):
    df = pd.read_csv(results_with_no_artifact_csv)
    df = df[df.has_gt == True].copy()
    paired = df.dropna(subset=["dice_no_artifact", "prec_no_artifact"])
    print(f"  Total rows with GT: {len(df)}")
    print(f"  Rows with GT + no-artifact reference: {len(paired)}")
    print(f"  Subjects: {sorted(paired.subject.unique())}")
    return df, paired


def mean_dice_by_d(paired, r_val):
    return paired[paired.r_val == r_val].groupby("d_frac")["dice_before"].mean()


def mean_prec_by_d(paired, r_val):
    return paired[paired.r_val == r_val].groupby("d_frac")["precision_before"].mean()


def no_artifact_mean(paired, crop=None):
    sub = paired if crop is None else paired[paired.crop == crop]
    per_struct_d = sub.groupby(["crop", "structure"])["dice_no_artifact"].mean()
    per_struct_p = sub.groupby(["crop", "structure"])["prec_no_artifact"].mean()
    return per_struct_d.mean(), per_struct_p.mean()


def plot_dice_prec_by_d(paired, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    r_vals = sorted(paired.r_val.unique())

    for ax, metric, getter, ylabel in [
        (axes[0], "Dice",      mean_dice_by_d, "Mean Dice"),
        (axes[1], "Precision", mean_prec_by_d, "Mean Precision"),
    ]:
        na_d, na_p = no_artifact_mean(paired)
        na_val = na_d if metric == "Dice" else na_p
        ax.axhline(na_val, color=NO_ARTIFACT_COLOR, linewidth=2,
                   linestyle="--", label=f"{NO_ARTIFACT_LABEL} ({na_val:.3f})", zorder=5)

        for r in r_vals:
            series = getter(paired, r)
            ax.plot(series.index * 100, series.values,
                    marker="o", color=COLORS_R[r], label=f"r={r}", linewidth=1.8)

        ax.set_xlabel("Shift fraction d (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs d")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

    fig.suptitle("TotalSegmentator: artifact performance vs no-artifact reference\n"
                 "(same subject×structure subset, before poset-based cleaning)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_dice_by_crop(paired, out_path):
    crops = sorted(paired.crop.unique())
    r_vals = sorted(paired.r_val.unique())
    fig, axes = plt.subplots(1, len(crops), figsize=(5 * len(crops), 5), sharey=True)
    if len(crops) == 1:
        axes = [axes]

    for ax, crop in zip(axes, crops):
        na_d, _ = no_artifact_mean(paired, crop=crop)
        ax.axhline(na_d, color=NO_ARTIFACT_COLOR, linewidth=2,
                   linestyle="--", label=f"{NO_ARTIFACT_LABEL} ({na_d:.3f})", zorder=5)

        for r in r_vals:
            sub = paired[(paired.crop == crop) & (paired.r_val == r)]
            series = sub.groupby("d_frac")["dice_before"].mean()
            ax.plot(series.index * 100, series.values,
                    marker="o", color=COLORS_R[r], label=f"r={r}", linewidth=1.8)

        ax.set_title(crop.replace("_to_", " → "), fontsize=9)
        ax.set_xlabel("d (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_xticks([5, 10, 20, 30, 40, 50])

    axes[0].set_ylabel("Mean Dice")
    fig.suptitle("Mean Dice per crop vs d — artifact conditions vs no-artifact reference\n"
                 "(before poset-based cleaning, paired subject×structure subset)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_dice_by_r(paired, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    d_vals = sorted(paired.d_frac.unique())
    cmap = matplotlib.colormaps.get_cmap("Oranges").resampled(len(d_vals) + 2)

    na_d, _ = no_artifact_mean(paired)
    ax.axhline(na_d, color=NO_ARTIFACT_COLOR, linewidth=2,
               linestyle="--", label=f"{NO_ARTIFACT_LABEL} ({na_d:.3f})", zorder=5)

    for idx, d in enumerate(d_vals):
        series = paired[paired.d_frac == d].groupby("r_val")["dice_before"].mean()
        ax.plot(series.index, series.values,
                marker="o", color=cmap(idx + 2),
                label=f"d={int(d * 100)}%", linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Ghost intensity r")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Mean Dice vs r — artifact conditions vs no-artifact reference\n"
                 "(all crops, before poset-based cleaning, paired subject×structure subset)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", required=True, type=Path,
                   help="Threshold eval dir containing results_with_no_artifact.csv "
                        "(e.g. wraparound_v4_eval/t100)")
    p.add_argument("--out_dir",  default=None, type=Path,
                   help="Where to save plots (defaults to --eval_dir)")
    args = p.parse_args()

    merged_csv = args.eval_dir / "results_with_no_artifact.csv"
    out_dir    = args.out_dir or args.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, paired = load(merged_csv)

    plot_dice_prec_by_d(paired, out_dir / "no_artifact_dice_prec_by_d.png")
    plot_dice_by_crop(paired,   out_dir / "no_artifact_dice_by_crop.png")
    plot_dice_by_r(paired,      out_dir / "no_artifact_dice_by_r.png")


if __name__ == "__main__":
    main()
