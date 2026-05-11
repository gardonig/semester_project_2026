"""
Plot Dice and Precision vs shift fraction d for the morphological opening baseline.
Shows: no-artifact reference, raw prediction (before), LCC-only, opening r=1, opening r=2.
All lines averaged over ghost intensity r and all available subjects.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_NOART   = "#1a9641"
C_BEFORE  = "#555555"
C_LCC     = "#ff7f0e"
C_R1      = "#1f77b4"
C_R2      = "#d62728"

METHODS = [
    ("lcc_only",  "LCC only",      C_LCC,    "--"),
    ("radius_1",  "Opening r=1",   C_R1,     "-"),
    ("radius_2",  "Opening r=2",   C_R2,     "-"),
]


def load_erosion(base: Path) -> pd.DataFrame:
    dfs = []
    for subj_dir in sorted(base.glob("partial_s*")):
        for tag in ("lcc_only", "radius_1", "radius_2"):
            csv = subj_dir / tag / "results.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                df["method_tag"] = tag
                dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No results.csv found under {base}")
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged[merged.has_gt == True].copy()
    return merged


def get_noart(pc_csv: Path, subjects: list[str]) -> tuple[float, float]:
    df = pd.read_csv(pc_csv)
    df = df[(df.has_gt == True) & (df.subject.isin(subjects))].dropna(
        subset=["dice_no_artifact", "prec_no_artifact"]
    )
    return df["dice_no_artifact"].mean(), df["prec_no_artifact"].mean()


def make_plot(df: pd.DataFrame, before_col: str, after_col: str,
              noart_val: float, ylabel: str, out_path: Path) -> None:
    d_vals = sorted(df.d_frac.unique())
    d_pct  = [d * 100 for d in d_vals]

    fig, ax = plt.subplots(figsize=(8, 5))

    # No-artifact dot
    ax.scatter([0], [noart_val], color=C_NOART, zorder=6, s=80,
               marker="*", label=f"no artifact ({noart_val:.3f})")

    # Before (raw) line
    before_by_d = df.groupby("d_frac")[before_col].mean().reindex(d_vals)
    ax.plot([0] + d_pct, [noart_val] + list(before_by_d.values),
            color=C_BEFORE, lw=1.8, ls="-", marker="o", ms=4, label="before (raw)")

    # Erosion methods
    for tag, label, color, ls in METHODS:
        sub = df[df.method_tag == tag]
        after_by_d = sub.groupby("d_frac")[after_col].mean().reindex(d_vals)
        ax.plot([0] + d_pct, [noart_val] + list(after_by_d.values),
                color=color, lw=1.8, ls=ls, marker="s", ms=4, label=label)

    ax.set_xlabel("Shift fraction d (%)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{ylabel} by shift fraction d — morphological opening baseline\n"
                 f"(9 subjects, averaged over ghost intensity r)", fontsize=10)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50"])
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nSubjects: {sorted(df.subject.unique())}")
    print(f"Rows: {len(df)}")
    print(f"\n{'Method':<15} {'Dice before':>11} {'Dice after':>10} {'Delta Dice':>10} "
          f"{'Prec before':>11} {'Prec after':>10} {'Delta Prec':>10}")
    print("-" * 80)
    before_dice = df[df.method_tag == "lcc_only"]["dice_before"].mean()
    before_prec = df[df.method_tag == "lcc_only"]["precision_before"].mean()
    print(f"{'before (raw)':<15} {before_dice:>11.4f} {'—':>10} {'—':>10} "
          f"{before_prec:>11.4f} {'—':>10} {'—':>10}")
    for tag, label, _, _ in METHODS:
        sub = df[df.method_tag == tag]
        print(f"{label:<15} {sub.dice_before.mean():>11.4f} {sub.dice_erosion.mean():>10.4f} "
              f"{sub.delta_erosion.mean():>+10.4f} "
              f"{sub.precision_before.mean():>11.4f} {sub.precision_erosion.mean():>10.4f} "
              f"{sub.delta_prec_erosion.mean():>+10.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--erosion_dir", required=True, type=Path)
    p.add_argument("--pc_csv",      required=True, type=Path,
                   help="results_with_no_artifact.csv from poset cleaning eval")
    p.add_argument("--out_dir",     default=None,  type=Path)
    args = p.parse_args()

    out_dir = args.out_dir or args.erosion_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_erosion(args.erosion_dir)
    subjects = sorted(df.subject.unique())
    noart_dice, noart_prec = get_noart(args.pc_csv, subjects)
    print(f"No-artifact reference (subjects={subjects}): "
          f"dice={noart_dice:.4f}, prec={noart_prec:.4f}")
    print_summary(df)

    make_plot(df, "dice_before", "dice_erosion", noart_dice,
              "Mean Dice", out_dir / "dice_by_d_erosion_baseline.png")
    make_plot(df, "precision_before", "precision_erosion", noart_prec,
              "Mean Precision", out_dir / "prec_by_d_erosion_baseline.png")


if __name__ == "__main__":
    main()
