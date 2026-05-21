"""
Plot Dice and Precision vs artifact shift fraction d, before and after cleaning.

Produces exactly two plots:
  dice_by_d_before_and_after_cleaning.png
  prec_by_d_before_and_after_cleaning.png

Each plot shows one line per ghost intensity r for both before and after cleaning,
plus a single shared dot at d=0 for the no-artifact reference (one value, not per-r).

CM3-style merged CSVs include dice_no_artifact / prec_no_artifact. CM4 `results.csv`
does not — pass `--ref_csv` pointing at CM3 `results_with_no_artifact.csv` (same grid)
to attach reference columns via merge on subject, crop, tag, structure.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS_R = {0.25: "#4393c3", 0.50: "#2166ac", 0.75: "#f4a582", 1.00: "#d6604d"}
NO_ARTIFACT_COLOR = "#1a9641"

MERGE_KEYS = ("subject", "crop", "tag", "structure")
REF_COLS = ("dice_no_artifact", "prec_no_artifact")


def _has_gt_mask(df: pd.DataFrame) -> pd.Series:
    hg = df["has_gt"]
    if hg.dtype == bool:
        return hg
    return hg.astype(str).str.lower().isin(("true", "1", "yes"))


def load(csv_path: Path, ref_csv: Optional[Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[_has_gt_mask(df)].copy()

    missing_ref = not all(c in df.columns for c in REF_COLS)
    if missing_ref:
        if ref_csv is None:
            raise SystemExit(
                f"{csv_path} lacks {REF_COLS}; pass --ref_csv "
                "(e.g. data/wraparound_experiments/wraparound_v4_eval/t100/results_with_no_artifact.csv)"
            )
        ref = pd.read_csv(ref_csv)
        miss = [c for c in MERGE_KEYS + REF_COLS if c not in ref.columns]
        if miss:
            raise SystemExit(f"ref CSV missing columns: {miss}")
        ref = ref[list(MERGE_KEYS + REF_COLS)].drop_duplicates(subset=list(MERGE_KEYS))
        df = df.merge(ref, on=list(MERGE_KEYS), how="left")

    df = df.dropna(subset=list(REF_COLS))
    print(f"  Rows: {len(df)}  subjects: {sorted(df.subject.unique())}")
    return df


def make_plot(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    no_art_col: str,
    ylabel: str,
    out_path: Path,
    cleaning_label: str,
) -> None:
    r_vals = sorted(df.r_val.unique())
    fig, ax = plt.subplots(figsize=(8, 5))

    # Single shared no-artifact point — one value averaged across all structures/subjects
    # (computed per structure first to avoid subjects with more structures dominating)
    no_art_val = (df.groupby(["subject", "crop", "structure"])[no_art_col]
                  .mean().mean())

    for r in r_vals:
        sub = df[df.r_val == r]
        color = COLORS_R[r]

        before = sub.groupby("d_frac")[before_col].mean().sort_index()
        after  = sub.groupby("d_frac")[after_col].mean().sort_index()

        d_pct = [d * 100 for d in before.index]

        ax.plot([0] + d_pct, [no_art_val] + list(before.values),
                color=color, linewidth=1.8, linestyle="-",
                marker="o", markersize=4, label=f"r={r}, before")
        ax.plot([0] + d_pct, [no_art_val] + list(after.values),
                color=color, linewidth=1.8, linestyle="--",
                marker="s", markersize=4, label=f"r={r}, after cleaning")

    # Single no-artifact dot (shared origin for all lines)
    ax.scatter([0], [no_art_val], color=NO_ARTIFACT_COLOR, zorder=6,
               s=80, marker="*", label=f"no artifact ({no_art_val:.3f})")

    ax.set_xlabel("Shift fraction d (%)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"{ylabel} before and after {cleaning_label}\nvs artifact shift fraction d  (d=0: no artifact)",
        fontsize=10,
    )
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xticklabels(["0", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50"])
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval_dir", required=True, type=Path,
                   help="Eval dir (e.g. .../wraparound_v4_eval_cm4/t100)")
    p.add_argument(
        "--input_csv",
        default=None,
        type=Path,
        help="Results CSV (default: results_with_no_artifact.csv if present, else results.csv)",
    )
    p.add_argument(
        "--ref_csv",
        default=None,
        type=Path,
        help="Reference CSV with dice_no_artifact & prec_no_artifact (required for CM4 results.csv)",
    )
    p.add_argument(
        "--cleaning_label",
        default="poset-based cleaning",
        help="Short phrase for plot titles (e.g. 'CM4 (center-conflict) cleaning')",
    )
    p.add_argument("--out_dir", default=None, type=Path,
                   help="Where to save plots (defaults to --eval_dir)")
    args = p.parse_args()

    out_dir = args.out_dir or args.eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_csv is not None:
        csv_path = args.input_csv
    else:
        cand_a = args.eval_dir / "results_with_no_artifact.csv"
        cand_b = args.eval_dir / "results.csv"
        if cand_a.exists():
            csv_path = cand_a
        elif cand_b.exists():
            csv_path = cand_b
        else:
            raise SystemExit(f"No {cand_a.name} or {cand_b.name} under {args.eval_dir}")

    ref_csv = args.ref_csv
    if ref_csv is None and csv_path.name == "results.csv":
        guess = (
            args.eval_dir.parent.parent / "wraparound_v4_eval" / "t100" / "results_with_no_artifact.csv"
        )
        if guess.exists():
            ref_csv = guess
            print(f"  Using default ref_csv: {ref_csv}")

    df = load(csv_path, ref_csv)

    make_plot(
        df,
        before_col="dice_before",
        after_col="dice_pc",
        no_art_col="dice_no_artifact",
        ylabel="Mean F1",
        out_path=out_dir / "f1_by_d_before_and_after_cleaning.png",
        cleaning_label=args.cleaning_label,
    )

    make_plot(
        df,
        before_col="precision_before",
        after_col="precision_pc",
        no_art_col="prec_no_artifact",
        ylabel="Mean Precision",
        out_path=out_dir / "prec_by_d_before_and_after_cleaning.png",
        cleaning_label=args.cleaning_label,
    )


if __name__ == "__main__":
    main()
