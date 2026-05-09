"""
Append missing rows from a partial evaluation CSV into the main results.csv,
then regenerate report.md and plots.

Usage
-----
    python scripts/cleaning/patch_missing_rows.py \
        --eval_dir data/experiments/wraparound_v4_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TAGS = {"t100": 1.00, "t099": 0.99, "t095": 0.95}


def patch_tag(tag_dir: Path, thresh: float) -> None:
    main_csv   = tag_dir / "results.csv"
    partial_csv = tag_dir / "partial_fix_s0022" / "results.csv"

    if not partial_csv.exists():
        print(f"[{tag_dir.name}] partial CSV not found, skipping")
        return
    if not main_csv.exists():
        print(f"[{tag_dir.name}] main results.csv not found, skipping")
        return

    df_main    = pd.read_csv(main_csv)
    df_partial = pd.read_csv(partial_csv)

    key = ["subject", "crop", "d_frac", "r_val", "structure"]
    existing = set(map(tuple, df_main[key].values.tolist()))
    new_rows  = df_partial[
        ~df_partial[key].apply(tuple, axis=1).isin(existing)
    ]

    if new_rows.empty:
        print(f"[{tag_dir.name}] no new rows to add")
        return

    print(f"[{tag_dir.name}] appending {len(new_rows)} new rows")
    df_merged = pd.concat([df_main, new_rows], ignore_index=True)
    df_merged.sort_values(["subject", "crop", "d_frac", "r_val", "structure"],
                          inplace=True)
    df_merged.to_csv(main_csv, index=False)
    print(f"[{tag_dir.name}] saved {len(df_merged)} total rows → {main_csv}")

    # Regenerate report and plots
    try:
        import sys
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "cleaning"))
        from evaluate_cleaning_methods import make_report, make_plots
        rows = df_merged.to_dict("records")
        make_plots(rows, tag_dir, poset_threshold=thresh)
        make_report(df_merged, tag_dir, n_total=len(df_merged),
                    has_prec=True, poset_threshold=thresh)
        print(f"[{tag_dir.name}] regenerated report and plots")
    except Exception as e:
        print(f"[{tag_dir.name}] could not regenerate report: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_dir", required=True, type=Path,
                   help="wraparound_v4_eval root containing t095/ t099/ t100/")
    args = p.parse_args()

    for tag, thresh in TAGS.items():
        tag_dir = args.eval_dir / tag
        if tag_dir.exists():
            patch_tag(tag_dir, thresh)
        else:
            print(f"[{tag}] directory not found, skipping")


if __name__ == "__main__":
    main()
