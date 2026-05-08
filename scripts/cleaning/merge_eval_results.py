"""
Merge partial per-subject evaluation CSVs into one results.csv and regenerate plots.

Usage:
    python scripts/cleaning/merge_eval_results.py \
        --eval_base /scratch/gardonig/wraparound_v4_eval \
        --tags t095 t099 t100
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def merge_tag(base: Path, tag: str) -> None:
    tag_dir = base / tag
    partials = sorted(tag_dir.glob("partial_*/results.csv"))
    if not partials:
        print(f"[{tag}] no partial CSVs found — skipping")
        return

    dfs = [pd.read_csv(p) for p in partials]
    df = pd.concat(dfs, ignore_index=True)
    out_csv = tag_dir / "results.csv"
    df.to_csv(out_csv, index=False)
    print(f"[{tag}] merged {len(partials)} subjects → {len(df)} rows → {out_csv}")

    # Regenerate plots + report
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "cleaning"))
    from evaluate_cleaning_methods import make_plots
    threshold_map = {"t095": 0.95, "t099": 0.99, "t100": 1.00}
    thresh = threshold_map.get(tag, 0.95)
    make_plots(df.to_dict("records"), tag_dir, poset_threshold=thresh)
    print(f"[{tag}] plots + report written to {tag_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval_base", required=True, type=Path)
    p.add_argument("--tags", nargs="+", default=["t095", "t099", "t100"])
    args = p.parse_args()

    for tag in args.tags:
        merge_tag(args.eval_base, tag)


if __name__ == "__main__":
    main()
