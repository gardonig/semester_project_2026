"""
Summarize cleaning results from per-subject CSV files.

Usage
-----
    python scripts/summarize_results.py data/results/mri_conservative/
    python scripts/summarize_results.py data/results/v201_cc/ --sort delta
    python scripts/summarize_results.py data/results/mri_conservative/ data/results/mri_aggressive/ --compare
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def load_dir(results_dir: Path) -> List[dict]:
    rows = []
    for f in sorted(results_dir.glob("*.csv")):
        with open(f, newline="") as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def summarize(rows: List[dict]) -> Dict[str, dict]:
    by_struct: Dict[str, list] = defaultdict(list)
    for r in rows:
        by_struct[r["structure"]].append(r)

    summary = {}
    for struct, entries in by_struct.items():
        deltas      = [float(e["delta"]) for e in entries]
        before      = [float(e["dice_before"]) for e in entries]
        after       = [float(e["dice_after"])  for e in entries]
        vox         = [int(e["voxels_removed"]) for e in entries]
        improved    = sum(1 for d in deltas if d >  0.0001)
        degraded    = sum(1 for d in deltas if d < -0.0001)
        unchanged   = len(deltas) - improved - degraded
        summary[struct] = {
            "n":            len(deltas),
            "mean_before":  sum(before) / len(before),
            "mean_after":   sum(after)  / len(after),
            "mean_delta":   sum(deltas) / len(deltas),
            "max_improve":  max(deltas),
            "max_degrade":  min(deltas),
            "improved":     improved,
            "degraded":     degraded,
            "unchanged":    unchanged,
            "total_vox_removed": sum(vox),
        }
    return summary


def print_summary(results_dir: Path, rows: List[dict], summary: Dict[str, dict], sort_by: str) -> None:
    subjects = len(set(r["subject"] for r in rows))
    total_improved  = sum(1 for r in rows if float(r["delta"]) >  0.0001)
    total_degraded  = sum(1 for r in rows if float(r["delta"]) < -0.0001)
    total_unchanged = len(rows) - total_improved - total_degraded
    mean_before = sum(float(r["dice_before"]) for r in rows) / len(rows)
    mean_after  = sum(float(r["dice_after"])  for r in rows) / len(rows)

    print(f"\n{'='*70}")
    print(f"  {results_dir.name}  —  {subjects} subjects  |  {len(rows):,} structure evaluations")
    print(f"{'='*70}")
    print(f"  Overall mean Dice:  before={mean_before:.4f}  after={mean_after:.4f}  Δ={mean_after-mean_before:+.6f}")
    print(f"  Improved: {total_improved}  |  Degraded: {total_degraded}  |  Unchanged: {total_unchanged}")

    key_map = {
        "delta":   lambda x: -x[1]["mean_delta"],
        "name":    lambda x:  x[0],
        "improved":lambda x: -x[1]["improved"],
        "voxels":  lambda x: -x[1]["total_vox_removed"],
    }
    sort_key = key_map.get(sort_by, key_map["delta"])
    sorted_structs = sorted(summary.items(), key=sort_key)

    changed = [(s, v) for s, v in sorted_structs if v["improved"] > 0 or v["degraded"] > 0]

    if not changed:
        print("\n  No structures changed.\n")
        return

    col = max(len(s) for s, _ in changed) + 2
    hdr = (f"\n  {'structure':<{col}} {'n':>5} {'mean_before':>11} {'mean_after':>10} "
           f"{'mean_Δ':>8} {'▲impr':>6} {'▼degr':>6} {'vox_removed':>12}")
    print(hdr)
    print(f"  {'-'*(col+63)}")

    for struct, v in changed:
        marker = "▲" if v["mean_delta"] > 0.0001 else ("▼" if v["mean_delta"] < -0.0001 else " ")
        print(f"  {struct:<{col}} {v['n']:>5} {v['mean_before']:>11.4f} {v['mean_after']:>10.4f} "
              f"  {v['mean_delta']:>+7.4f}{marker} {v['improved']:>6} {v['degraded']:>6} "
              f"{v['total_vox_removed']:>12,}")
    print()


def print_comparison(dir_a: Path, rows_a: List[dict], dir_b: Path, rows_b: List[dict]) -> None:
    sum_a = summarize(rows_a)
    sum_b = summarize(rows_b)
    all_structs = sorted(set(sum_a) | set(sum_b))
    changed = [s for s in all_structs
               if (sum_a.get(s, {}).get("improved", 0) or sum_a.get(s, {}).get("degraded", 0) or
                   sum_b.get(s, {}).get("improved", 0) or sum_b.get(s, {}).get("degraded", 0))]

    if not changed:
        print("\n  No structures changed in either run.\n")
        return

    col = max(len(s) for s in changed) + 2
    print(f"\n{'='*70}")
    print(f"  Comparison: {dir_a.name}  vs  {dir_b.name}")
    print(f"{'='*70}")
    hdr = f"\n  {'structure':<{col}} {'Δ '+dir_a.name:>14} {'Δ '+dir_b.name:>14} {'diff':>8}"
    print(hdr)
    print(f"  {'-'*(col+40)}")

    for s in changed:
        da = sum_a.get(s, {}).get("mean_delta", 0.0)
        db = sum_b.get(s, {}).get("mean_delta", 0.0)
        diff = db - da
        marker = "▲" if diff > 0.0001 else ("▼" if diff < -0.0001 else " ")
        print(f"  {s:<{col}} {da:>+14.4f} {db:>+14.4f} {diff:>+8.4f}{marker}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", type=Path, help="Results directories (one or two)")
    p.add_argument("--sort", default="delta",
                   choices=["delta", "name", "improved", "voxels"],
                   help="Sort structures by (default: mean_delta descending)")
    p.add_argument("--compare", action="store_true",
                   help="Side-by-side comparison of two result directories")
    args = p.parse_args()

    if args.compare:
        if len(args.dirs) != 2:
            print("--compare requires exactly two directories", file=sys.stderr)
            sys.exit(1)
        rows_a = load_dir(args.dirs[0])
        rows_b = load_dir(args.dirs[1])
        print_comparison(args.dirs[0], rows_a, args.dirs[1], rows_b)
    else:
        for d in args.dirs:
            rows = load_dir(d)
            if not rows:
                print(f"  [skip] no CSV files in {d}")
                continue
            print_summary(d, rows, summarize(rows), args.sort)


if __name__ == "__main__":
    main()
