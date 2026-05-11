"""
Merge partial erosion-sweep results (one subdir per subject) into a single
combined CSV per radius variant, regenerate per-radius plots, and produce
the radius-sweep comparison plot.

Expected input structure (produced by erosion_sweep_array.sh):
  <base_dir>/
    partial_s0175/
      lcc_only/results.csv
      radius_1/results.csv
      radius_2/results.csv
      ...
    partial_s0236/
      lcc_only/results.csv
      ...

Output structure:
  <base_dir>/
    lcc_only/results.csv          (all subjects combined)
    lcc_only/report.md + plots
    radius_1/results.csv
    radius_1/report.md + plots
    ...
    radius_sweep.png              (comparison across all radii)

Usage
-----
    python scripts/cleaning/merge_erosion_sweep.py \\
        --base_dir data/experiments/wraparound_v4_eval/erosion_baseline
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_dir", type=Path,
                   default="data/experiments/wraparound_v4_eval/erosion_baseline",
                   help="Root dir produced by erosion_sweep_array.sh")
    args = p.parse_args()

    base_dir = args.base_dir

    # Import evaluation helpers from the baseline script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_erosion_baseline import make_plots, make_report, make_sweep_plots

    # ------------------------------------------------------------------
    # Discover which radius variants exist
    # ------------------------------------------------------------------
    partial_dirs = sorted(base_dir.glob("partial_*"))
    if not partial_dirs:
        print(f"No partial_* directories found in {base_dir}")
        return

    print(f"Found {len(partial_dirs)} partial directories: "
          f"{[d.name for d in partial_dirs]}")

    # Collect variant names from the first partial dir that has subdirs
    variants: list[str] = []
    for d in partial_dirs:
        variants = sorted(
            sub.name for sub in d.iterdir()
            if sub.is_dir() and (sub / "results.csv").exists()
        )
        if variants:
            break

    if not variants:
        print("No results.csv files found in any partial directory.")
        return

    print(f"Variants: {variants}")

    # ------------------------------------------------------------------
    # Merge per variant
    # ------------------------------------------------------------------
    # sweep_results is keyed by (method, radius_int) for make_sweep_plots()
    sweep_results: OrderedDict = OrderedDict()

    for variant in variants:
        # Parse variant name → method + radius
        if variant == "lcc_only":
            method, radius = "lcc_only", 0
        elif variant.startswith("radius_"):
            method  = "opening_lcc"
            radius  = int(variant.split("_")[1])
        else:
            print(f"  [skip] unrecognised variant: {variant}")
            continue

        # Collect rows from all partial dirs for this variant
        all_rows = []
        for pdir in partial_dirs:
            csv_path = pdir / variant / "results.csv"
            if not csv_path.exists():
                print(f"  [warn] missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            all_rows.extend(df.to_dict("records"))

        if not all_rows:
            print(f"  [skip] no rows for {variant}")
            continue

        n_subj = len({r["subject"] for r in all_rows})
        print(f"\n{variant}: {len(all_rows)} rows across {n_subj} subjects")

        # Save combined CSV
        out_dir = base_dir / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_out = out_dir / "results.csv"
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Saved combined CSV → {csv_out}")

        # Regenerate per-variant plots and report
        make_plots(all_rows, out_dir, method=method, radius=radius)
        make_report(all_rows, out_dir, method=method, radius=radius)

        sweep_results[(method, radius)] = all_rows

    # ------------------------------------------------------------------
    # Sweep comparison plot (all radii together)
    # ------------------------------------------------------------------
    if sweep_results:
        make_sweep_plots(sweep_results, base_dir)
        print(f"\nSweep plot saved to {base_dir / 'radius_sweep.png'}")
    else:
        print("No data to plot.")


if __name__ == "__main__":
    main()
