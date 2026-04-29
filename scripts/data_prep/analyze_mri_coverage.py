"""
Analyze structure coverage in TotalSegmentatorMRI dataset.

1. Reports the top subjects by number of GT structures present.
2. Finds the smallest set of subjects such that every structure
   appears in at least MIN_APPEARANCES of the selected subjects.
   Uses a greedy approach (NP-hard exact solution not needed in practice).

Usage:
    python scripts/data_prep/analyze_mri_coverage.py \
        --mri_dir ~/Desktop/ETH/semester_project/TotalsegmentatorMRI_dataset_v200 \
        --top 20

    # Cross-reference against our 117-structure list
    python scripts/data_prep/analyze_mri_coverage.py \
        --mri_dir ~/Desktop/ETH/semester_project/TotalsegmentatorMRI_dataset_v200 \
        --structures data/structures/totalseg_v2_com.json \
        --min_appearances 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


def scan_dataset(mri_dir: Path) -> dict[str, set[str]]:
    """Returns {subject_id: set of structure names present}."""
    coverage: dict[str, set[str]] = {}
    for subj_dir in sorted(mri_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        seg_dir = subj_dir / "segmentations"
        if not seg_dir.exists():
            # some datasets put masks directly in subject folder
            seg_dir = subj_dir
        masks = list(seg_dir.glob("*.nii.gz"))
        if not masks:
            continue
        structures = {p.name.replace(".nii.gz", "") for p in masks}
        coverage[subj_dir.name] = structures
    return coverage


def greedy_min_cover(
    coverage: dict[str, set[str]],
    target_structures: set[str],
    min_appearances: int,
) -> tuple[list[str], dict[str, int]]:
    """
    Greedy set cover: iteratively pick the subject that maximally increases
    the number of (structure, appearance) slots filled toward min_appearances.
    Stops when all target structures are covered or no improvement is possible.
    Returns (selected_subjects, final_appearance_counts).
    """
    counts: dict[str, int] = defaultdict(int)
    selected: list[str] = []
    remaining = set(target_structures)  # structures not yet at min_appearances

    while remaining:
        best_subj = None
        best_gain = 0
        for subj, structs in coverage.items():
            if subj in selected:
                continue
            # gain = number of remaining structures this subject would advance
            gain = sum(
                1 for s in structs
                if s in remaining and counts[s] < min_appearances
            )
            if gain > best_gain:
                best_gain = gain
                best_subj = subj

        if best_subj is None or best_gain == 0:
            break  # no more improvement possible

        selected.append(best_subj)
        for s in coverage[best_subj]:
            if s in target_structures:
                counts[s] += 1
        remaining = {s for s in remaining if counts[s] < min_appearances}

    return selected, dict(counts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mri_dir",         required=True, type=Path)
    p.add_argument("--structures",      default=None,  type=Path,
                   help="Path to com JSON to use as target structure list")
    p.add_argument("--top",             default=20,    type=int,
                   help="How many top subjects to report")
    p.add_argument("--min_appearances", default=3,     type=int,
                   help="Min times each structure must appear in selected set")
    args = p.parse_args()

    print(f"Scanning {args.mri_dir} ...")
    coverage = scan_dataset(args.mri_dir)
    print(f"Found {len(coverage)} subjects\n")

    # ------------------------------------------------------------------ #
    # 1. Top subjects by structure count
    # ------------------------------------------------------------------ #
    ranked = sorted(coverage.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"{'Rank':<6} {'Subject':<20} {'Structures':>10}")
    print("-" * 38)
    for i, (subj, structs) in enumerate(ranked[:args.top], 1):
        print(f"{i:<6} {subj:<20} {len(structs):>10}")

    all_structures = set(s for structs in coverage.values() for s in structs)
    print(f"\nTotal unique structures across dataset: {len(all_structures)}")

    # ------------------------------------------------------------------ #
    # 2. Greedy minimum cover
    # ------------------------------------------------------------------ #
    if args.structures:
        with open(args.structures) as f:
            com_data = json.load(f)
        target = {s["name"] for s in com_data["structures"]}
        print(f"\nTarget structure list: {len(target)} structures from {args.structures.name}")
    else:
        target = all_structures
        print(f"\nTarget: all {len(target)} structures found in dataset")

    in_dataset = target & all_structures
    missing_from_dataset = target - all_structures
    if missing_from_dataset:
        print(f"  {len(missing_from_dataset)} target structures NOT present in any MRI subject:")
        for s in sorted(missing_from_dataset):
            print(f"    - {s}")

    print(f"\nGreedy cover — min {args.min_appearances} appearances per structure...")
    selected, counts = greedy_min_cover(coverage, in_dataset, args.min_appearances)

    covered = {s for s in in_dataset if counts.get(s, 0) >= args.min_appearances}
    uncoverable = in_dataset - covered

    print(f"\nSelected {len(selected)} subjects to cover {len(covered)}/{len(in_dataset)} structures ≥{args.min_appearances}x:")
    for i, subj in enumerate(selected, 1):
        n = len(coverage[subj] & in_dataset)
        print(f"  {i:>3}. {subj:<20}  ({n} target structures)")

    if uncoverable:
        print(f"\n{len(uncoverable)} structures could not reach {args.min_appearances} appearances:")
        for s in sorted(uncoverable):
            print(f"  - {s}  (appears {counts.get(s, 0)}x)")


if __name__ == "__main__":
    main()
