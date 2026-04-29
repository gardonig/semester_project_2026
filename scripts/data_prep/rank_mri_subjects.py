from pathlib import Path
import nibabel as nib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

MRI_DIR = Path("/Users/rabbit/Desktop/ETH/semester_project/Anatomy_Posets/data/datasets/TotalsegmentatorMRI_dataset_v200")

def count_nonempty(subj_dir):
    seg_dir = subj_dir / "segmentations"
    if not seg_dir.exists():
        return subj_dir.name, 0
    count = sum(1 for f in seg_dir.glob("*.nii.gz") if np.any(nib.load(str(f)).dataobj))
    return subj_dir.name, count

subjects = [d for d in sorted(MRI_DIR.iterdir()) if d.is_dir()]
results = []

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(count_nonempty, s): s for s in subjects}
    done = 0
    for fut in as_completed(futures):
        results.append(fut.result())
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(subjects)} done...")

results.sort(key=lambda x: -x[1])
print(f"\n{'Rank':<6} {'Subject':<12} {'Non-empty':>10}")
print("-" * 30)
for i, (name, count) in enumerate(results[:15], 1):
    print(f"{i:<6} {name:<12} {count:>10}")

print("\nTop 10:", [name for name, _ in results[:10]])
