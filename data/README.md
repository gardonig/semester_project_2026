# Data Directory

```
data/
├── datasets/          Source datasets (GT labels + CTs)
├── predictions/       TotalSegmentator inference outputs
├── results/           Dice evaluation CSVs (before/after cleaning)
├── posets/            Saved poset JSON files
├── structures/        Input CoM structure files
└── segmentations/     Precomputed bounding-box statistics
```

---

## datasets/

| Folder | Modality | Subjects | Structures | Contents |
|---|---|---|---|---|
| `amos22/labels/` | CT | 240 | 15 abdominal | Multi-label `.nii.gz` per subject |
| `flare22/labels/` | CT | 50 | 13 abdominal | Multi-label `.nii.gz` per subject |
| `totalseg_v201_small/data/` | CT | 102 | 117 | Per-structure masks + `meta.csv` (scanner, age, pathology) |
| `verse2020/labels/` | CT | 61 | Vertebrae | Per-subject `*_seg-vert_msk.nii.gz` + centroid JSONs |
| `verse2020/rawdata/` | CT | 61 | — | Raw CT scans (`*_ct.nii.gz`) — spine-focused (truncated FOV) |

**Cluster-only datasets** (on octopus03, not tracked locally):
- `totalseg_v201/` — 1,228 CT scans with 117 GT structures (TotalSegmentator benchmark)
- `totalseg_mri_v200/` — 616 MRI scans with 50 GT structures

---

## predictions/

TotalSegmentator inference outputs. Each folder contains per-subject subdirectories with one `.nii.gz` per predicted structure.

| Folder | Source dataset | TS version | Subjects |
|---|---|---|---|
| `amos_v157/` | AMOS22 | v1.5.7 | 240 |
| `amos_v157_cleaned/` | AMOS22 | v1.5.7 + CC cleaning | 14 |
| `flare22/` | FLARE22 | v1.5.7 | 50 |
| `v201_small/` | TotalSeg v201 small | v2.x | 102 |

---

## results/

Dice evaluation CSVs. Each CSV has columns: `subject, structure, dice_before, dice_after, delta, voxels_removed`.

| Folder | Dataset | Method | Subjects | Mean Δ Dice |
|---|---|---|---|---|
| `v201_cc/` | TotalSeg v201 (CT) | CC cleaning (conservative) | 109 | −0.000144 |
| `mri_conservative/` | TotalSeg MRI v200 | CC cleaning (conservative) | 553 | −0.000002 |
| `mri_aggressive/` | TotalSeg MRI v200 | CC cleaning (aggressive) | — | pending |

**Note**: Near-zero improvement on v201 and MRI is expected — TotalSegmentator was trained/benchmarked on these datasets. VerSe2020 (spine-only truncated CTs) is the intended validation target.

---

## posets/

Saved poset JSON files encoding anatomical spatial constraints.

| File | Description |
|---|---|
| `llm_sessions/llm_claude_v157.json` | Main poset used for all experiments (LLM-generated, 157 structures) |
| `merged_sessions/` | Multi-rater merged posets |
| `clinician_sessions/` | Human-annotated sessions |
| `tests/` | Development/test sessions |

---

## Citation

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for dataset citations.
