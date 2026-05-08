# Data Directory

```text
data/
├── datasets/        Source MRI volumes + GT segmentations — gitignored
├── experiments/     Artifact MRIs, segmentations, evaluation results — gitignored
├── posets/          Poset JSON files (empirical, clinician, LLM, merged)
└── structures/      Centre-of-mass atlases
```

---

## datasets/

**Cluster-only** (on octopus03, not tracked locally):

| Folder | Modality | Subjects | Structures | Notes |
| --- | --- | --- | --- | --- |
| `TotalsegmentatorMRI_dataset_v200/` | MRI | 616 | 50 | GT segmentations; 10 held out as test set |

---

## experiments/

Generated data — gitignored. Recreate with the scripts in `scripts/data_prep/` and `scripts/cleaning/`.

| Folder | Contents |
| --- | --- |
| `wraparound_v3/` | WM3 artifact MRIs and TotalSegmentator predictions (880 conditions × 10 subjects) |
| `wraparound_v3_eval/` | CM3 evaluation results at threshold = 0.95 |
| `wraparound_v3_eval_t100/` | CM3 evaluation results at threshold = 1.00 |

---

## posets/

Saved poset JSON files encoding anatomical spatial constraints.

| File / Folder | Description |
| --- | --- |
| `empirical/totalseg_mri_empirical_poset.json` | **Primary poset used for all experiments.** Empirical ordering of 50 structures from 606 MRI subjects (10 test subjects excluded). P ≥ 0.99 on 303 vertical-axis pairs. |
| `empirical/totalseg_v2_empirical_poset.json` | Legacy CT-derived empirical poset (117 structures, full TotalSegmentator v2.01 benchmark). Kept for reference. |
| `clinician_sessions/` | Human-annotated poset sessions |
| `llm_sessions/` | LLM-generated poset sessions |
| `merged_sessions/` | Multi-annotator merged probability posets |
| `tests/` | Development and test sessions |

---

## structures/

Centre-of-mass atlases. Neither is required by CM3 at inference time; both are available if a spatial prior is needed.

| File | Modality | Structures | Subjects | Normalisation |
| --- | --- | --- | --- | --- |
| `totalseg_mri_com_landmark.json` | MRI | 50 | test-set excluded | Vertebrae span (L5 bottom → C1 top = 0–100) |
| `totalseg_v2_com.json` | CT | 117 | TotalSeg v2.01 benchmark | Image extent (fraction of scan height) |

**MRI version (`totalseg_mri_com_landmark.json`)** — covers the same 50 structures TotalSegmentator MRI predicts. Landmark normalisation makes CoM values anatomically consistent across scans with different fields of view. Computed with test subjects excluded to prevent data leakage.

**CT version (`totalseg_v2_com.json`)** — covers all 117 CT label structures including individual vertebrae levels (C1–L5) not available in the MRI label set. Image-extent normalisation is less consistent across scans but the larger structure set may be useful for CT-specific work.

---

## Citation

See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for dataset citations.
