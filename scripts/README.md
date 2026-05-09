# Scripts

Organised by task — pick the folder for the work you want to do.

---

## `cleaning/` — Segmentation constraint post-processing

| Script | Purpose |
| --- | --- |
| `poset_constraint_postprocessing.py` | Core cleaning pipeline: applies poset constraints via CC analysis, outputs cleaned masks |
| `evaluate_cleaning_methods.py` | Evaluate poset-based cleaning on all 10 subjects × WM3 artifact conditions; outputs Dice/Precision CSV, plots, and report |
| `save_cleaned_segmentations.py` | Run poset-based cleaning on a single artifact condition and write cleaned NIfTI files to disk for 3D Slicer |
| `summarize_results.py` | Aggregate per-subject CSVs into per-structure summary tables |
| `visualize_cleaning.py` | Axial-slice visualisation of before/after cleaning (green=kept, red=removed, blue=GT) |
| `plot_dice_by_d_r.py` | Plot ΔDice/ΔPrecision heatmaps and bar charts from a results CSV |
| `segment_artifacts_array.sh` | SLURM array: runs TotalSegmentator on one artifact per task (skips existing) |
| `clean_artifacts_array.sh` | SLURM array: runs poset-based cleaning per artifact condition |
| `clean_all_artifacts_batch.sh` | SLURM: runs poset-based cleaning for all subjects in one batch job |
| `evaluate_v3_batch.sh` | SLURM: full 10-subject evaluation at a given threshold |
| `segment_one_subject.sh` | SLURM job: TotalSegmentator for all artifacts of one subject |
| `submit_segmentation_jobs.sh` | Submits one `segment_one_subject.sh` job per unsegmented subject |

---

### Cleaning methods

**Preliminary method M1 — Unidirectional (internal development variant, not published)**
For each constraint pair `(i above j)`, finds the LCC of `j` and removes any non-LCC component of `i` whose S-I extent lies entirely below `j`'s LCC inferior boundary. Handles superior-anatomy ghosts at the inferior crop edge but always trusts the LCC even when it is the ghost.

**Preliminary method M2 — Symmetric (internal development variant, not published)**
Extends M1 in both directions: also removes non-LCC components of `j` that sit entirely above `i`'s LCC superior boundary. Catches both wrap directions but accumulates errors — an incorrectly cleaned anchor misleads subsequent pairs.

**Poset-based cleaning — Middle-out + constraint-consistency (published method)**
Processes pairs in middle-out order (pairs closest to image centre first). Two key design decisions:

- **Constraint-consistency guard:** for pair `(i above j)`, if i's LCC is entirely below j's LCC, the constraint is already violated. The pair ordering itself encodes the CoM prior — i should be superior — so i's LCC is identified as the ghost. All i-components below j's inferior boundary are removed with `protect_anchor=False` (the ghost LCC is not spared). Any real fragment of i already above j is preserved and becomes the new anchor.
- **Middle-out ordering:** pairs whose combined observed LCC midpoint is closest to the image centre are processed first. Already-cleaned central anchors propagate outward to inform peripheral structure cleaning.

Poset-based cleaning requires only the predicted masks, the poset, and the image orientation — no atlas, no crop coordinates.

---

### Evaluation metrics — why precision, and why we keep Dice

**Why precision is the primary metric.**
Poset-based cleaning is purely subtractive — it can only remove voxels, never add them. Recall = TP / (TP + FN) is therefore mathematically invariant to cleaning. Precision = TP / (TP + FP) is sensitive only to the dimension poset-based cleaning actually operates on (FP reduction) and is the metric that most faithfully reflects cleaning quality.

**Why Dice is still reported.**
Dice is the de-facto standard for segmentation evaluation and enables direct comparison with published TotalSegmentator benchmarks. It also makes trade-offs visible: a method could improve precision while marginally degrading Dice by removing a few TPs, and both facts matter.

---

### Statistical evaluation — full results (threshold = 0.95)

*Dataset: 10 subjects × 3 crops × 40 artifact conditions (10 d × 4 r) = 1 200 (subject, crop, condition) tuples. Wilcoxon signed-rank test on non-zero differences; per-structure p-values corrected with Benjamini–Hochberg FDR (α = 0.05).*

#### Overall

| Metric | Mean Δ | n nonzero | p-value | Result |
| --- | --- | --- | --- | --- |
| Dice | −0.00089 | 1327 / 12673 | 8.7 × 10⁻⁴⁸ | significant **degradation** |
| Precision | +0.00166 | 1360 / 12673 | 3.1 × 10⁻¹⁰¹ | significant **improvement** |

The opposite directions confirm the metric argument above: poset-based cleaning systematically reduces FP at the cost of a very small Dice penalty driven by a handful of structures.

#### By crop region

| Crop | Metric | Mean Δ | p-value | Result |
| --- | --- | --- | --- | --- |
| brain → heart | Dice | +0.00025 | 7.9 × 10⁻⁷ | improvement *** |
| brain → heart | Precision | +0.00046 | 7.9 × 10⁻⁷ | improvement *** |
| heart → kidney | Dice | −0.00041 | 3.5 × 10⁻²⁹ | degradation *** |
| heart → kidney | Precision | +0.00099 | 6.7 × 10⁻³⁹ | improvement *** |
| kidney → hip | Dice | −0.00143 | 1.2 × 10⁻¹⁸ | degradation *** |
| kidney → hip | Precision | +0.00235 | 1.1 × 10⁻⁵⁹ | improvement *** |

Precision improves in all three crops. Dice degrades in the two lower-body crops, driven by `small_bowel` and the iliac vessels (see per-structure table).

#### By ghost intensity (r)

| r | Metric | Mean Δ | p-value | Result |
| --- | --- | --- | --- | --- |
| 0.25 | Dice | −0.00166 | 2.9 × 10⁻⁶ | degradation *** |
| 0.25 | Precision | −0.00086 | 1.2 × 10⁻¹⁶ | degradation *** |
| 0.50 | Dice | −0.00096 | 1.3 × 10⁻¹⁶ | degradation *** |
| 0.50 | Precision | +0.00162 | 3.8 × 10⁻³¹ | improvement *** |
| 0.75 | Dice | −0.00069 | 8.6 × 10⁻¹⁸ | degradation *** |
| 0.75 | Precision | +0.00267 | 1.6 × 10⁻³¹ | improvement *** |
| 1.00 | Dice | −0.00017 | 3.7 × 10⁻¹⁴ | degradation *** |
| 1.00 | Precision | +0.00350 | 3.0 × 10⁻²⁸ | improvement *** |

At r = 0.25 (weakest ghost) both metrics degrade — the ghost is too faint to distinguish from real anatomy and poset-based cleaning over-removes. From r = 0.5 onward precision improves consistently and strongly.

#### By shift fraction (d)

| d | Metric | Mean Δ | p-value | Result |
| --- | --- | --- | --- | --- |
| 0.05 | Dice | −0.00183 | 8.3 × 10⁻⁴ | degradation *** |
| 0.05 | Precision | +0.00018 | 1.7 × 10⁻¹⁰ | improvement *** |
| 0.10 | Dice | −0.00147 | 4.5 × 10⁻⁶ | degradation *** |
| 0.10 | Precision | +0.00076 | 5.1 × 10⁻¹¹ | improvement *** |
| 0.15 | Dice | −0.00115 | 9.0 × 10⁻⁶ | degradation *** |
| 0.15 | Precision | +0.00112 | 7.1 × 10⁻¹² | improvement *** |
| 0.20 | Dice | −0.00133 | 7.8 × 10⁻⁸ | degradation *** |
| 0.20 | Precision | +0.00025 | 2.5 × 10⁻¹¹ | improvement *** |
| 0.25 | Dice | −0.00110 | 9.8 × 10⁻⁶ | degradation *** |
| 0.25 | Precision | +0.00125 | 1.9 × 10⁻⁹ | improvement *** |
| 0.30 | Dice | −0.00039 | 1.4 × 10⁻⁶ | degradation *** |
| 0.30 | Precision | +0.00153 | 1.6 × 10⁻¹¹ | improvement *** |
| 0.35 | Dice | +0.00007 | 4.0 × 10⁻⁸ | improvement *** |
| 0.35 | Precision | +0.00317 | 9.1 × 10⁻¹⁵ | improvement *** |
| 0.40 | Dice | −0.00070 | 4.5 × 10⁻⁶ | degradation *** |
| 0.40 | Precision | +0.00209 | 6.2 × 10⁻¹² | improvement *** |
| 0.45 | Dice | −0.00039 | 3.5 × 10⁻⁶ | degradation *** |
| 0.45 | Precision | +0.00332 | 1.2 × 10⁻¹² | improvement *** |
| 0.50 | Dice | −0.00053 | 1.2 × 10⁻⁵ | degradation *** |
| 0.50 | Precision | +0.00324 | 1.0 × 10⁻¹¹ | improvement *** |

Precision improves significantly at every d. Effect sizes are roughly constant across d — shift fraction has little bearing on cleaning quality.

#### Per structure (BH-FDR corrected, α = 0.05)

##### Dice — significantly degraded

| Structure | Mean ΔDice | n nonzero | p_adj |
| --- | --- | --- | --- |
| iliac_artery_right | −0.07768 | 42 | 5.3 × 10⁻⁸ |
| iliac_vena_right | −0.02490 | 19 | 2.6 × 10⁻³ |
| small_bowel | −0.00376 | 309 | 9.5 × 10⁻³⁴ |
| stomach | −0.00040 | 113 | 4.5 × 10⁻⁹ |
| gluteus_maximus_left | −0.00008 | 8 | 0.013 |
| femur_right | −0.00001 | 8 | 0.013 |

##### Dice — significantly improved

| Structure | Mean ΔDice | n nonzero | p_adj |
| --- | --- | --- | --- |
| portal_vein_and_splenic_vein | +0.01051 | 93 | 6.9 × 10⁻¹⁶ |
| duodenum | +0.00611 | 61 | 3.7 × 10⁻⁴ |
| brain | +0.00233 | 31 | 3.3 × 10⁻⁶ |
| pancreas | +0.00216 | 64 | 1.9 × 10⁻⁹ |
| liver | +0.00121 | 99 | 4.7 × 10⁻⁸ |
| kidney_right | +0.00053 | 104 | 2.7 × 10⁻¹⁴ |
| aorta | +0.00038 | 43 | 4.6 × 10⁻⁸ |

##### Precision — significantly degraded

| Structure | Mean ΔPrecision | n nonzero | p_adj |
| --- | --- | --- | --- |
| small_bowel | −0.00106 | 309 | 9.8 × 10⁻³⁵ |
| iliac_artery_right | −0.00491 | 42 | 0.017 |
| femur_right | −0.00001 | 10 | 0.014 |

##### Precision — significantly improved

| Structure | Mean ΔPrecision | n nonzero | p_adj |
| --- | --- | --- | --- |
| urinary_bladder | +0.02810 | 21 | 0.021 |
| portal_vein_and_splenic_vein | +0.01916 | 93 | 7.2 × 10⁻¹⁶ |
| iliac_vena_left | +0.01625 | 8 | 0.014 |
| duodenum | +0.00975 | 61 | 9.3 × 10⁻⁶ |
| pancreas | +0.00728 | 64 | 3.0 × 10⁻¹¹ |
| brain | +0.00427 | 31 | 3.8 × 10⁻⁶ |
| liver | +0.00353 | 108 | 1.3 × 10⁻⁹ |
| stomach | +0.00271 | 117 | 4.4 × 10⁻¹¹ |
| kidney_right | +0.00096 | 104 | 2.1 × 10⁻¹⁴ |
| aorta | +0.00050 | 43 | 4.1 × 10⁻⁸ |
| kidney_left | +0.00011 | 23 | 6.8 × 10⁻⁵ |

The `iliac_artery_right` degradation is the largest single failure and accounts for the majority of the mean Dice drop.

---

## `data_prep/` — Data preparation

| File | Purpose |
| --- | --- |
| `simulate_wraparound_artifact.py` | Simulate S-I MRI wrap-around artifacts; sweeps over d (shift fraction) and r (ghost intensity) |
| `simulate_all_subjects.sh` | Run WM3 simulation for all 10 test subjects |
| `compute_com_landmark_normalized.py` | Compute MRI CoM atlas: landmark-normalised (vertebrae span L5→C1 = 0–100); test subjects excluded |
| `compute_empirical_poset.py` | Compute empirical probability poset from GT bounding boxes; test subjects excluded |
| `rank_mri_subjects.py` | Rank MRI subjects by number of non-empty GT structure masks |
| `analyze_mri_coverage.py` | Analyse per-structure GT coverage across the MRI dataset |
| `run_totalseg_on_artifacts.sh` | Run TotalSegmentator on a set of artifact NIfTI files |
| `visualize_mri_wraparound.py` | Visualise coronal slices before/after artifact simulation |
