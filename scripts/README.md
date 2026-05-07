# Scripts

Organised by task — pick the folder for the work you want to do.

---

## `cleaning/` — Segmentation constraint post-processing

| Script | Purpose |
| --- | --- |
| `poset_constraint_postprocessing.py` | Main cleaning pipeline: applies poset constraints via CC analysis, outputs cleaned masks and Dice CSV |
| `summarize_results.py` | Aggregate per-subject CSVs into per-structure summary tables; supports side-by-side comparison |
| `visualize_cleaning.py` | Axial-slice visualisation of before/after cleaning (green=kept, red=removed, blue=GT) |
| `truncated_fov_experiment.py` | Simulate truncated FOV scans and evaluate cleaning benefit |
| `evaluate_cleaning_methods.py` | Compare three cleaning methods (CM1 unidirectional, CM2 symmetric, CM3 middle-out+prior) on all 18 MRI WM3 artifact conditions; outputs Dice CSV and comparison plots |
| `save_cleaned_segmentations.py` | Run any one cleaning method on a single artifact condition and write cleaned NIfTI files to disk for 3D Slicer visualisation |
| `segment_one_subject.sh` | sbatch job: runs TotalSegmentator on all `mri_artifact.nii.gz` files for one subject (skips existing); submit with `--export=SUBJECT=sXXXX` |
| `submit_segmentation_jobs.sh` | Submits one `segment_one_subject.sh` sbatch job per unsegmented subject (s0022, s0167, s0186, s0187, s0219, s0236, s0237, s0243, s0250) |

### Cleaning methods

**CM1 — Unidirectional (current baseline)**
For each constraint pair `(i above j)`, finds the LCC of `j` and removes any non-LCC component of `i` whose S-I extent lies entirely below `j`'s LCC inferior boundary. Handles superior-anatomy ghosts at the inferior crop edge.

**CM2 — Symmetric**
Extends CM1 in both directions: also removes non-LCC components of `j` that sit entirely above `i`'s LCC superior boundary. Catches both wrap directions but is more aggressive and occasionally removes correct LCCs.

**CM3 — Middle-out + spatial prior (fixed)**
Like CM2, but selects the "real" component using the atlas CoM prior (`totalseg_v2_com.json`) instead of pure LCC, and processes pairs ordered from most-central structures outward. Two key design decisions:

- **Size dominance guard**: the prior only overrides the LCC when the two largest CCs are within 5× of each other in volume; if the LCC is clearly dominant it is trusted unconditionally.
- **Out-of-crop anchor guard**: if a structure's atlas-expected position falls outside `[0, N)` for the current crop, it is skipped as an anchor entirely rather than falling back to its LCC. This prevents ghost predictions of out-of-crop structures (e.g. a 2-voxel femur fragment at the top of a heart–kidney crop) from acting as spurious anchors that incorrectly remove large legitimate components.

**Benchmark results** (subject s0175, 18 WM3 artifact conditions, threshold=0.95):

| Method | Mean ΔDice | Improved | Degraded |
| --- | --- | --- | --- |
| CM1 unidirectional | +0.00025 | 7/236 (3.0%) | 3/236 (1.3%) |
| CM2 symmetric | −0.00033 | 12/236 (5.1%) | 8/236 (3.4%) |
| CM3 middle-out+prior | +0.00011 | 4/236 (1.7%) | 0/236 (0.0%) |

CM3 is the most conservative: it never degrades a structure and the out-of-crop guard eliminates cascading anchor errors. CM1 has the highest mean improvement. CM2 finds more individual improvements but also makes the most mistakes.

---

### Evaluation metrics — why precision, and why we keep Dice

#### Why precision is the primary metric for evaluating cleaning

CM3 is a purely subtractive method: it can only remove voxels, never add them. This has a precise consequence for the standard metrics:

- **True positives (TP)** are unchanged — cleaning cannot create new correct predictions.
- **False negatives (FN)** are unchanged — missed GT voxels remain missed.
- **False positives (FP)** decrease — ghost artifact voxels are removed.

Because recall = TP / (TP + FN), recall is mathematically invariant to cleaning. Dice = 2·TP / (2·TP + FP + FN) depends on FN as well as FP, so its maximum achievable value after perfect cleaning is capped at 2·Recall / (1 + Recall). When recall is already poor — which happens for structures the segmenter struggles with even before the artifact — removing every FP barely moves the Dice needle. In the worst case (TP = 0, e.g. a structure entirely missed by the network but hallucinated in the ghost), Dice stays 0 before and after cleaning even when the cleaning is perfect.

Precision = TP / (TP + FP) is sensitive only to the dimension cleaning actually operates on. Removing ghost FPs directly and proportionally improves precision regardless of recall. It is therefore the metric that most faithfully reflects what CM3 does.

#### Why Dice is still reported

Dice is the de-facto standard metric for segmentation model evaluation and appears in virtually all TotalSegmentator benchmarks. Reporting it allows direct comparison with published baselines and makes the trade-offs visible: a method could improve precision (fewer spurious predictions) while marginally degrading Dice (by also accidentally removing a few TPs), and both facts matter.

---

### Statistical evaluation of CM3 — full results (threshold = 0.95)

*Dataset: 10 subjects × 3 crops × 40 artifact conditions (10 d × 4 r) = 1 200 (subject, crop, condition) tuples. Wilcoxon signed-rank test on non-zero differences; per-structure p-values corrected with Benjamini–Hochberg FDR (α = 0.05).*

#### Overall

| Metric | Mean Δ | n nonzero | p-value | Result |
| --- | --- | --- | --- | --- |
| Dice | −0.00089 | 1327 / 12673 | 8.7 × 10⁻⁴⁸ | significant **degradation** |
| Precision | +0.00166 | 1360 / 12673 | 3.1 × 10⁻¹⁰¹ | significant **improvement** |

The opposite directions confirm the metric argument above: CM3 systematically shifts FP to TP at the cost of a very small Dice penalty driven by a handful of structures.

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

At r = 0.25 (weakest ghost) both metrics degrade: the ghost is too faint to clearly distinguish from real anatomy and CM3 over-removes. From r = 0.5 onward precision improves consistently and strongly; Dice degrades at all r but the magnitude shrinks as r increases.

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

Precision improves significantly at every d. Dice degrades at all d except 0.35 (marginally positive). Effect sizes are small and roughly constant across d, suggesting the shift fraction has little bearing on cleaning quality.

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
| autochthon_right | +0.00001 | 10 | 3.8 × 10⁻³ |
| kidney_left | +0.00000 | 16 | 9.1 × 10⁻⁴ |
| hip_left | +0.00000 | 13 | 0.026 |

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
| gluteus_maximus_right | +0.00504 | 6 | 0.045 |
| autochthon_right | +0.00235 | 12 | 1.2 × 10⁻³ |
| kidney_right | +0.00096 | 104 | 2.1 × 10⁻¹⁴ |
| aorta | +0.00050 | 43 | 4.1 × 10⁻⁸ |
| kidney_left | +0.00011 | 23 | 6.8 × 10⁻⁵ |
| inferior_vena_cava | +0.00010 | 10 | 8.5 × 10⁻³ |
| hip_left | +0.00008 | 16 | 0.014 |

The `iliac_artery_right` degradation in both metrics is the largest single failure: it accounts for the vast majority of the mean Dice drop and warrants targeted inspection. The `small_bowel` degradation is significant but small in effect size; its high n_nonzero reflects that the structure is large and frequently touched by the crop boundary rather than a systematic cleaning error.

---

## `poset_construction/` — Building anatomical posets

| Script | Purpose |
| --- | --- |
| `llm_poset_builder.py` | Query an LLM to fill a poset matrix interactively |
| `generate_llm_poset_knowledge.py` | Generate LLM poset for the default structure set |
| `generate_llm_poset_v157.py` | Generate LLM poset specifically for TotalSegmentator v157 structures |

---

## `data_prep/` — Data preparation

| File | Purpose |
| --- | --- |
| `compute_com_from_gt.py` | Compute average CoM for every TS structure from GT masks; outputs GUI-ready JSON |
| `compute_empirical_poset.py` | Compute empirical probability poset from GT bounding boxes across all subjects; outputs probability matrix JSON usable by the cleaning script |
| `simulate_wraparound_artifact.py` | Simulate S-I MRI wrap-around artifacts; sweeps over d (shift fraction) and r (ghost intensity) |
| `CoM_extractor.ipynb` | Original CoM extraction notebook (56 structures, older dataset) |

### Artifact simulation — brightness normalisation variants

`simulate_wraparound_artifact.py` implements Equations 1–4 from the paper. The brightness normalisation step (Eq. 4) has three meaningful variants with different realism trade-offs:

#### WM1 — Paper formula (matches paper Eq. 4)

```python
I_s[V == 2] *= I2 / I1   # I2 = wrapped area sum, I1 = non-wrapped area sum
```

Maintains the brightness ratio between wrapped and non-wrapped areas from the original image. Works well for horizontal wrap-around (large wrapped area). For vertical/S-I wrap-around with small d, I2/I1 ≪ 1 and the wrapped strip appears very dark or black — visually unrealistic but consistent with the paper.

#### WM2 — Region-local normalisation

```python
I_r2 = float(I_r[V == 2].sum())
I_s[V == 2] *= I2 / I_r2
```

Scales the wrapped region back to its original total brightness. The ghost redistributes signal within the strip rather than darkening it. Avoids the blackout but is still normalised.

#### WM3 — No normalisation (most physically realistic, currently used)

```python
I_s = I_r.copy()
```

The ghost adds directly on top of the existing signal, exactly as MRI aliasing works in hardware. The `r` parameter alone controls ghost intensity. The wrapped area appears slightly brighter than original, which matches real-world observations.

To switch variant, edit lines ~255–260 of `simulate_wraparound_artifact.py` and regenerate artifacts.

---

## `dev/` — Research and visualisation helpers

| Script | Purpose |
| --- | --- |
| `algorithm1_matrix_walkthrough.py` | Step-through demo of the gap-based query algorithm for n=4 structures |
| `view_segmentation.py` | Quick viewer for NIfTI segmentation masks |
| `view_full_body_male.py` | Render the full-body visible-human volume tensor |
| `stand_alone_poset_anatomy.py` | Self-contained prototype of the poset GUI (no package install needed) |
