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
| `evaluate_cleaning_methods.py` | Compare three cleaning methods (M1 unidirectional, M2 symmetric, M3 middle-out+prior) on all 18 MRI wrap-around artifact conditions; outputs Dice CSV and comparison plots |
| `save_cleaned_segmentations.py` | Run any one cleaning method on a single artifact condition and write cleaned NIfTI files to disk for 3D Slicer visualisation |
| `segment_one_subject.sh` | sbatch job: runs TotalSegmentator on all `mri_artifact.nii.gz` files for one subject (skips existing); submit with `--export=SUBJECT=sXXXX` |
| `submit_segmentation_jobs.sh` | Submits one `segment_one_subject.sh` sbatch job per unsegmented subject (s0022, s0167, s0186, s0187, s0219, s0236, s0237, s0243, s0250) |

### Cleaning methods

**M1 — Unidirectional (current baseline)**
For each constraint pair `(i above j)`, finds the LCC of `j` and removes any non-LCC component of `i` whose S-I extent lies entirely below `j`'s LCC inferior boundary. Handles superior-anatomy ghosts at the inferior crop edge.

**M2 — Symmetric**
Extends M1 in both directions: also removes non-LCC components of `j` that sit entirely above `i`'s LCC superior boundary. Catches both wrap directions but is more aggressive and occasionally removes correct LCCs.

**M3 — Middle-out + spatial prior (fixed)**
Like M2, but selects the "real" component using the atlas CoM prior (`totalseg_v2_com.json`) instead of pure LCC, and processes pairs ordered from most-central structures outward. Two key design decisions:

- **Size dominance guard**: the prior only overrides the LCC when the two largest CCs are within 5× of each other in volume; if the LCC is clearly dominant it is trusted unconditionally.
- **Out-of-crop anchor guard**: if a structure's atlas-expected position falls outside `[0, N)` for the current crop, it is skipped as an anchor entirely rather than falling back to its LCC. This prevents ghost predictions of out-of-crop structures (e.g. a 2-voxel femur fragment at the top of a heart–kidney crop) from acting as spurious anchors that incorrectly remove large legitimate components.

**Benchmark results** (subject s0175, 18 wrap-around conditions, threshold=0.95):

| Method | Mean ΔDice | Improved | Degraded |
| --- | --- | --- | --- |
| M1 unidirectional | +0.00025 | 7/236 (3.0%) | 3/236 (1.3%) |
| M2 symmetric | −0.00033 | 12/236 (5.1%) | 8/236 (3.4%) |
| M3 middle-out+prior | +0.00011 | 4/236 (1.7%) | 0/236 (0.0%) |

M3 is the most conservative: it never degrades a structure and the out-of-crop guard eliminates cascading anchor errors. M1 has the highest mean improvement. M2 finds more individual improvements but also makes the most mistakes.

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

#### V1 — Paper formula (currently used, matches existing 880 artifacts)

```python
I_s[V == 2] *= I2 / I1   # I2 = wrapped area sum, I1 = non-wrapped area sum
```

Maintains the brightness ratio between wrapped and non-wrapped areas from the original image. Works well for horizontal wrap-around (large wrapped area). For vertical/S-I wrap-around with small d, I2/I1 ≪ 1 and the wrapped strip appears very dark or black — visually unrealistic but consistent with the paper.

#### V2 — Region-local normalisation

```python
I_r2 = float(I_r[V == 2].sum())
I_s[V == 2] *= I2 / I_r2
```

Scales the wrapped region back to its original total brightness. The ghost redistributes signal within the strip rather than darkening it. Avoids the blackout but is still normalised.

#### V3 — No normalisation (most physically realistic)

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
