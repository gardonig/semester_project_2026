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

---

## `poset_construction/` — Building anatomical posets

| Script | Purpose |
| --- | --- |
| `llm_poset_builder.py` | Query an LLM to fill a poset matrix interactively |
| `generate_llm_poset_knowledge.py` | Generate LLM poset for the default structure set |
| `generate_llm_poset_v157.py` | Generate LLM poset specifically for TotalSegmentator v157 structures |

---

## `segmentation/` — Third-party segmentor tools

| Script | Purpose |
| --- | --- |
| `compare_segmenters.py` | Dice comparison across TotalSegmentator, MedSAM, VibeSeg |
| `run_medsam.py` | Run MedSAM inference on a dataset |
| `setup_medsam.sh` | Install MedSAM and dependencies into a venv |
| `setup_vibeseg.sh` | Install VibeSeg and dependencies into a venv |

---

## `data_prep/` — Data preparation

| File | Purpose |
| --- | --- |
| `compute_com_from_gt.py` | Compute average CoM for every TS structure from GT masks; outputs GUI-ready JSON |
| `compute_empirical_poset.py` | Compute empirical probability poset from GT bounding boxes across all subjects; outputs probability matrix JSON usable by the cleaning script |
| `CoM_extractor.ipynb` | Original CoM extraction notebook (56 structures, older dataset) |

---

## `dev/` — Research and visualisation helpers

| Script | Purpose |
| --- | --- |
| `algorithm1_matrix_walkthrough.py` | Step-through demo of the gap-based query algorithm for n=4 structures |
| `view_segmentation.py` | Quick viewer for NIfTI segmentation masks |
| `view_full_body_male.py` | Render the full-body visible-human volume tensor |
| `stand_alone_poset_anatomy.py` | Self-contained prototype of the poset GUI (no package install needed) |
