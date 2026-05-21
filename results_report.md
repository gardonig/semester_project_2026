# Results: Wrap-Around Artefact Cleaning

---

## Overview

Ten whole-body MRI subjects were segmented with TotalSegmentator under synthetic wrap-around artefacts (ghost intensity *r* ∈ {0.25, 0.50, 0.75, 1.00}; shift fraction *d* ∈ {0.05, …, 0.50}; 40 conditions per subject). At maximum severity (*r*=1.00, *d*=0.50) Dice and **F1** (identical to Dice here — standard binary overlap) drop from **0.821** to **0.493** and Precision from **0.839** to **0.785** relative to the clean reference. We compare three subtractive baselines — **LCC only** (largest connected component per structure), **morphological opening** with ball radii **1** and **2** — to **Poset-Based Cleaning** (anatomical poset constraints plus SI **mid-plane proximity** conflict resolution at threshold *t*=1.00). Merged metrics for Poset-Based Cleaning are in `data/wraparound_experiments/wraparound_v4_eval_cm4/t100/results.csv` (**25 866** structure×condition pairs). Baselines use the same crops and subjects but **19 801** pairs per method (erosion evaluation merge); artifact Dice/Prec therefore differ slightly from the Poset-Based Cleaning row, which is evaluated on the full pair set.

---

## 1. Method comparison — the core trade-off

All methods only remove predicted voxels. Removing ghost false positives raises Precision; removing true anatomy lowers Dice/F1. **F1** matches **Dice** in every row below (harmonic mean of precision and recall vs ground truth).

**Summary (mean ± SD over all artefact conditions):**

| Method | Dice (artifact) | F1 (artifact) | Dice (cleaned) | F1 (cleaned) | Δ Dice | Δ F1 | Prec (artifact) | Prec (cleaned) | Δ Prec | Prec recovery† | ΔPrec/|ΔDice| |
| ------ | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No cleaning | 0.757 | 0.757 | — | — | — | — | 0.809 | — | — | 0% | — |
| LCC only | 0.751 | 0.751 | 0.704 | 0.704 | -0.047 ± 0.138 | -0.047 ± 0.138 | 0.806 | 0.81 | +0.003 ± 0.065 | 11% | 0.07 |
| Opening r=1 | 0.751 | 0.751 | 0.673 | 0.673 | -0.078 ± 0.158 | -0.078 ± 0.158 | 0.806 | 0.849 | +0.043 ± 0.078 | **129%** (overshoot) | 0.55 |
| Opening r=2 | 0.751 | 0.751 | 0.649 | 0.649 | -0.102 ± 0.183 | -0.102 ± 0.183 | 0.806 | 0.853 | +0.046 ± 0.086 | **141%** (overshoot) | 0.46 |
| **Poset-Based Cleaning (t=1.00)** | 0.757 | 0.757 | 0.755 | 0.755 | -0.001 ± 0.075 | -0.001 ± 0.075 | 0.809 | 0.819 | +0.010 ± 0.088 | 34% | 8.65 |

†Prec recovery = (Prec cleaned − Prec artifact) / (Prec reference − Prec artifact), reference = no-artefact run.

Morphological opening (r=1) maximises the raw ΔPrec/|ΔDice| ratio among baselines but **overshoots** Precision past the no-artefact level while dropping Dice sharply. **Poset-Based Cleaning** keeps mean Dice/F1 nearly unchanged (**≈ −0.001**) with a clear mean Precision gain (**≈ +0.010**, **~34%** recovery); the efficiency ratio is huge only because |ΔDice| is near zero (not comparable to high-loss methods).

**Figures:** `data/wraparound_experiments/wraparound_v4_eval_cm4/fig_tradeoff_scatter.png`, `fig_efficiency_bar.png` (and sibling plots from `plot_wraparound_method_figures.py`).

---

## 2. Adaptivity — effect vs shift fraction *d*

LCC and opening apply a fixed morphological rule at every *d*. Poset-Based Cleaning fires only when poset constraints are violated, so mean ΔDice and ΔPrec stay near zero for small ghosts (*d* ≤ 0.10) then grow with severity.

**ΔDice / ΔF1 / ΔPrec vs *d* (averaged over *r*).** ΔF1 = ΔDice in every cell.

| *d* | LCC ΔDice | Op. r=1 | Op. r=2 | Poset-Based ΔDice | LCC ΔPrec | Op. r=1 | Op. r=2 | Poset-Based ΔPrec |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.05 | −0.046 | −0.077 | −0.101 | 0.000 | +0.003 | +0.045 | +0.050 | 0.000 |
| 0.10 | −0.046 | −0.077 | −0.104 | 0.000 | +0.004 | +0.044 | +0.049 | 0.000 |
| 0.15 | −0.046 | −0.077 | −0.101 | +0.001 | +0.004 | +0.044 | +0.049 | +0.003 |
| 0.20 | −0.046 | −0.079 | −0.103 | +0.003 | +0.004 | +0.042 | +0.046 | +0.009 |
| 0.25 | −0.048 | −0.079 | −0.104 | +0.005 | +0.002 | +0.043 | +0.045 | +0.012 |
| 0.30 | −0.046 | −0.078 | −0.102 | +0.006 | +0.005 | +0.043 | +0.047 | +0.015 |
| 0.35 | −0.047 | −0.078 | −0.102 | +0.004 | +0.004 | +0.042 | +0.044 | +0.017 |
| 0.40 | −0.048 | −0.077 | −0.101 | −0.000 | +0.004 | +0.041 | +0.045 | +0.016 |
| 0.45 | −0.048 | −0.077 | −0.100 | −0.006 | +0.004 | +0.041 | +0.045 | +0.019 |
| 0.50 | −0.050 | −0.078 | −0.103 | −0.027 | +0.002 | +0.040 | +0.043 | +0.014 |

**Figures:** `fig_adaptivity_by_d.png`; heatmaps of mean ΔDice over (*d*, *r*) for each method — `data/wraparound_experiments/wraparound_v4_eval_cm4/t100/heatmap_delta_d_r.png` (Poset-Based Cleaning) and the erosion baseline bundle under `wraparound_v4_eval/` as produced by your plotting scripts.

---

## 3. Regional analysis — three crop windows

| Method | Crop | Dice (art) | F1 (art) | Dice (cl) | F1 (cl) | Δ Dice | Δ F1 | Prec (art) | Prec (cl) | Δ Prec |
| ------ | ---- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LCC only | brain → heart | 0.704 | 0.704 | 0.642 | 0.642 | -0.062 ± 0.163 | -0.062 ± 0.163 | 0.74 | 0.734 | -0.006 ± 0.107 |
| Opening r=1 | brain → heart | 0.704 | 0.704 | 0.628 | 0.628 | -0.076 ± 0.171 | -0.076 ± 0.171 | 0.74 | 0.775 | +0.035 ± 0.121 |
| Opening r=2 | brain → heart | 0.704 | 0.704 | 0.6 | 0.6 | -0.104 ± 0.195 | -0.104 ± 0.195 | 0.74 | 0.773 | +0.033 ± 0.141 |
| Poset-Based Cleaning | brain → heart | 0.704 | 0.704 | 0.675 | 0.675 | -0.029 ± 0.154 | -0.029 ± 0.154 | 0.74 | 0.753 | +0.013 ± 0.124 |
| LCC only | heart → kidney | 0.685 | 0.685 | 0.633 | 0.633 | -0.053 ± 0.154 | -0.053 ± 0.154 | 0.75 | 0.753 | +0.003 ± 0.101 |
| Opening r=1 | heart → kidney | 0.685 | 0.685 | 0.597 | 0.597 | -0.089 ± 0.166 | -0.089 ± 0.166 | 0.75 | 0.787 | +0.037 ± 0.111 |
| Opening r=2 | heart → kidney | 0.685 | 0.685 | 0.578 | 0.578 | -0.107 ± 0.181 | -0.107 ± 0.181 | 0.75 | 0.784 | +0.033 ± 0.112 |
| Poset-Based Cleaning | heart → kidney | 0.738 | 0.738 | 0.738 | 0.738 | -0.000 ± 0.101 | -0.000 ± 0.101 | 0.789 | 0.809 | +0.020 ± 0.119 |
| LCC only | kidney → hip | 0.776 | 0.776 | 0.732 | 0.732 | -0.044 ± 0.129 | -0.044 ± 0.129 | 0.83 | 0.835 | +0.005 ± 0.040 |
| Opening r=1 | kidney → hip | 0.776 | 0.776 | 0.701 | 0.701 | -0.075 ± 0.154 | -0.075 ± 0.154 | 0.83 | 0.875 | +0.045 ± 0.057 |
| Opening r=2 | kidney → hip | 0.776 | 0.776 | 0.676 | 0.676 | -0.100 ± 0.182 | -0.100 ± 0.182 | 0.83 | 0.882 | +0.052 ± 0.066 |
| Poset-Based Cleaning | kidney → hip | 0.776 | 0.776 | 0.777 | 0.777 | +0.001 ± 0.019 | +0.001 ± 0.019 | 0.83 | 0.833 | +0.003 ± 0.048 |

**kidney → hip:** Poset-Based Cleaning is the only method with a small **positive** mean ΔDice (**+0.001**); LCC and opening both lose Dice on average here while raising Precision. **brain → heart:** opening and LCC reduce Dice strongly; Poset-Based Cleaning limits mean ΔDice to about **−0.03** with a modest Precision lift. **heart → kidney:** Poset-Based Cleaning is near-neutral in Dice with **+0.02** mean ΔPrec on average.

**Figure:** `fig_regional_bars.png`.

---

## 4. Structural extremes (Poset-Based Cleaning)

At the structure level, the largest mean improvements under Poset-Based Cleaning are pelvic / lower abdominal (e.g. **gluteus_medius**, **iliopsoas**, **small_bowel**); the largest mean degradations hit small thoraco-abdominal organs and vessels when ordering conflicts are ambiguous (e.g. **adrenal_gland_left**, **humerus_right**, **portal_vein_and_splenic_vein**). Full ranked tables and pair-level counts are in **Appendix A4**.

---

## Appendix: stratified tables (Poset-Based Cleaning on CM4 merge; baselines on erosion merge)

## A1. Artefact severity vs reference (means on Poset-Based Cleaning *artifact* rows, *N* = 25 866)

*Dice drop / Prec drop vs clean reference (0.821 / 0.839). F1 = Dice.*

### By ghost intensity *r*

| *r* | Dice | F1 | Dice drop | Prec | Prec drop |
| --- | :---: | :---: | :---: | :---: | :---: |
| 0.25 | 0.8 | 0.8 | -0.021 | 0.83 | -0.009 |
| 0.5 | 0.773 | 0.773 | -0.048 | 0.818 | -0.021 |
| 0.75 | 0.739 | 0.739 | -0.082 | 0.8 | -0.039 |
| 1.0 | 0.712 | 0.712 | -0.109 | 0.784 | -0.055 |

### By shift fraction *d*

| *d* | Dice | F1 | Dice drop | Prec | Prec drop |
| --- | :---: | :---: | :---: | :---: | :---: |
| 0.05 | 0.807 | 0.807 | -0.014 | 0.835 | -0.004 |
| 0.1 | 0.8 | 0.8 | -0.021 | 0.834 | -0.005 |
| 0.15 | 0.787 | 0.787 | -0.034 | 0.827 | -0.012 |
| 0.2 | 0.773 | 0.773 | -0.048 | 0.819 | -0.02 |
| 0.25 | 0.758 | 0.758 | -0.063 | 0.808 | -0.031 |
| 0.3 | 0.744 | 0.744 | -0.077 | 0.801 | -0.038 |
| 0.35 | 0.734 | 0.734 | -0.087 | 0.796 | -0.043 |
| 0.4 | 0.727 | 0.727 | -0.094 | 0.791 | -0.048 |
| 0.45 | 0.717 | 0.717 | -0.104 | 0.786 | -0.053 |
| 0.5 | 0.713 | 0.713 | -0.108 | 0.785 | -0.054 |

---

## A2. By ghost intensity *r*

*F1 = Dice in every cell.*

| Method | *r* | Dice (art) | Dice (cl) | Δ Dice | Prec (art) | Prec (cl) | Δ Prec |
| ------ | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| LCC only | 0.25 | 0.79 | 0.744 | -0.046 ± 0.137 | 0.825 | 0.828 | +0.003 ± 0.039 |
| Opening r=1 | 0.25 | 0.79 | 0.713 | -0.077 ± 0.159 | 0.825 | 0.87 | +0.045 ± 0.056 |
| Opening r=2 | 0.25 | 0.79 | 0.687 | -0.103 ± 0.186 | 0.825 | 0.875 | +0.050 ± 0.066 |
| Poset-Based Cleaning | 0.25 | 0.8 | 0.798 | -0.002 ± 0.036 | 0.83 | 0.831 | +0.001 ± 0.018 |
| LCC only | 0.5 | 0.765 | 0.718 | -0.047 ± 0.138 | 0.814 | 0.817 | +0.003 ± 0.061 |
| Opening r=1 | 0.5 | 0.765 | 0.686 | -0.079 ± 0.159 | 0.814 | 0.856 | +0.042 ± 0.076 |
| Opening r=2 | 0.5 | 0.765 | 0.662 | -0.102 ± 0.183 | 0.814 | 0.861 | +0.048 ± 0.085 |
| Poset-Based Cleaning | 0.5 | 0.773 | 0.772 | -0.001 ± 0.059 | 0.818 | 0.822 | +0.003 ± 0.052 |
| LCC only | 0.75 | 0.736 | 0.688 | -0.048 ± 0.137 | 0.8 | 0.803 | +0.003 ± 0.073 |
| Opening r=1 | 0.75 | 0.736 | 0.657 | -0.079 ± 0.158 | 0.8 | 0.84 | +0.041 ± 0.085 |
| Opening r=2 | 0.75 | 0.736 | 0.634 | -0.102 ± 0.182 | 0.8 | 0.844 | +0.044 ± 0.093 |
| Poset-Based Cleaning | 0.75 | 0.739 | 0.738 | -0.001 ± 0.090 | 0.8 | 0.815 | +0.014 ± 0.103 |
| LCC only | 1.0 | 0.711 | 0.663 | -0.048 ± 0.139 | 0.785 | 0.789 | +0.004 ± 0.079 |
| Opening r=1 | 1.0 | 0.711 | 0.636 | -0.075 ± 0.156 | 0.785 | 0.827 | +0.043 ± 0.090 |
| Opening r=2 | 1.0 | 0.711 | 0.611 | -0.101 ± 0.180 | 0.785 | 0.829 | +0.044 ± 0.095 |
| Poset-Based Cleaning | 1.0 | 0.712 | 0.712 | -0.001 ± 0.101 | 0.784 | 0.808 | +0.024 ± 0.133 |

---

## A3. By shift fraction *d*

*F1 = Dice in every cell.*

| Method | *d* | Dice (art) | Dice (cl) | Δ Dice | Prec (art) | Prec (cl) | Δ Prec |
| ------ | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| LCC only | 0.05 | 0.796 | 0.75 | -0.046 ± 0.137 | 0.828 | 0.831 | +0.003 ± 0.034 |
| Opening r=1 | 0.05 | 0.796 | 0.72 | -0.077 ± 0.159 | 0.828 | 0.874 | +0.045 ± 0.054 |
| Opening r=2 | 0.05 | 0.796 | 0.696 | -0.101 ± 0.184 | 0.828 | 0.879 | +0.050 ± 0.063 |
| Poset-Based Cleaning | 0.05 | 0.807 | 0.807 | +0.000 ± 0.000 | 0.835 | 0.835 | +0.000 ± 0.000 |
| LCC only | 0.1 | 0.788 | 0.742 | -0.046 ± 0.137 | 0.828 | 0.831 | +0.004 ± 0.038 |
| Opening r=1 | 0.1 | 0.788 | 0.711 | -0.077 ± 0.159 | 0.828 | 0.872 | +0.044 ± 0.057 |
| Opening r=2 | 0.1 | 0.788 | 0.685 | -0.104 ± 0.187 | 0.828 | 0.877 | +0.049 ± 0.067 |
| Poset-Based Cleaning | 0.1 | 0.8 | 0.8 | +0.000 ± 0.000 | 0.834 | 0.834 | +0.000 ± 0.001 |
| LCC only | 0.15 | 0.775 | 0.73 | -0.046 ± 0.136 | 0.822 | 0.826 | +0.004 ± 0.044 |
| Opening r=1 | 0.15 | 0.775 | 0.698 | -0.077 ± 0.158 | 0.822 | 0.866 | +0.044 ± 0.062 |
| Opening r=2 | 0.15 | 0.775 | 0.674 | -0.101 ± 0.182 | 0.822 | 0.871 | +0.049 ± 0.072 |
| Poset-Based Cleaning | 0.15 | 0.787 | 0.788 | +0.001 ± 0.020 | 0.827 | 0.83 | +0.003 ± 0.041 |
| LCC only | 0.2 | 0.763 | 0.716 | -0.046 ± 0.138 | 0.813 | 0.817 | +0.004 ± 0.059 |
| Opening r=1 | 0.2 | 0.763 | 0.684 | -0.079 ± 0.161 | 0.813 | 0.855 | +0.042 ± 0.075 |
| Opening r=2 | 0.2 | 0.763 | 0.66 | -0.103 ± 0.184 | 0.813 | 0.86 | +0.046 ± 0.084 |
| Poset-Based Cleaning | 0.2 | 0.773 | 0.776 | +0.003 ± 0.032 | 0.819 | 0.828 | +0.009 ± 0.078 |
| LCC only | 0.25 | 0.751 | 0.703 | -0.048 ± 0.140 | 0.805 | 0.807 | +0.002 ± 0.070 |
| Opening r=1 | 0.25 | 0.751 | 0.672 | -0.079 ± 0.158 | 0.805 | 0.848 | +0.043 ± 0.078 |
| Opening r=2 | 0.25 | 0.751 | 0.647 | -0.104 ± 0.185 | 0.805 | 0.851 | +0.045 ± 0.090 |
| Poset-Based Cleaning | 0.25 | 0.758 | 0.763 | +0.005 ± 0.049 | 0.808 | 0.82 | +0.012 ± 0.090 |
| LCC only | 0.3 | 0.742 | 0.696 | -0.046 ± 0.138 | 0.801 | 0.806 | +0.005 ± 0.071 |
| Opening r=1 | 0.3 | 0.742 | 0.664 | -0.078 ± 0.158 | 0.801 | 0.844 | +0.043 ± 0.082 |
| Opening r=2 | 0.3 | 0.742 | 0.64 | -0.102 ± 0.184 | 0.801 | 0.848 | +0.047 ± 0.087 |
| Poset-Based Cleaning | 0.3 | 0.744 | 0.751 | +0.006 ± 0.066 | 0.801 | 0.817 | +0.015 ± 0.100 |
| LCC only | 0.35 | 0.733 | 0.685 | -0.047 ± 0.138 | 0.795 | 0.8 | +0.004 ± 0.076 |
| Opening r=1 | 0.35 | 0.733 | 0.655 | -0.078 ± 0.159 | 0.795 | 0.837 | +0.042 ± 0.090 |
| Opening r=2 | 0.35 | 0.733 | 0.631 | -0.102 ± 0.182 | 0.795 | 0.84 | +0.044 ± 0.095 |
| Poset-Based Cleaning | 0.35 | 0.734 | 0.738 | +0.004 ± 0.076 | 0.796 | 0.812 | +0.017 ± 0.106 |
| LCC only | 0.4 | 0.726 | 0.678 | -0.048 ± 0.137 | 0.792 | 0.796 | +0.004 ± 0.075 |
| Opening r=1 | 0.4 | 0.726 | 0.648 | -0.077 ± 0.156 | 0.792 | 0.833 | +0.041 ± 0.089 |
| Opening r=2 | 0.4 | 0.726 | 0.625 | -0.101 ± 0.180 | 0.792 | 0.837 | +0.045 ± 0.094 |
| Poset-Based Cleaning | 0.4 | 0.727 | 0.727 | -0.000 ± 0.083 | 0.791 | 0.807 | +0.016 ± 0.106 |
| LCC only | 0.45 | 0.717 | 0.669 | -0.048 ± 0.137 | 0.788 | 0.792 | +0.004 ± 0.079 |
| Opening r=1 | 0.45 | 0.717 | 0.64 | -0.077 ± 0.156 | 0.788 | 0.829 | +0.041 ± 0.089 |
| Opening r=2 | 0.45 | 0.717 | 0.617 | -0.100 ± 0.180 | 0.788 | 0.833 | +0.045 ± 0.097 |
| Poset-Based Cleaning | 0.45 | 0.717 | 0.712 | -0.006 ± 0.103 | 0.786 | 0.806 | +0.019 ± 0.116 |
| LCC only | 0.5 | 0.715 | 0.665 | -0.050 ± 0.138 | 0.786 | 0.788 | +0.002 ± 0.079 |
| Opening r=1 | 0.5 | 0.715 | 0.638 | -0.078 ± 0.156 | 0.786 | 0.827 | +0.040 ± 0.091 |
| Opening r=2 | 0.5 | 0.715 | 0.612 | -0.103 ± 0.181 | 0.786 | 0.829 | +0.043 ± 0.101 |
| Poset-Based Cleaning | 0.5 | 0.713 | 0.687 | -0.027 ± 0.159 | 0.785 | 0.799 | +0.014 ± 0.129 |

---

## A4. Poset-Based Cleaning — pair counts and per-structure extremes

Source: `data/wraparound_experiments/wraparound_v4_eval_cm4/t100/results.csv` and machine `report.md`. Improved / degraded use |Δ| > 0.0001 on the paired row (same definitions as `report.md`).

| Metric | Mean ΔDice (=ΔF1) | Mean ΔPrec | Improved | Degraded | **Net** |
| --- | :---: | :---: | ---: | ---: | ---: |
| Dice / F1 | −0.001 | — | 622 | 251 | **371** |
| Precision | — | +0.010 | 891 | 46 | **845** |

### Top 10 structures by mean ΔDice

| Structure | Dice (art) | F1 (art) | Dice (cl) | F1 (cl) | Δ Dice | Δ F1 | Prec (art) | Prec (cl) | Δ Prec |
| --------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| gluteus_medius_left | 0.806 | 0.806 | 0.846 | 0.846 | +0.040 ± 0.162 | +0.040 ± 0.162 | 0.835 | 0.888 | +0.053 ± 0.201 |
| iliopsoas_right | 0.808 | 0.808 | 0.834 | 0.834 | +0.026 ± 0.093 | +0.026 ± 0.093 | 0.789 | 0.826 | +0.037 ± 0.127 |
| gluteus_medius_right | 0.828 | 0.828 | 0.851 | 0.851 | +0.023 ± 0.119 | +0.023 ± 0.119 | 0.865 | 0.928 | +0.063 ± 0.236 |
| iliopsoas_left | 0.797 | 0.797 | 0.817 | 0.817 | +0.021 ± 0.081 | +0.021 ± 0.081 | 0.801 | 0.834 | +0.033 ± 0.116 |
| hip_left | 0.709 | 0.709 | 0.721 | 0.721 | +0.013 ± 0.071 | +0.013 ± 0.071 | 0.725 | 0.767 | +0.042 ± 0.183 |
| small_bowel | 0.773 | 0.773 | 0.781 | 0.781 | +0.008 ± 0.035 | +0.008 ± 0.035 | 0.817 | 0.844 | +0.026 ± 0.110 |
| spleen | 0.772 | 0.772 | 0.779 | 0.779 | +0.007 ± 0.076 | +0.007 ± 0.076 | 0.797 | 0.812 | +0.015 ± 0.093 |
| iliac_artery_right | 0.646 | 0.646 | 0.653 | 0.653 | +0.007 ± 0.049 | +0.007 ± 0.049 | 0.763 | 0.768 | +0.005 ± 0.035 |
| femur_right | 0.827 | 0.827 | 0.83 | 0.83 | +0.003 ± 0.010 | +0.003 ± 0.010 | 0.853 | 0.86 | +0.007 ± 0.026 |
| iliac_artery_left | 0.646 | 0.646 | 0.649 | 0.649 | +0.003 ± 0.025 | +0.003 ± 0.025 | 0.714 | 0.724 | +0.011 ± 0.094 |

### Bottom 10 structures by mean ΔDice

| Structure | Dice (art) | F1 (art) | Dice (cl) | F1 (cl) | Δ Dice | Δ F1 | Prec (art) | Prec (cl) | Δ Prec |
| --------- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| adrenal_gland_left | 0.714 | 0.714 | 0.686 | 0.686 | -0.028 ± 0.137 | -0.028 ± 0.137 | 0.764 | 0.775 | +0.011 ± 0.057 |
| humerus_right | 0.57 | 0.57 | 0.544 | 0.544 | -0.026 ± 0.117 | -0.026 ± 0.117 | 0.717 | 0.685 | -0.031 ± 0.133 |
| portal_vein_and_splenic_vein | 0.724 | 0.724 | 0.7 | 0.7 | -0.024 ± 0.127 | -0.024 ± 0.127 | 0.768 | 0.783 | +0.016 ± 0.078 |
| duodenum | 0.821 | 0.821 | 0.798 | 0.798 | -0.023 ± 0.136 | -0.023 ± 0.136 | 0.842 | 0.849 | +0.007 ± 0.080 |
| gallbladder | 0.599 | 0.599 | 0.578 | 0.578 | -0.021 ± 0.130 | -0.021 ± 0.130 | 0.785 | 0.787 | +0.002 ± 0.014 |
| scapula_right | 0.288 | 0.288 | 0.269 | 0.269 | -0.019 ± 0.059 | -0.019 ± 0.059 | 0.693 | 0.714 | +0.021 ± 0.063 |
| scapula_left | 0.342 | 0.342 | 0.323 | 0.323 | -0.018 ± 0.084 | -0.018 ± 0.084 | 0.692 | 0.749 | +0.057 ± 0.217 |
| heart | 0.821 | 0.821 | 0.803 | 0.803 | -0.018 ± 0.128 | -0.018 ± 0.128 | 0.936 | 0.938 | +0.002 ± 0.012 |
| esophagus | 0.693 | 0.693 | 0.677 | 0.677 | -0.016 ± 0.111 | -0.016 ± 0.111 | 0.799 | 0.804 | +0.005 ± 0.035 |
| kidney_right | 0.823 | 0.823 | 0.808 | 0.808 | -0.015 ± 0.105 | -0.015 ± 0.105 | 0.843 | 0.843 | -0.000 ± 0.091 |
