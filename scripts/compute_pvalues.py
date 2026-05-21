"""Compute two-sided Wilcoxon signed-rank p-values for all report tables."""
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

CSV = "data/wraparound_experiments/wraparound_v4_eval_cm4/t100/results.csv"
df = pd.read_csv(CSV)
df = df[df["has_gt"] == True].copy()

# Structures in the report tables
TOP5_STRUCTURES = [
    "gluteus_medius_left", "iliopsoas_right", "gluteus_medius_right",
    "iliopsoas_left", "hip_left",
]
BOT5_STRUCTURES = [
    "adrenal_gland_left", "humerus_right", "portal_vein_and_splenic_vein",
    "duodenum", "gallbladder",
]
TOP10_STRUCTURES = TOP5_STRUCTURES + [
    "small_bowel", "spleen", "iliac_artery_right", "femur_right", "iliac_artery_left",
]
BOT10_STRUCTURES = BOT5_STRUCTURES + [
    "scapula_right", "scapula_left", "heart", "esophagus", "kidney_right",
]

CROP_MAP = {
    "brain_to_heart": "Brain$\\rightarrow$Heart",
    "heart_to_kidney": "Heart$\\rightarrow$Kidney",
    "kidney_to_hip":   "Kidney$\\rightarrow$Hip",
}


def fmt_p(p):
    """Format p-value for LaTeX."""
    if p >= 0.05:
        return "n.s."
    if p >= 0.001:
        exp = int(np.floor(np.log10(p)))
        mant = p / 10**exp
        return f"${mant:.1f} \\times 10^{{{exp}}}$"
    exp = int(np.floor(np.log10(p)))
    mant = p / 10**exp
    if abs(mant - 1.0) < 0.05:
        return f"$10^{{{exp}}}$"
    return f"${mant:.1f} \\times 10^{{{exp}}}$"


def wilcox_p(vals):
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    nonzero = vals[vals != 0]
    if len(nonzero) < 5:
        return 1.0
    try:
        _, p = wilcoxon(nonzero, alternative="two-sided")
        return p
    except Exception:
        return 1.0


# --------------------------------------------------------------------------
# 1. Shift fraction table (all d values)
# --------------------------------------------------------------------------
print("=" * 60)
print("TABLE: shift fraction (tab:shift_effect)")
print("=" * 60)
for d in sorted(df["d_frac"].unique()):
    sub = df[df["d_frac"] == d]
    p_f1   = wilcox_p(sub["delta_pc"])
    p_prec = wilcox_p(sub["delta_prec_pc"])
    print(f"d={d:.2f}  p_F1={fmt_p(p_f1)}  p_Prec={fmt_p(p_prec)}")

# --------------------------------------------------------------------------
# 2. Intensity table
# --------------------------------------------------------------------------
print()
print("=" * 60)
print("TABLE: ghost intensity (tab:intensity_effect)")
print("=" * 60)
for r in sorted(df["r_val"].unique()):
    sub = df[df["r_val"] == r]
    p_f1   = wilcox_p(sub["delta_pc"])
    p_prec = wilcox_p(sub["delta_prec_pc"])
    print(f"r={r:.2f}  p_F1={fmt_p(p_f1)}  p_Prec={fmt_p(p_prec)}")

# --------------------------------------------------------------------------
# 3. Region table
# --------------------------------------------------------------------------
print()
print("=" * 60)
print("TABLE: region (tab:region_effect)")
print("=" * 60)
for crop, label in CROP_MAP.items():
    sub = df[df["crop"] == crop]
    p_f1   = wilcox_p(sub["delta_pc"])
    p_prec = wilcox_p(sub["delta_prec_pc"])
    print(f"{label}  p_F1={fmt_p(p_f1)}  p_Prec={fmt_p(p_prec)}")

# --------------------------------------------------------------------------
# 4. Per-structure improvements / degradations (main text top/bottom 5)
# --------------------------------------------------------------------------
def struct_pvals(structures, label):
    print()
    print("=" * 60)
    print(f"TABLE: {label}")
    print("=" * 60)
    for s in structures:
        sub = df[df["structure"] == s]
        if len(sub) == 0:
            print(f"  {s}: NOT FOUND")
            continue
        p_f1   = wilcox_p(sub["delta_pc"])
        p_prec = wilcox_p(sub["delta_prec_pc"])
        print(f"  {s}: p_F1={fmt_p(p_f1)}  p_Prec={fmt_p(p_prec)}")

struct_pvals(TOP5_STRUCTURES,  "tab:improvements (top 5)")
struct_pvals(BOT5_STRUCTURES,  "tab:degradations (bottom 5)")
struct_pvals(TOP10_STRUCTURES, "appendix tab:top_10")
struct_pvals(BOT10_STRUCTURES, "appendix tab:bottom_10")
