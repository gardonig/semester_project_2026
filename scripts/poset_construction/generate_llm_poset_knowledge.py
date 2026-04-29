"""
Generate LLM Poset from Anatomical Knowledge
=============================================

Fills the three poset matrices (vertical, mediolateral, anteroposterior)
using hard-coded anatomical position data derived from expert knowledge.

Position encoding:
  vertical   : higher value = more superior (head = 95, fibula = 8)
  lateral    : positive = patient's LEFT, negative = patient's RIGHT
  anteroposterior : positive = more ANTERIOR, negative = more POSTERIOR

Matrix convention (same as merged_consensus.json):
  matrix[i][j] = 1   → structure i is ABOVE / LEFT-OF / ANTERIOR-TO structure j
  matrix[i][j] = -1  → structure i is NOT above / not left-of / not anterior-to j
  matrix[i][j] = 0   → ambiguous / overlapping
  matrix[i][i] = -1  (diagonal: not strictly above/left-of/anterior-to itself)

Threshold logic:
  If the difference between positions exceeds the threshold → 1 or -1
  Otherwise → 0 (ambiguous / overlapping)
"""

from __future__ import annotations
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Structure order (must match merged_consensus.json exactly)
# ---------------------------------------------------------------------------

STRUCTURES = [
    "brain",                              # 0
    "humerus_left",                       # 1
    "humerus_right",                      # 2
    "lung_left",                          # 3
    "lung_right",                         # 4
    "esophagus",                          # 5
    "spinal_cord",                        # 6
    "heart",                              # 7
    "aorta",                              # 8
    "spleen",                             # 9
    "vertebrae",                          # 10
    "stomach",                            # 11
    "liver",                              # 12
    "adrenal_gland_right",               # 13
    "intervertebral_discs",              # 14
    "pancreas",                           # 15
    "portal_vein_and_splenic_vein",      # 16
    "adrenal_gland_left",                # 17
    "gallbladder",                        # 18
    "kidney_left",                        # 19
    "autochthon_right",                  # 20
    "autochthon_left",                   # 21
    "kidney_right",                       # 22
    "inferior_vena_cava",                # 23
    "duodenum",                           # 24
    "colon",                              # 25
    "small_bowel",                        # 26
    "iliopsoas_left",                     # 27
    "iliopsoas_right",                    # 28
    "iliac_artery_right",                # 29
    "sacrum",                             # 30
    "iliac_artery_left",                 # 31
    "gluteus_medius_right",              # 32
    "iliac_vena_right",                  # 33
    "gluteus_medius_left",               # 34
    "iliac_vena_left",                   # 35
    "hip_right",                          # 36
    "gluteus_minimus_right",             # 37
    "gluteus_minimus_left",              # 38
    "hip_left",                           # 39
    "urinary_bladder",                    # 40
    "gluteus_maximus_right",             # 41
    "gluteus_maximus_left",              # 42
    "prostate",                           # 43
    "thigh_medial_compartment_left",     # 44
    "thigh_medial_compartment_right",    # 45
    "sartorius_right",                    # 46
    "quadriceps_femoris_left",           # 47
    "quadriceps_femoris_right",          # 48
    "sartorius_left",                     # 49
    "thigh_posterior_compartment_right", # 50
    "thigh_posterior_compartment_left",  # 51
    "femur_right",                        # 52
    "femur_left",                         # 53
    "tibia",                              # 54
    "fibula",                             # 55
]

# ---------------------------------------------------------------------------
# Anatomical position data  [vertical, lateral, anteroposterior]
#
# vertical (sup-inf centroid estimate, 8–95):
#   brain=95, humerus≈78, thorax(lung/heart)≈74–77, upper-abd≈59–63,
#   mid-abd≈51–58, pelvis≈33–42, thigh≈20–22, lower-leg=8
#
# lateral (patient's left=+, patient's right=−, midline=0):
#   far-left arms/hip=±3, thoracic sides=±2, paraspinal/adrenal=±1.5
#
# anteroposterior (anterior=+, posterior=−):
#   quadriceps/bladder=+2, stomach/liver/heart=+1, colon/femur=0,
#   pancreas/iliopsoas/IVC=−1, kidney/adrenal=−2, spine/autochthon=−3
# ---------------------------------------------------------------------------

POSITIONS = [
    # [vert, lat,   ap ]
    [95,  0.0,  0.0],   # 0  brain
    [78,  3.0,  0.0],   # 1  humerus_left
    [78, -3.0,  0.0],   # 2  humerus_right
    [77,  2.0, -1.0],   # 3  lung_left
    [77, -2.0, -1.0],   # 4  lung_right
    [73,  0.0, -2.0],   # 5  esophagus
    [65,  0.0, -3.0],   # 6  spinal_cord
    [74,  1.0,  1.0],   # 7  heart
    [65,  0.0, -1.0],   # 8  aorta
    [62,  2.5, -1.0],   # 9  spleen
    [65,  0.0, -3.0],   # 10 vertebrae
    [60,  1.0,  1.0],   # 11 stomach
    [63, -2.0,  1.0],   # 12 liver
    [62, -1.5, -2.0],   # 13 adrenal_gland_right
    [60,  0.0, -3.0],   # 14 intervertebral_discs
    [60,  0.0, -1.0],   # 15 pancreas
    [61,  0.0, -1.0],   # 16 portal_vein_and_splenic_vein
    [62,  1.5, -2.0],   # 17 adrenal_gland_left
    [59, -1.5,  1.0],   # 18 gallbladder
    [61,  1.5, -2.0],   # 19 kidney_left
    [63, -1.0, -3.0],   # 20 autochthon_right
    [63,  1.0, -3.0],   # 21 autochthon_left
    [60, -1.5, -2.0],   # 22 kidney_right
    [56, -0.5, -2.0],   # 23 inferior_vena_cava
    [57, -0.5, -1.0],   # 24 duodenum
    [52,  0.0,  0.0],   # 25 colon
    [51,  0.0,  1.0],   # 26 small_bowel
    [47,  1.5, -1.0],   # 27 iliopsoas_left
    [47, -1.5, -1.0],   # 28 iliopsoas_right
    [42, -1.5, -1.0],   # 29 iliac_artery_right
    [38,  0.0, -3.0],   # 30 sacrum
    [42,  1.5, -1.0],   # 31 iliac_artery_left
    [40, -2.0, -2.0],   # 32 gluteus_medius_right
    [42, -1.5, -1.0],   # 33 iliac_vena_right
    [40,  2.0, -2.0],   # 34 gluteus_medius_left
    [42,  1.5, -1.0],   # 35 iliac_vena_left
    [38, -2.5,  0.0],   # 36 hip_right
    [39, -2.0, -1.5],   # 37 gluteus_minimus_right
    [39,  2.0, -1.5],   # 38 gluteus_minimus_left
    [38,  2.5,  0.0],   # 39 hip_left
    [35,  0.0,  2.0],   # 40 urinary_bladder
    [37, -2.5, -3.0],   # 41 gluteus_maximus_right
    [37,  2.5, -3.0],   # 42 gluteus_maximus_left
    [33,  0.0,  0.0],   # 43 prostate
    [22,  1.5,  0.0],   # 44 thigh_medial_compartment_left
    [22, -1.5,  0.0],   # 45 thigh_medial_compartment_right
    [21, -2.0,  1.0],   # 46 sartorius_right
    [20,  2.0,  2.0],   # 47 quadriceps_femoris_left
    [20, -2.0,  2.0],   # 48 quadriceps_femoris_right
    [21,  2.0,  1.0],   # 49 sartorius_left
    [22, -2.0, -2.0],   # 50 thigh_posterior_compartment_right
    [22,  2.0, -2.0],   # 51 thigh_posterior_compartment_left
    [22, -2.0,  0.0],   # 52 femur_right
    [22,  2.0,  0.0],   # 53 femur_left
    [ 8,  0.0,  1.0],   # 54 tibia
    [ 8,  0.5, -1.0],   # 55 fibula
]

# ---------------------------------------------------------------------------
# Thresholds: differences ≤ threshold → 0 (ambiguous / overlapping)
# ---------------------------------------------------------------------------

VERT_THRESH = 5.0   # ~1–2 vertebral levels of uncertainty
LAT_THRESH  = 0.8   # ~half a unit lateral difference required
AP_THRESH   = 0.8   # same

# ---------------------------------------------------------------------------
# Matrix computation
# ---------------------------------------------------------------------------

def compare(val_i: float, val_j: float, threshold: float) -> int:
    diff = val_i - val_j
    if diff > threshold:
        return 1
    elif diff < -threshold:
        return -1
    return 0


def build_matrices() -> tuple[list, list, list]:
    n = len(STRUCTURES)
    assert len(POSITIONS) == n, f"Expected {n} positions, got {len(POSITIONS)}"

    def empty() -> list:
        return [[0] * n for _ in range(n)]

    vert = empty()
    lat  = empty()
    ap   = empty()

    for i in range(n):
        for j in range(n):
            if i == j:
                vert[i][j] = lat[i][j] = ap[i][j] = -1
            else:
                vert[i][j] = compare(POSITIONS[i][0], POSITIONS[j][0], VERT_THRESH)
                lat[i][j]  = compare(POSITIONS[i][1], POSITIONS[j][1], LAT_THRESH)
                ap[i][j]   = compare(POSITIONS[i][2], POSITIONS[j][2], AP_THRESH)

    return vert, lat, ap


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    vert, lat, ap = build_matrices()

    result = {
        "structures": [{"name": s} for s in STRUCTURES],
        "matrix_vertical": vert,
        "matrix_mediolateral": lat,
        "matrix_anteroposterior": ap,
    }

    out = Path(__file__).resolve().parent.parent.parent / "data" / "posets" / "llm_sessions" / "llm_claude.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    # Quick sanity check printout
    n = len(STRUCTURES)
    total = n * (n - 1)
    for axis_name, matrix in [("vertical", vert), ("mediolateral", lat), ("anteroposterior", ap)]:
        ones   = sum(matrix[i][j] ==  1 for i in range(n) for j in range(n) if i != j)
        minus1 = sum(matrix[i][j] == -1 for i in range(n) for j in range(n) if i != j)
        zeros  = sum(matrix[i][j] ==  0 for i in range(n) for j in range(n) if i != j)
        print(f"{axis_name:25s}  1={ones:4d}  -1={minus1:4d}  0={zeros:4d}  (total off-diag={total})")

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
