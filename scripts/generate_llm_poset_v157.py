"""
Generate LLM Poset for TotalSegmentator v1.5.7  —  strict non-overlap semantics
================================================================================

matrix[i][j] = 1  iff  ENTIRE structure i is strictly above/left-of/anterior-to
                        ENTIRE structure j  (= ranges do not overlap at all)
matrix[i][j] = -1 iff  ENTIRE structure j is strictly above/left-of/anterior-to i
matrix[i][j] = 0  iff  the ranges overlap (ambiguous / indeterminate)
matrix[i][i] = -1 (diagonal convention)

Coordinate system
-----------------
  vertical      :  vmin (most inferior) … vmax (most superior)   scale 0–100
  lateral       :  lmin (most rightward, negative) … lmax (most leftward, positive)
  anteroposterior: apmin (most posterior, negative) … apmax (most anterior, positive)

  matrix_vertical[i][j]      = 1  iff  vmin[i] > vmax[j]
  matrix_mediolateral[i][j]  = 1  iff  lmin[i] > lmax[j]
  matrix_anteroposterior[i][j] = 1  iff  apmin[i] > apmax[j]

Output: data/posets/llm_sessions/llm_claude_v157.json
"""

from __future__ import annotations
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Structure order (alphabetical, matches amos_0102 ls output)
# ---------------------------------------------------------------------------

STRUCTURES = [
    "adrenal_gland_left", "adrenal_gland_right", "aorta",
    "autochthon_left", "autochthon_right", "brain",
    "clavicula_left", "clavicula_right", "colon", "duodenum",
    "esophagus", "face", "femur_left", "femur_right", "gallbladder",
    "gluteus_maximus_left", "gluteus_maximus_right",
    "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right",
    "heart_atrium_left", "heart_atrium_right", "heart_myocardium",
    "heart_ventricle_left", "heart_ventricle_right",
    "hip_left", "hip_right", "humerus_left", "humerus_right",
    "iliac_artery_left", "iliac_artery_right",
    "iliac_vena_left", "iliac_vena_right",
    "iliopsoas_left", "iliopsoas_right",
    "inferior_vena_cava", "kidney_left", "kidney_right", "liver",
    "lung_lower_lobe_left", "lung_lower_lobe_right",
    "lung_middle_lobe_right", "lung_upper_lobe_left", "lung_upper_lobe_right",
    "pancreas", "portal_vein_and_splenic_vein", "pulmonary_artery",
    "rib_left_1",  "rib_left_10",  "rib_left_11",  "rib_left_12",
    "rib_left_2",  "rib_left_3",   "rib_left_4",   "rib_left_5",
    "rib_left_6",  "rib_left_7",   "rib_left_8",   "rib_left_9",
    "rib_right_1", "rib_right_10", "rib_right_11", "rib_right_12",
    "rib_right_2", "rib_right_3",  "rib_right_4",  "rib_right_5",
    "rib_right_6", "rib_right_7",  "rib_right_8",  "rib_right_9",
    "sacrum", "scapula_left", "scapula_right", "small_bowel",
    "spleen", "stomach", "trachea", "urinary_bladder",
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4",
    "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "vertebrae_T1",  "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
    "vertebrae_T2",  "vertebrae_T3",  "vertebrae_T4",  "vertebrae_T5",
    "vertebrae_T6",  "vertebrae_T7",  "vertebrae_T8",  "vertebrae_T9",
]

assert len(STRUCTURES) == 104

# ---------------------------------------------------------------------------
# Bounding ranges per structure
#   Each entry: [vmin, vmax,  lmin, lmax,  apmin, apmax]
#
# Vertebral-level reference (approx centre on 0–100 scale):
#   skull=100, C1=92, C7=86, T1=85, T2=83, T3=81 ... T12=63, L1=61 ... L5=53, sacrum~47
#   (thoracic spaced 2 units, cervical/lumbar also 2 units)
#
# Lateral: −5=far patient-right,  +5=far patient-left
# AP:      −5=far posterior,       +5=far anterior
# ---------------------------------------------------------------------------

# Helper: rib range centred on its thoracic vertebra centre ± 0.6
# T1 centre=84.5, T2=82.5, ..., T12=62.5   → rib_N centre = 84.5 − (N−1)×2
def _rib(n: int, side: str) -> list[float]:
    centre = 84.5 - (n - 1) * 2.0
    v = [centre - 0.6, centre + 0.6]
    # Lateral: rib spans from spine attachment (~0.5) out to lateral chest wall (~3.5)
    # then curves anteriorly back to sternum (lat 0) for ribs 1-7.
    # Use full range including sternum crossing:
    l = [0.0, 3.5] if side == "L" else [-3.5, 0.0]
    # AP: true ribs (1-7) reach the sternum; false ribs (8-10) shorter; floating (11-12) don't reach front
    if n <= 7:
        ap = [-3.5, 2.0]
    elif n <= 10:
        ap = [-3.5, 0.5]
    else:
        ap = [-3.5, -1.0]
    return v + l + ap

RANGES: dict[str, list[float]] = {
    # ── Soft-tissue head ────────────────────────────────────────────────────
    "brain":          [89, 100, -3.0,  3.0, -4.0,  2.0],
    "face":           [82,  96, -2.5,  2.5,  0.0,  5.0],

    # ── Neck ────────────────────────────────────────────────────────────────
    # Trachea: C4 (larynx) → T4 (carina).  Upper bound 88.5 so brain vmin=89 clears it.
    "trachea":        [78,  88.5, -0.5,  0.5, -0.5,  2.0],
    # Esophagus AP upper bound kept at −1.1 so heart_ventricle_right (apmin=−1.0) is strictly anterior.
    "esophagus":      [65,  88,   -0.5,  0.5, -4.0, -1.1],

    # ── Shoulder girdle / upper extremity ───────────────────────────────────
    "clavicula_left":  [83, 86,  0.0,  3.5, -0.5,  2.0],
    "clavicula_right": [83, 86, -3.5,  0.0, -0.5,  2.0],
    "scapula_left":    [74, 85,  0.5,  4.0, -4.5, -1.5],
    "scapula_right":   [74, 85, -4.0, -0.5, -4.5, -1.5],
    "humerus_left":    [68, 85,  1.5,  4.5, -2.0,  2.0],
    "humerus_right":   [68, 85, -4.5, -1.5, -2.0,  2.0],

    # ── Thoracic viscera ────────────────────────────────────────────────────
    "lung_upper_lobe_left":   [75, 87,  0.5,  3.5, -3.5,  0.5],
    "lung_upper_lobe_right":  [75, 87, -3.5, -0.5, -3.5,  0.5],
    "lung_middle_lobe_right": [73, 80, -3.5, -0.5, -2.0,  0.5],
    "lung_lower_lobe_left":   [62, 78,  0.5,  3.5, -4.0, -0.5],
    "lung_lower_lobe_right":  [62, 78, -3.5, -0.5, -4.0, -0.5],
    "heart_atrium_left":      [71, 78, -0.5,  2.0, -4.0, -1.5],
    "heart_atrium_right":     [72, 78, -2.0,  0.5, -3.0, -0.5],
    "heart_myocardium":       [68, 80, -2.0,  2.0, -4.0,  1.0],
    "heart_ventricle_left":   [68, 75, -0.5,  2.0, -2.5,  1.0],
    "heart_ventricle_right":  [68, 77, -1.5,  0.5, -1.0,  2.0],
    "pulmonary_artery":       [76, 83, -1.5,  1.5, -1.0,  2.0],
    "aorta":                  [54, 80, -1.5,  1.0, -3.5,  1.0],

    # ── Ribs (auto-generated) ──────────────────────────────────────────────
    **{f"rib_left_{n}":  _rib(n, "L") for n in range(1, 13)},
    **{f"rib_right_{n}": _rib(n, "R") for n in range(1, 13)},

    # ── Upper abdominal viscera ─────────────────────────────────────────────
    "liver":                       [60, 73, -3.5,  1.0, -2.0,  2.5],
    "spleen":                      [60, 70,  1.5,  3.5, -4.0, -1.0],
    "stomach":                     [57, 70, -1.0,  2.5, -1.5,  2.0],
    "gallbladder":                 [59, 66, -2.5, -0.5,  0.0,  3.0],
    "adrenal_gland_left":          [63, 68,  0.5,  2.0, -4.0, -1.5],
    "adrenal_gland_right":         [63, 68, -2.0, -0.5, -4.0, -1.5],
    "portal_vein_and_splenic_vein":[60, 67, -1.0,  2.0, -3.0, -0.5],
    "pancreas":                    [58, 65, -1.5,  2.0, -3.5, -0.5],
    "kidney_left":                 [58, 65,  1.0,  2.5, -4.0, -1.5],
    "kidney_right":                [57, 65, -2.5, -1.0, -4.0, -1.5],
    "duodenum":                    [57, 65, -2.0,  0.5, -3.5,  0.5],
    "inferior_vena_cava":          [54, 72, -1.5,  0.0, -3.5, -1.5],

    # ── Paraspinal muscles (span thoracolumbar) ─────────────────────────────
    "autochthon_left":   [54, 86,  0.5,  2.5, -5.0, -2.0],
    "autochthon_right":  [54, 86, -2.5, -0.5, -5.0, -2.0],

    # ── Mid / lower abdominal viscera ───────────────────────────────────────
    "colon":      [42, 68, -3.0,  3.0, -2.5,  2.5],
    "small_bowel":[47, 64, -2.5,  2.5, -1.5,  2.5],

    # ── Pelvic structures ──────────────────────────────────────────────────
    "iliopsoas_left":        [34, 63,  0.0,  2.0, -3.0,  0.5],
    "iliopsoas_right":       [34, 63, -2.0,  0.0, -3.0,  0.5],
    "iliac_artery_left":     [43, 56,  0.0,  2.0, -3.5, -0.5],
    "iliac_artery_right":    [43, 56, -2.0,  0.0, -3.5, -0.5],
    "iliac_vena_left":       [43, 56,  0.0,  2.0, -3.5, -0.5],
    "iliac_vena_right":      [43, 56, -2.0,  0.0, -3.5, -0.5],
    "gluteus_maximus_left":  [35, 52,  1.0,  3.5, -5.0, -2.0],
    "gluteus_maximus_right": [35, 52, -3.5, -1.0, -5.0, -2.0],
    "gluteus_medius_left":   [38, 52,  1.5,  4.0, -4.0, -1.0],
    "gluteus_medius_right":  [38, 52, -4.0, -1.5, -4.0, -1.0],
    "gluteus_minimus_left":  [36, 48,  1.5,  4.0, -3.0, -0.5],
    "gluteus_minimus_right": [36, 48, -4.0, -1.5, -3.0, -0.5],
    "hip_left":              [37, 43,  1.5,  3.5, -2.0,  2.0],
    "hip_right":             [37, 43, -3.5, -1.5, -2.0,  2.0],
    "urinary_bladder":       [33, 40, -1.5,  1.5,  0.5,  3.5],
    "sacrum":                [42, 52, -1.5,  1.5, -5.0, -2.0],

    # ── Lower extremity ─────────────────────────────────────────────────────
    "femur_left":  [14, 43,  1.0,  3.0, -2.0,  2.0],
    "femur_right": [14, 43, -3.0, -1.0, -2.0,  2.0],

    # ── Individual vertebrae ────────────────────────────────────────────────
    # Each band has a 0.1-unit gap to the next so adjacent vertebrae are STRICTLY ordered.
    # Cervical (1-unit centres): C1=91.65, C2=90.65, ..., C7=85.65
    # Thoracic (2-unit centres): T1=84.65, T2=82.65, ..., T12=62.65
    # Lumbar   (2-unit centres): L1=60.65, L2=58.65, ..., L5=52.65
    # Band half-width 0.45  → each band is [centre−0.45, centre+0.45], gap = 0.1
    "vertebrae_C1": [91.2, 92.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_C2": [90.2, 91.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_C3": [89.2, 90.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_C4": [88.2, 89.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_C5": [87.2, 88.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_C6": [86.2, 87.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_C7": [85.2, 86.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T1": [84.2, 85.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T2": [82.2, 84.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T3": [80.2, 82.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T4": [78.2, 80.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T5": [76.2, 78.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T6": [74.2, 76.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T7": [72.2, 74.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T8": [70.2, 72.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T9": [68.2, 70.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T10":[66.2, 68.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T11":[64.2, 66.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_T12":[62.2, 64.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_L1": [60.2, 62.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_L2": [58.2, 60.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_L3": [56.2, 58.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_L4": [54.2, 56.1, -0.5, 0.5, -4.5, -1.5],
    "vertebrae_L5": [52.2, 54.1, -0.5, 0.5, -4.5, -1.5],
}

assert set(STRUCTURES) == set(RANGES), \
    f"Missing: {set(STRUCTURES) - set(RANGES)}\nExtra: {set(RANGES) - set(STRUCTURES)}"

# ---------------------------------------------------------------------------
# Matrix computation
# ---------------------------------------------------------------------------

def build_matrices():
    n = len(STRUCTURES)
    r = [RANGES[s] for s in STRUCTURES]

    def empty():
        return [[0] * n for _ in range(n)]

    vert = empty()
    lat  = empty()
    ap   = empty()

    for i in range(n):
        for j in range(n):
            if i == j:
                vert[i][j] = lat[i][j] = ap[i][j] = -1
                continue
            vi, vj = r[i], r[j]
            # vertical
            if vi[0] > vj[1]:   vert[i][j] =  1   # i entirely above j
            elif vi[1] < vj[0]: vert[i][j] = -1   # i entirely below j
            # mediolateral
            if vi[2] > vj[3]:   lat[i][j] =  1    # i entirely left of j
            elif vi[3] < vj[2]: lat[i][j] = -1    # i entirely right of j
            # anteroposterior
            if vi[4] > vj[5]:   ap[i][j] =  1     # i entirely anterior to j
            elif vi[5] < vj[4]: ap[i][j] = -1     # i entirely posterior to j

    return vert, lat, ap

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    vert, lat, ap = build_matrices()
    n = len(STRUCTURES)
    total = n * (n - 1)

    result = {
        "structures": [
            {"name": s, "com_vertical": 0.0, "com_lateral": 0.0, "com_anteroposterior": 0.0}
            for s in STRUCTURES
        ],
        "matrix_vertical":        vert,
        "matrix_mediolateral":    lat,
        "matrix_anteroposterior": ap,
    }

    out = Path(__file__).resolve().parent.parent / "data" / "posets" / "llm_sessions" / "llm_claude_v157.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    for name, matrix in [("vertical", vert), ("mediolateral", lat), ("anteroposterior", ap)]:
        ones   = sum(matrix[i][j] ==  1 for i in range(n) for j in range(n) if i != j)
        minus1 = sum(matrix[i][j] == -1 for i in range(n) for j in range(n) if i != j)
        zeros  = sum(matrix[i][j] ==  0 for i in range(n) for j in range(n) if i != j)
        print(f"{name:25s}  1={ones:5d}  -1={minus1:5d}  0={zeros:5d}  (total={total})")

    print(f"\nSaved {n} structures → {out}")


if __name__ == "__main__":
    main()
