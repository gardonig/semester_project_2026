# Real cleaning cases — subject s0175 (coronal; SI axis 2× screen stretch)

Good rows: d≤0.35, ΔDice>0, ranked by largest `vox_removed_pc` (figure uses ranks [0, 2, 3, 4, 5, 6] of that list, skipping the 2nd). Bad rows: d≤0.5, ΔDice<0, same ranking (figure uses ranks [1, 2, 3, 4, 5, 6], skipping the 1st).

- **good d≤0.35 (s0175)**: `s0175` / `heart_to_kidney` / `d035_r100` / `small_bowel` ΔDice=+0.09963 — `conflict_prefer_i_remove_j`
- **good d≤0.35 (s0175)**: `s0175` / `heart_to_kidney` / `d025_r100` / `gluteus_medius_left` ΔDice=+0.13270 — `conflict_prefer_i_remove_j`
- **good d≤0.35 (s0175)**: `s0175` / `heart_to_kidney` / `d020_r100` / `humerus_left` ΔDice=+0.40517 — `conflict_prefer_j_remove_i`
- **good d≤0.35 (s0175)**: `s0175` / `heart_to_kidney` / `d035_r075` / `iliopsoas_right` ΔDice=+0.15126 — `normal_remove_j_above_i`
- **good d≤0.35 (s0175)**: `s0175` / `kidney_to_hip` / `d035_r100` / `femur_right` ΔDice=+0.04832 — `normal_remove_j_above_i`
- **good d≤0.35 (s0175)**: `s0175` / `heart_to_kidney` / `d030_r075` / `iliopsoas_left` ΔDice=+0.02456 — `conflict_prefer_i_remove_j`
- **bad d≤0.5 (s0175)**: `s0175` / `brain_to_heart` / `d050_r100` / `lung_left` ΔDice=-0.96204 — `conflict_prefer_j_remove_i`
- **bad d≤0.5 (s0175)**: `s0175` / `heart_to_kidney` / `d050_r100` / `liver` ΔDice=-0.30598 — `conflict_prefer_j_remove_i`
- **bad d≤0.5 (s0175)**: `s0175` / `brain_to_heart` / `d050_r050` / `humerus_right` ΔDice=-0.73796 — `conflict_prefer_j_remove_i`
- **bad d≤0.5 (s0175)**: `s0175` / `heart_to_kidney` / `d045_r075` / `kidney_right` ΔDice=-0.19109 — `conflict_prefer_j_remove_i`
- **bad d≤0.5 (s0175)**: `s0175` / `brain_to_heart` / `d045_r075` / `scapula_right` ΔDice=-0.20041 — `conflict_prefer_j_remove_i`
- **bad d≤0.5 (s0175)**: `s0175` / `heart_to_kidney` / `d040_r075` / `inferior_vena_cava` ΔDice=-0.20389 — `conflict_prefer_j_remove_i`
