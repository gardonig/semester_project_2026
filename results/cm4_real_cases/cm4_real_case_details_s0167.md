# Real cleaning cases — subject s0167 (coronal; SI axis 2× screen stretch)

Good rows: d≤0.35, ΔDice>0, ranked by largest `vox_removed_pc` (up to 6; editorial rank skip when enough candidates). Bad rows: d≤0.5, ΔDice<0, same rule (up to 6).

- **good d≤0.35 (s0167)**: `s0167` / `heart_to_kidney` / `d035_r100` / `small_bowel` ΔDice=+0.22339 — `normal_remove_j_above_i`
- **good d≤0.35 (s0167)**: `s0167` / `kidney_to_hip` / `d035_r100` / `liver` ΔDice=+0.08677 — `normal_remove_i_below_j`
- **good d≤0.35 (s0167)**: `s0167` / `heart_to_kidney` / `d025_r100` / `iliopsoas_left` ΔDice=+0.16348 — `normal_remove_j_above_i`
- **good d≤0.35 (s0167)**: `s0167` / `kidney_to_hip` / `d030_r100` / `spleen` ΔDice=+0.11414 — `normal_remove_i_below_j`
- **good d≤0.35 (s0167)**: `s0167` / `heart_to_kidney` / `d030_r100` / `iliopsoas_right` ΔDice=+0.08543 — `normal_remove_j_above_i`
- **bad d≤0.5 (s0167)**: `s0167` / `heart_to_kidney` / `d045_r100` / `kidney_right` ΔDice=-0.89166 — `conflict_prefer_j_remove_i`
- **bad d≤0.5 (s0167)**: `s0167` / `heart_to_kidney` / `d035_r075` / `duodenum` ΔDice=-0.85789 — `conflict_prefer_j_remove_i`
- **bad d≤0.5 (s0167)**: `s0167` / `heart_to_kidney` / `d040_r050` / `gallbladder` ΔDice=-0.84098 — `conflict_prefer_j_remove_i`
- **bad d≤0.5 (s0167)**: `s0167` / `heart_to_kidney` / `d040_r100` / `portal_vein_and_splenic_vein` ΔDice=-0.53763 — `conflict_prefer_j_remove_i`
