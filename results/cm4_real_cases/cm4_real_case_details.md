# Real cleaning cases (coronal, square pixels)

- **good d=0.20**: `s0186` / `heart_to_kidney` / `d020_r100` / `gluteus_medius_right` ΔDice=+0.60676 — `conflict_prefer_i_remove_j`
- **good d=0.20**: `s0175` / `heart_to_kidney` / `d020_r075` / `gluteus_medius_left` ΔDice=+0.58414 — `conflict_prefer_i_remove_j`
- **good d=0.20**: `s0236` / `heart_to_kidney` / `d020_r100` / `iliopsoas_left` ΔDice=+0.41517 — `conflict_prefer_i_remove_j`
- **bad d=0.50**: `s0175` / `brain_to_heart` / `d050_r050` / `lung_right` ΔDice=-0.97620 — `conflict_prefer_j_remove_i`
- **bad d=0.50**: `s0175` / `brain_to_heart` / `d050_r075` / `lung_left` ΔDice=-0.96248 — `conflict_prefer_j_remove_i`
- **bad d=0.50**: `s0175` / `brain_to_heart` / `d050_r075` / `liver` ΔDice=-0.93992 — `conflict_prefer_j_remove_i`
- **bad (min d with ΔDice<0, d=0.20)**: `s0175` / `heart_to_kidney` / `d020_r075` / `humerus_right` ΔDice=-0.32711 — `normal_remove_i_below_j`
