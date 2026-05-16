# Real cleaning cases — subject s0250 (coronal; SI axis 2× screen stretch)

Good rows: d≤0.35, ΔDice>0, ranked by largest `vox_removed_pc` (up to 6; editorial rank skip when enough candidates). Bad rows: d≤0.5, ΔDice<0, same rule (up to 6). Good/bad pools use ΔDice (CSV has no delta_f1 — re-run evaluate_cleaning_methods).

Figure: one coronal MRI slice; mask overlays are **A/P silhouettes** on LR×SI. For this subject (`s0250`), LR and SI use **equal** on-screen pixel scale. **Trusted anchor (purple)** = constraint partner; **red** = removed TP, **green** = removed FP.

- **good d≤0.35 (s0250)** — targeted structure `small_bowel` @ `s0250` / `heart_to_kidney` / `d035_r100`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `small_bowel`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.80575 → 0.93180 (Δ +0.12605)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 101,894 total; TP (red) 0; FP (green) 101,894
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 101,894 removed; TP 0; FP 101,894

- **good d≤0.35 (s0250)** — targeted structure `iliopsoas_right` @ `s0250` / `heart_to_kidney` / `d030_r100`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `iliopsoas_right`
  - **Step (caption)**: `conflict_prefer_i_remove_j`
  - **F1** (full 3D mask vs GT): 0.39279 → 0.90872 (Δ +0.51594)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 44,798 total; TP (red) 0; FP (green) 44,798
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 44,798 removed; TP 0; FP 44,798

- **good d≤0.35 (s0250)** — targeted structure `spleen` @ `s0250` / `kidney_to_hip` / `d035_r100`
  - **Trusted anchor (purple silhouette)**: `iliac_artery_left` · poset pair `spleen` / `iliac_artery_left`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.16798 → 0.87872 (Δ +0.71074)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 34,120 total; TP (red) 0; FP (green) 34,120
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 34,120 removed; TP 0; FP 34,120

- **good d≤0.35 (s0250)** — targeted structure `iliopsoas_left` @ `s0250` / `heart_to_kidney` / `d035_r075`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `iliopsoas_left`
  - **Step (caption)**: `conflict_prefer_i_remove_j`
  - **F1** (full 3D mask vs GT): 0.45692 → 0.76158 (Δ +0.30465)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 32,146 total; TP (red) 0; FP (green) 32,146
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 32,146 removed; TP 0; FP 32,146

- **good d≤0.35 (s0250)** — targeted structure `pancreas` @ `s0250` / `kidney_to_hip` / `d025_r100`
  - **Trusted anchor (purple silhouette)**: `gluteus_medius_left` · poset pair `pancreas` / `gluteus_medius_left`
  - **Step (caption)**: `normal_remove_i_below_j`
  - **F1** (full 3D mask vs GT): 0.56086 → 0.68037 (Δ +0.11951)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 3,504 total; TP (red) 0; FP (green) 3,504
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 3,504 removed; TP 0; FP 3,504

