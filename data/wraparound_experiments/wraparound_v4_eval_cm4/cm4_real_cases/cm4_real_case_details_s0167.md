# Real cleaning cases — subject s0167 (coronal; SI axis 2× screen stretch)

Good rows: d≤0.35, ΔDice>0, ranked by largest `vox_removed_pc` (up to 6; editorial rank skip when enough candidates). Bad rows: d≤0.5, ΔDice<0, same rule (up to 6). Good/bad pools use ΔDice (CSV has no delta_f1 — re-run evaluate_cleaning_methods).

Figure: one coronal MRI slice; mask overlays are **A/P silhouettes** on LR×SI. For this subject (`s0167`), LR and SI use **equal** on-screen pixel scale. **Trusted anchor (purple)** = constraint partner; **red** = removed TP, **green** = removed FP.

- **good d≤0.35 (s0167)** — targeted structure `small_bowel` @ `s0167` / `heart_to_kidney` / `d035_r100`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `small_bowel`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.58832 → 0.81170 (Δ +0.22339)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 59,780 total; TP (red) 0; FP (green) 59,780
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 59,780 removed; TP 0; FP 59,780

- **good d≤0.35 (s0167)** — targeted structure `liver` @ `s0167` / `kidney_to_hip` / `d035_r100`
  - **Trusted anchor (purple silhouette)**: `gluteus_minimus_left` · poset pair `liver` / `gluteus_minimus_left`
  - **Step (caption)**: `normal_remove_i_below_j`
  - **F1** (full 3D mask vs GT): 0.84741 → 0.93418 (Δ +0.08677)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 43,806 total; TP (red) 0; FP (green) 43,806
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 43,806 removed; TP 0; FP 43,806

- **good d≤0.35 (s0167)** — targeted structure `iliopsoas_left` @ `s0167` / `heart_to_kidney` / `d025_r100`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `iliopsoas_left`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.61051 → 0.77398 (Δ +0.16348)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 6,612 total; TP (red) 0; FP (green) 6,612
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 6,288 removed; TP 0; FP 6,288

- **good d≤0.35 (s0167)** — targeted structure `spleen` @ `s0167` / `kidney_to_hip` / `d030_r100`
  - **Trusted anchor (purple silhouette)**: `gluteus_minimus_left` · poset pair `spleen` / `gluteus_minimus_left`
  - **Step (caption)**: `normal_remove_i_below_j`
  - **F1** (full 3D mask vs GT): 0.76906 → 0.88320 (Δ +0.11414)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 4,798 total; TP (red) 0; FP (green) 4,798
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 4,798 removed; TP 0; FP 4,798

- **good d≤0.35 (s0167)** — targeted structure `iliopsoas_right` @ `s0167` / `heart_to_kidney` / `d030_r100`
  - **Trusted anchor (purple silhouette)**: `heart` · poset pair `heart` / `iliopsoas_right`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.55790 → 0.64332 (Δ +0.08543)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 4,452 total; TP (red) 0; FP (green) 4,452
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 4,436 removed; TP 0; FP 4,436

- **bad d≤0.5 (s0167)** — targeted structure `duodenum` @ `s0167` / `heart_to_kidney` / `d035_r075`
  - **Trusted anchor (purple silhouette)**: `gluteus_maximus_left` · poset pair `duodenum` / `gluteus_maximus_left`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.85789 → 0.00000 (Δ -0.85789)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 9,978 total; TP (red) 8,802; FP (green) 1,176
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 9,978 removed; TP 8,802; FP 1,176

- **bad d≤0.5 (s0167)** — targeted structure `gallbladder` @ `s0167` / `heart_to_kidney` / `d040_r050`
  - **Trusted anchor (purple silhouette)**: `gluteus_medius_right` · poset pair `gallbladder` / `gluteus_medius_right`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.84098 → 0.00000 (Δ -0.84098)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 1,166 total; TP (red) 1,026; FP (green) 140
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 1,166 removed; TP 1,026; FP 140

- **bad d≤0.5 (s0167)** — targeted structure `portal_vein_and_splenic_vein` @ `s0167` / `heart_to_kidney` / `d040_r100`
  - **Trusted anchor (purple silhouette)**: `gluteus_minimus_right` · poset pair `portal_vein_and_splenic_vein` / `gluteus_minimus_right`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.53763 → 0.00000 (Δ -0.53763)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 1,032 total; TP (red) 400; FP (green) 632
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 1,032 removed; TP 400; FP 632

