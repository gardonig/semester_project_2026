# Real cleaning cases — subject s0186 (coronal; SI axis 2× screen stretch)

Good rows: d≤0.35, ΔDice>0, ranked by largest `vox_removed_pc` (up to 6; editorial rank skip when enough candidates). Bad rows: d≤0.5, ΔDice<0, same rule (up to 6). Good/bad pools use ΔDice (CSV has no delta_f1 — re-run evaluate_cleaning_methods).

Figure: one coronal MRI slice; mask overlays are **A/P silhouettes** on LR×SI. For this subject (`s0186`), LR and SI use **equal** on-screen pixel scale. **Trusted anchor (purple)** = constraint partner; **red** = removed TP, **green** = removed FP.

- **good d≤0.35 (s0186)** — targeted structure `gluteus_medius_right` @ `s0186` / `heart_to_kidney` / `d035_r100`
  - **Trusted anchor (purple silhouette)**: `pancreas` · poset pair `pancreas` / `gluteus_medius_right`
  - **Step (caption)**: `conflict_prefer_i_remove_j`
  - **F1** (full 3D mask vs GT): 0.00368 → 0.36782 (Δ +0.36413)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 60,214 total; TP (red) 0; FP (green) 60,214
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 60,214 removed; TP 0; FP 60,214

- **good d≤0.35 (s0186)** — targeted structure `gluteus_medius_left` @ `s0186` / `heart_to_kidney` / `d030_r100`
  - **Trusted anchor (purple silhouette)**: `pancreas` · poset pair `pancreas` / `gluteus_medius_left`
  - **Step (caption)**: `conflict_prefer_i_remove_j`
  - **F1** (full 3D mask vs GT): 0.04491 → 0.85320 (Δ +0.80830)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 44,386 total; TP (red) 0; FP (green) 44,386
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 44,386 removed; TP 0; FP 44,386

- **good d≤0.35 (s0186)** — targeted structure `iliopsoas_right` @ `s0186` / `heart_to_kidney` / `d025_r100`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `iliopsoas_right`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.73409 → 0.91068 (Δ +0.17659)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 12,496 total; TP (red) 0; FP (green) 12,496
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 12,496 removed; TP 0; FP 12,496

- **good d≤0.35 (s0186)** — targeted structure `stomach` @ `s0186` / `kidney_to_hip` / `d035_r100`
  - **Trusted anchor (purple silhouette)**: `iliac_artery_left` · poset pair `stomach` / `iliac_artery_left`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.17416 → 0.39641 (Δ +0.22225)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 10,514 total; TP (red) 0; FP (green) 10,514
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 10,514 removed; TP 0; FP 10,514

- **good d≤0.35 (s0186)** — targeted structure `small_bowel` @ `s0186` / `heart_to_kidney` / `d035_r075`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `small_bowel`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.90590 → 0.92552 (Δ +0.01962)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 9,318 total; TP (red) 0; FP (green) 9,318
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 9,318 removed; TP 0; FP 9,318

- **good d≤0.35 (s0186)** — targeted structure `iliopsoas_left` @ `s0186` / `heart_to_kidney` / `d030_r075`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `iliopsoas_left`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.82117 → 0.88884 (Δ +0.06767)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 4,794 total; TP (red) 0; FP (green) 4,794
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 4,794 removed; TP 0; FP 4,794

