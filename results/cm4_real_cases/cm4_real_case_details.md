# Real cleaning cases — subject s0175 (coronal; SI axis 2× screen stretch)

Good rows: d≤0.35, ΔDice>0, ranked by largest `vox_removed_pc` (up to 6; editorial rank skip when enough candidates). Bad rows: d≤0.5, ΔDice<0, same rule (up to 6). Good/bad pools use ΔDice (CSV has no delta_f1 — re-run evaluate_cleaning_methods).

Figure: one coronal MRI slice; mask overlays are **A/P silhouettes** on LR×SI. For this subject (`s0175`), the SI axis is drawn **2× taller** on screen than LR. **Trusted anchor (purple)** = constraint partner; **red** = removed TP, **green** = removed FP.

- **good d≤0.35 (s0175)** — targeted structure `small_bowel` @ `s0175` / `heart_to_kidney` / `d035_r100`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `small_bowel`
  - **Step (caption)**: `conflict_prefer_i_remove_j`
  - **F1** (full 3D mask vs GT): 0.06560 → 0.16523 (Δ +0.09963)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 26,693 total; TP (red) 0; FP (green) 26,693
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 26,693 removed; TP 0; FP 26,693

- **good d≤0.35 (s0175)** — targeted structure `gluteus_medius_left` @ `s0175` / `heart_to_kidney` / `d025_r100`
  - **Trusted anchor (purple silhouette)**: `stomach` · poset pair `stomach` / `gluteus_medius_left`
  - **Step (caption)**: `conflict_prefer_i_remove_j`
  - **F1** (full 3D mask vs GT): 0.00609 → 0.13879 (Δ +0.13270)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 12,251 total; TP (red) 0; FP (green) 12,251
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 12,251 removed; TP 0; FP 12,251

- **good d≤0.35 (s0175)** — targeted structure `humerus_left` @ `s0175` / `heart_to_kidney` / `d020_r100`
  - **Trusted anchor (purple silhouette)**: `hip_left` · poset pair `humerus_left` / `hip_left`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.30842 → 0.71359 (Δ +0.40517)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 6,458 total; TP (red) 0; FP (green) 6,458
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 6,458 removed; TP 0; FP 6,458

- **good d≤0.35 (s0175)** — targeted structure `iliopsoas_right` @ `s0175` / `heart_to_kidney` / `d035_r075`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `iliopsoas_right`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.55784 → 0.70910 (Δ +0.15126)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 5,823 total; TP (red) 0; FP (green) 5,823
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 5,823 removed; TP 0; FP 5,823

- **good d≤0.35 (s0175)** — targeted structure `femur_right` @ `s0175` / `kidney_to_hip` / `d035_r100`
  - **Trusted anchor (purple silhouette)**: `aorta` · poset pair `aorta` / `femur_right`
  - **Step (caption)**: `normal_remove_j_above_i`
  - **F1** (full 3D mask vs GT): 0.66883 → 0.71715 (Δ +0.04832)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 2,724 total; TP (red) 0; FP (green) 2,724
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 2,724 removed; TP 0; FP 2,724

- **good d≤0.35 (s0175)** — targeted structure `iliopsoas_left` @ `s0175` / `heart_to_kidney` / `d030_r075`
  - **Trusted anchor (purple silhouette)**: `esophagus` · poset pair `esophagus` / `iliopsoas_left`
  - **Step (caption)**: `conflict_prefer_i_remove_j`
  - **F1** (full 3D mask vs GT): 0.13896 → 0.16353 (Δ +0.02456)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 2,298 total; TP (red) 0; FP (green) 2,298
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 2,298 removed; TP 0; FP 2,298

- **bad d≤0.5 (s0175)** — targeted structure `lung_left` @ `s0175` / `brain_to_heart` / `d050_r100`
  - **Trusted anchor (purple silhouette)**: `gluteus_medius_left` · poset pair `lung_left` / `gluteus_medius_left`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.96204 → 0.00000 (Δ -0.96204)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 211,365 total; TP (red) 202,450; FP (green) 8,915
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 211,365 removed; TP 202,450; FP 8,915

- **bad d≤0.5 (s0175)** — targeted structure `liver` @ `s0175` / `heart_to_kidney` / `d050_r100`
  - **Trusted anchor (purple silhouette)**: `femur_left` · poset pair `liver` / `femur_left`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.30598 → 0.00000 (Δ -0.30598)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 35,058 total; TP (red) 33,358; FP (green) 1,700
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 35,058 removed; TP 33,358; FP 1,700

- **bad d≤0.5 (s0175)** — targeted structure `kidney_right` @ `s0175` / `heart_to_kidney` / `d045_r075`
  - **Trusted anchor (purple silhouette)**: `femur_right` · poset pair `kidney_right` / `femur_right`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.19109 → 0.00000 (Δ -0.19109)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 2,788 total; TP (red) 2,421; FP (green) 367
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 2,788 removed; TP 2,421; FP 367

- **bad d≤0.5 (s0175)** — targeted structure `scapula_right` @ `s0175` / `brain_to_heart` / `d045_r075`
  - **Trusted anchor (purple silhouette)**: `gluteus_medius_left` · poset pair `scapula_right` / `gluteus_medius_left`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.20041 → 0.00000 (Δ -0.20041)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 1,164 total; TP (red) 920; FP (green) 244
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 1,164 removed; TP 920; FP 244

- **bad d≤0.5 (s0175)** — targeted structure `inferior_vena_cava` @ `s0175` / `heart_to_kidney` / `d040_r075`
  - **Trusted anchor (purple silhouette)**: `gluteus_minimus_right` · poset pair `inferior_vena_cava` / `gluteus_minimus_right`
  - **Step (caption)**: `conflict_prefer_j_remove_i`
  - **F1** (full 3D mask vs GT): 0.20389 → 0.00000 (Δ -0.20389)
  - **Removed 3D voxels** (matches figure: all `pred & ~cleaned` on target — blue / red / green): 917 total; TP (red) 796; FP (green) 121
  - **Caption event only** (one logged step; subset if multiple steps touched the organ): 917 removed; TP 796; FP 121

