"""
Simulate MRI Wrap-Around Artifact (Section 2.1)
================================================

For each anatomical crop window (= scanner FOV), d slices just outside each
FOV boundary contribute the aliased ghost:

    extended window = N + 2d  slices   (d above + N FOV + d below)
    ┌──────────────────────────────────────────────────────────────┐
    │  d slices above FOV  →  wrap to INFERIOR end of FOV image   │
    │  ──────────────────────────────────────────────────────────  │
    │       N slices = FOV image  (what TotalSegmentator sees)     │
    │  ──────────────────────────────────────────────────────────  │
    │  d slices below FOV  →  wrap to SUPERIOR end of FOV image   │
    └──────────────────────────────────────────────────────────────┘

d is expressed as a fraction of N (the FOV height).  Different d values
simulate different amounts of FOV under-prescription.  r controls the ghost
intensity.  Brightness normalisation (Eqs. 3-4) preserves the contrast ratio
of the wrapped vs. unwrapped area.

Usage
-----
    # Single subject, visuals only
    python scripts/data_prep/simulate_wraparound_artifact.py \\
        --mri_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \\
        --subjects s0175 \\
        --shift_fracs 0.10 0.25 0.50 \\
        --intensities 0.5 1.0 \\
        --out_dir data/experiments/wraparound \\
        --visualize

    # Top-N subjects + TotalSegmentator
    python scripts/data_prep/simulate_wraparound_artifact.py \\
        --mri_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \\
        --top_n 10 \\
        --shift_fracs 0.10 0.25 0.50 \\
        --intensities 0.5 1.0 \\
        --out_dir data/experiments/wraparound \\
        --visualize \\
        --run_totalseg \\
        --totalseg .totalseg_venv/bin/TotalSegmentator
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes


# ---------------------------------------------------------------------------
# Crop definitions: (name, superior_anchor_structure, inferior_anchor_structure)
# Using structures available in TotalsegmentatorMRI_v200.
# ---------------------------------------------------------------------------
CROPS = [
    ("brain_to_heart",  "brain",       "heart"),
    ("heart_to_kidney", "heart",       "kidney_left"),
    ("kidney_to_hip",   "kidney_left", "hip_left"),
]

FALLBACKS = {
    "kidney_left": "kidney_right",
    "hip_left":    "hip_right",
}

CROP_MARGIN_VOXELS = 5


# ---------------------------------------------------------------------------
# Subject ranking
# ---------------------------------------------------------------------------

def _count_nonempty(subj_dir: Path) -> tuple[str, int]:
    seg_dir = subj_dir / "segmentations"
    if not seg_dir.exists():
        return subj_dir.name, 0
    count = sum(
        1 for f in seg_dir.glob("*.nii.gz")
        if np.any(nib.load(str(f)).dataobj)
    )
    return subj_dir.name, count


def get_best_subjects(mri_dir: Path, n: int = 1) -> list[str]:
    """Return the n subjects with the most non-empty segmentation masks."""
    subjects = [d for d in sorted(mri_dir.iterdir())
                if d.is_dir() and d.name != "__pycache__"]
    results: list[tuple[str, int]] = []
    print(f"Ranking {len(subjects)} subjects by non-empty mask count...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_count_nonempty, s): s for s in subjects}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(subjects)} done...")
    results.sort(key=lambda x: -x[1])
    top = [name for name, _ in results[:n]]
    for i, (name, count) in enumerate(results[:n], 1):
        print(f"  Rank {i}: {name}  ({count} non-empty structures)")
    return top


# ---------------------------------------------------------------------------
# Axis utilities
# ---------------------------------------------------------------------------

def load_mri_axes(
    mri_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    """
    Load MRI volume and determine anatomical axes.

    Returns
    -------
    data      : float32 array
    affine    : (4,4) affine matrix
    orig_dtype: original NIfTI data dtype (for saving back)
    si_ax     : voxel axis for S-I direction
    si_sign   : +1 if 'S' (index 0 = inferior, high = superior),
                -1 if 'I' (index 0 = superior, high = inferior)
    ap_ax     : voxel axis for A-P direction
    lr_ax     : voxel axis for L-R direction
    """
    img = nib.load(str(mri_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    orig_dtype = img.get_data_dtype()
    codes = aff2axcodes(affine)

    si_ax = ap_ax = lr_ax = None
    si_sign = 0
    for ax, code in enumerate(codes):
        if code == "S":
            si_ax, si_sign = ax, +1
        elif code == "I":
            si_ax, si_sign = ax, -1
        elif code in ("A", "P"):
            ap_ax = ax
        elif code in ("L", "R"):
            lr_ax = ax

    if si_ax is None:
        raise ValueError(f"Cannot determine S-I axis from affine codes {codes}")
    remaining = [i for i in range(3) if i != si_ax]
    if ap_ax is None:
        ap_ax = remaining[0]
    if lr_ax is None:
        lr_ax = remaining[1] if remaining[1] != ap_ax else remaining[0]

    return data, affine, orig_dtype, si_ax, si_sign, ap_ax, lr_ax


def _si_slice(ndim: int, si_ax: int, lo: int, hi: int) -> tuple:
    s = [slice(None)] * ndim
    s[si_ax] = slice(lo, hi)
    return tuple(s)


# ---------------------------------------------------------------------------
# Artifact simulation (Equations 1–4)
# ---------------------------------------------------------------------------

def simulate_wraparound_from_crop(
    full_volume: np.ndarray,
    si_ax: int,
    si_sign: int,
    lo: int,
    hi: int,
    d: int,
    r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate bidirectional S-I wrap-around for a crop window [lo, hi].

    The anatomy just outside the FOV boundaries contributes the ghost:
      - d slices above the FOV  →  wrapped to the INFERIOR end of the FOV image
      - d slices below the FOV  →  wrapped to the SUPERIOR end of the FOV image

    'Above' and 'below' are defined in anatomical (superior/inferior) terms;
    the mapping to array index direction depends on si_sign.

    Parameters
    ----------
    full_volume : full 3D float32 MRI array (used as context for the ghost)
    si_ax       : voxel axis for S-I
    si_sign     : +1 if 'S' (high index = superior), -1 if 'I' (low = superior)
    lo, hi      : crop slice indices along si_ax (hi exclusive); FOV = N = hi-lo
    d           : number of slices to wrap from each side (1 ≤ d ≤ N//2)
    r           : ghost intensity multiplier

    Returns
    -------
    I     : the FOV crop (N slices, float32)
    I_hat : artifact layer (same shape as I)
    I_s   : brightness-normalised artifacted image (same shape as I)
    """
    M_full = full_volume.shape[si_ax]
    N = hi - lo
    d = max(1, min(d, N // 2))
    ndim = full_volume.ndim

    I = full_volume[_si_slice(ndim, si_ax, lo, hi)].astype(np.float32)
    I_hat = np.zeros_like(I)

    def _copy_external(src_lo: int, src_hi: int, dst_lo: int, dst_hi: int) -> None:
        """Copy full_volume[src_lo:src_hi] * r into I_hat[dst_lo:dst_hi] along si_ax."""
        src_lo = max(src_lo, 0)
        src_hi = min(src_hi, M_full)
        avail = src_hi - src_lo
        if avail <= 0:
            return
        # Adjust destination if source is clipped
        actual_dst_lo = dst_lo + (d - avail) if src_lo == 0 else dst_lo
        actual_dst_hi = actual_dst_lo + avail
        if actual_dst_hi > N or actual_dst_lo < 0:
            return
        I_hat[_si_slice(ndim, si_ax, actual_dst_lo, actual_dst_hi)] = (
            full_volume[_si_slice(ndim, si_ax, src_lo, src_hi)] * r
        )

    if si_sign == +1:
        # 'S' axis: high index = superior
        # Above FOV = full[hi : hi+d]  →  inferior end of I = I[0:d]
        _copy_external(hi, hi + d,     dst_lo=0,   dst_hi=d)
        # Below FOV = full[lo-d : lo]  →  superior end of I = I[N-d:N]
        _copy_external(lo - d, lo,     dst_lo=N-d, dst_hi=N)
    else:
        # 'I' axis: low index = superior
        # Above FOV = full[lo-d : lo]  →  inferior end of I = I[N-d:N]
        _copy_external(lo - d, lo,     dst_lo=N-d, dst_hi=N)
        # Below FOV = full[hi : hi+d]  →  superior end of I = I[0:d]
        _copy_external(hi, hi + d,     dst_lo=0,   dst_hi=d)

    # Eq. 2: overlay masked to body foreground
    F     = (I > 0)
    F_hat = (I_hat > 0)
    I_r   = (I + I_hat) * F

    # Wrapped-area map: 0=background, 1=unwrapped body, 2=wrapped overlap
    V = (F.astype(np.int8) + F_hat.astype(np.int8)) * F.astype(np.int8)

    # Eqs. 3–4: brightness normalisation
    I1 = float(I[V == 1].sum())
    I2 = float(I[V == 2].sum())
    I_s = I_r.copy()
    if I1 > 0 and I2 > 0:
        I_s[V == 2] *= I2 / I1

    return I, I_hat, I_s


# ---------------------------------------------------------------------------
# Structure-based cropping
# ---------------------------------------------------------------------------

def _resolve_structure(seg_dir: Path, name: str) -> Optional[Path]:
    p = seg_dir / f"{name}.nii.gz"
    if p.exists():
        return p
    fallback = FALLBACKS.get(name)
    if fallback:
        p2 = seg_dir / f"{fallback}.nii.gz"
        if p2.exists():
            return p2
    return None


def get_structure_si_extent(
    seg_dir: Path, struct_name: str, si_ax: int
) -> Optional[tuple[int, int]]:
    """Return (lo_idx, hi_idx) non-zero extent of structure along si_ax, or None."""
    seg_path = _resolve_structure(seg_dir, struct_name)
    if seg_path is None:
        return None
    mask = np.asarray(nib.load(str(seg_path)).dataobj)
    other = tuple(ax for ax in range(mask.ndim) if ax != si_ax)
    proj = mask.any(axis=other)
    idx = np.where(proj)[0]
    if len(idx) == 0:
        return None
    return int(idx.min()), int(idx.max())


def structures_in_crop(
    seg_dir: Path, lo: int, hi: int, si_ax: int
) -> list[str]:
    """List structure names with any voxel within [lo, hi) along si_ax."""
    present = []
    for seg_path in sorted(seg_dir.glob("*.nii.gz")):
        mask = np.asarray(nib.load(str(seg_path)).dataobj)
        if mask[_si_slice(mask.ndim, si_ax, lo, hi)].any():
            present.append(seg_path.name.replace(".nii.gz", ""))
    return present


def crop_affine(affine: np.ndarray, si_ax: int, lo: int) -> np.ndarray:
    """Return updated affine after cropping lo slices from the start of si_ax."""
    new_affine = affine.copy()
    if lo > 0:
        new_affine[:3, 3] += affine[:3, si_ax] * lo
    return new_affine


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _normalise(arr: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(arr, [2, 98])
    arr = np.clip(arr, p2, p98)
    denom = p98 - p2 if p98 > p2 else 1.0
    return (arr - p2) / denom


def _mid_coronal(volume: np.ndarray, ap_ax: int, si_ax: int) -> np.ndarray:
    """Extract mid-A-P coronal slice with superior at the top of the image."""
    mid = volume.shape[ap_ax] // 2
    s = [slice(None)] * volume.ndim
    s[ap_ax] = mid
    coronal = volume[tuple(s)]                                 # 2D: (dim_a, dim_b)
    axes_remaining = [ax for ax in range(volume.ndim) if ax != ap_ax]
    si_pos = axes_remaining.index(si_ax)
    if si_pos != 0:
        coronal = coronal.T                                    # put SI as axis 0
    return np.flipud(coronal)                                  # row 0 = superior


def visualize_coronal_comparison(
    I: np.ndarray,
    I_hat: np.ndarray,
    I_s: np.ndarray,
    ap_ax: int,
    si_ax: int,
    title: str,
    out_path: Path,
) -> None:
    panels = [
        (_mid_coronal(I,     ap_ax, si_ax), "FOV image (clean)"),
        (_mid_coronal(I_hat, ap_ax, si_ax), "Artifact layer Î"),
        (_mid_coronal(I_s,   ap_ax, si_ax), "Artifacted I_s"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, (img, ttl) in zip(axes, panels):
        ax.imshow(_normalise(img), cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(ttl, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# TotalSegmentator runner (pattern from truncated_fov_experiment.py)
# ---------------------------------------------------------------------------

def run_totalseg(mri_path: Path, out_dir: Path, totalseg_bin: str) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [totalseg_bin, "-i", str(mri_path), "-o", str(out_dir), "--fast"]
    env = os.environ.copy()
    Path(env.get("TMPDIR", "/tmp")).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"    [error] TotalSegmentator failed:\n{result.stderr[-600:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# Save NIfTI helper
# ---------------------------------------------------------------------------

def save_nifti(data: np.ndarray, affine: np.ndarray, orig_dtype: np.dtype, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        arr = np.clip(data, info.min, info.max).astype(orig_dtype)
    else:
        arr = data.astype(orig_dtype)
    nib.save(nib.Nifti1Image(arr, affine), str(path))


# ---------------------------------------------------------------------------
# Per-subject experiment
# ---------------------------------------------------------------------------

def run_subject(
    subject: str,
    mri_dir: Path,
    out_dir: Path,
    shift_fracs: list[float],
    intensities: list[float],
    visualize: bool,
    totalseg_bin: Optional[str],
) -> None:
    mri_path = mri_dir / subject / "mri.nii.gz"
    seg_dir  = mri_dir / subject / "segmentations"
    if not mri_path.exists():
        print(f"  [skip] {subject}: mri.nii.gz not found")
        return

    print(f"\n{'='*60}\n  Subject: {subject}\n{'='*60}")
    data, affine, orig_dtype, si_ax, si_sign, ap_ax, lr_ax = load_mri_axes(mri_path)
    M = data.shape[si_ax]
    print(f"  Volume shape: {data.shape}  si_ax={si_ax}  si_sign={si_sign:+d}  M={M}")

    subj_out = out_dir / subject

    # Resolve crop windows once per subject
    crops_resolved: list[tuple[str, int, int]] = []
    for crop_name, struct_sup, struct_inf in CROPS:
        ext_sup = get_structure_si_extent(seg_dir, struct_sup, si_ax)
        ext_inf = get_structure_si_extent(seg_dir, struct_inf, si_ax)
        if ext_sup is None:
            print(f"  [skip crop {crop_name}] '{struct_sup}' absent or empty")
            continue
        if ext_inf is None:
            print(f"  [skip crop {crop_name}] '{struct_inf}' absent or empty")
            continue

        # For 'S' (high = superior): superior structure has high idx, inferior has low idx
        # For 'I' (low = superior): superior structure has low idx, inferior has high idx
        if si_sign == +1:
            lo = max(ext_inf[0] - CROP_MARGIN_VOXELS, 0)
            hi = min(ext_sup[1] + CROP_MARGIN_VOXELS, M)
        else:
            lo = max(ext_sup[0] - CROP_MARGIN_VOXELS, 0)
            hi = min(ext_inf[1] + CROP_MARGIN_VOXELS, M)

        if lo >= hi:
            print(f"  [skip crop {crop_name}] degenerate range [{lo}, {hi}]")
            continue

        N = hi - lo
        print(f"  Crop '{crop_name}': [{lo}, {hi})  N={N} slices")

        # Write structures_present.txt once (independent of d/r)
        crop_base = subj_out / crop_name
        if seg_dir.exists():
            crop_base.mkdir(parents=True, exist_ok=True)
            present = structures_in_crop(seg_dir, lo, hi, si_ax)
            (crop_base / "structures_present.txt").write_text("\n".join(present) + "\n")
            print(f"    {len(present)} structures in FOV")

        crops_resolved.append((crop_name, lo, hi))

    if not crops_resolved:
        print("  No valid crop windows — skipping subject")
        return

    # Sweep conditions
    for crop_name, lo, hi in crops_resolved:
        N = hi - lo
        crop_affine_mat = crop_affine(affine, si_ax, lo)

        for d_frac in shift_fracs:
            d_vox = max(1, int(round(d_frac * N)))
            for r in intensities:
                tag = f"d{int(d_frac * 100):03d}_r{int(r * 100):03d}"
                cond_dir = subj_out / crop_name / tag
                cond_dir.mkdir(parents=True, exist_ok=True)

                print(f"\n  [{crop_name}] {tag}: d={d_vox}/{N} slices ({d_frac:.0%}), r={r}")

                I, I_hat, I_s = simulate_wraparound_from_crop(
                    data, si_ax, si_sign, lo, hi, d_vox, r
                )

                # Save artifacted crop volume
                art_path = cond_dir / "mri_artifact.nii.gz"
                save_nifti(I_s, crop_affine_mat, orig_dtype, art_path)

                # Visualise
                if visualize:
                    visualize_coronal_comparison(
                        I, I_hat, I_s, ap_ax, si_ax,
                        title=f"{subject}  {crop_name}  {tag}  [{lo}:{hi}]  d={d_vox}",
                        out_path=cond_dir / "coronal_comparison.png",
                    )

                # Run TotalSegmentator
                if totalseg_bin:
                    seg_out = cond_dir / "segmentations"
                    print(f"    Running TotalSegmentator...")
                    ok = run_totalseg(art_path, seg_out, totalseg_bin)
                    print(f"    TotalSegmentator {'OK' if ok else 'FAILED'}")

    print(f"\n  Done: {subj_out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mri_dir",      required=True, type=Path,
                   help="TotalsegmentatorMRI_dataset_v200 root")
    p.add_argument("--subjects",     nargs="+", default=None,
                   help="Explicit subject IDs (overrides --top_n)")
    p.add_argument("--top_n",        type=int, default=1,
                   help="Process the N best-covered subjects (default: 1)")
    p.add_argument("--shift_fracs",  nargs="+", type=float, default=[0.10, 0.25, 0.50],
                   help="d as fraction of crop-window height N (default: 0.10 0.25 0.50)")
    p.add_argument("--intensities",  nargs="+", type=float, default=[0.5, 1.0],
                   help="Ghost intensity r values (default: 0.5 1.0)")
    p.add_argument("--out_dir",      required=True, type=Path)
    p.add_argument("--visualize",    action="store_true",
                   help="Save 3-panel coronal comparison PNGs")
    p.add_argument("--run_totalseg", action="store_true",
                   help="Run TotalSegmentator on each artifacted volume")
    p.add_argument("--totalseg",     default="TotalSegmentator",
                   help="Path to TotalSegmentator binary")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    totalseg_bin = args.totalseg if args.run_totalseg else None

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = get_best_subjects(args.mri_dir, n=args.top_n)

    print(f"Subjects:    {subjects}")
    print(f"d fractions: {args.shift_fracs}")
    print(f"Intensities: {args.intensities}")
    print(f"Output:      {args.out_dir}")

    for subject in subjects:
        run_subject(
            subject, args.mri_dir, args.out_dir,
            args.shift_fracs, args.intensities,
            args.visualize, totalseg_bin,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
