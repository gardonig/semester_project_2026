"""
Visualize MRI wrap-around artifacts.

For each subject, extracts a mid-sagittal slice (shows full head-foot extent)
and the first + last axial slices. Saves a grid image per page of subjects
so you can visually scan for wrap-around (anatomy appearing at FOV edges).

Usage:
    python scripts/data_prep/visualize_mri_wraparound.py \
        --mri_dir data/datasets/TotalsegmentatorMRI_dataset_v200 \
        --out_dir data/datasets/mri_wraparound_vis \
        --per_page 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes


def load_slices(mri_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns (sagittal_mid, axial_first, axial_last) as 2D arrays, or None on failure."""
    try:
        img = nib.load(str(mri_path))
        data = np.asarray(img.dataobj, dtype=np.float32)
    except Exception:
        return None

    codes = aff2axcodes(img.affine)
    # Find the superior-inferior axis (S or I)
    si_ax = next((i for i, c in enumerate(codes) if c in ('S', 'I')), 2)

    # Reorder so axis 2 = S/I (axial slices along last axis)
    axes = [i for i in range(3) if i != si_ax] + [si_ax]
    data = np.transpose(data, axes)
    # data shape: (x, y, axial)

    # Sagittal = mid slice along axis 0 (left-right), showing (y, axial)
    sag = data[data.shape[0] // 2, :, :]   # shape: (y, axial)
    sag = np.rot90(sag)                     # put superior at top

    axial_first = data[:, :, 0]
    axial_last  = data[:, :, -1]

    def normalise(arr: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(arr, [2, 98])
        arr = np.clip(arr, p2, p98)
        denom = p98 - p2 if p98 > p2 else 1.0
        return (arr - p2) / denom

    return normalise(sag), normalise(axial_first), normalise(axial_last)


def make_page(
    subjects: list[tuple[str, Path]],
    page_idx: int,
    out_dir: Path,
) -> None:
    n = len(subjects)
    fig, axes = plt.subplots(n, 3, figsize=(9, n * 1.8))
    if n == 1:
        axes = [axes]

    col_titles = ["Sagittal (full extent)", "Axial — first slice", "Axial — last slice"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=7, pad=3)

    for row, (subj_id, mri_path) in enumerate(subjects):
        slices = load_slices(mri_path)
        row_axes = axes[row]
        for ax in row_axes:
            ax.axis("off")
        if slices is None:
            row_axes[0].set_ylabel(subj_id, fontsize=6, rotation=0, labelpad=40, va='center')
            continue

        sag, ax_first, ax_last = slices
        row_axes[0].imshow(sag,      cmap="gray", aspect="auto", interpolation="nearest")
        row_axes[1].imshow(ax_first, cmap="gray", aspect="equal", interpolation="nearest")
        row_axes[2].imshow(ax_last,  cmap="gray", aspect="equal", interpolation="nearest")
        row_axes[0].set_ylabel(subj_id, fontsize=6, rotation=0, labelpad=45, va='center')

    plt.suptitle(f"MRI Wrap-Around Visual Check — Page {page_idx + 1}", fontsize=9, y=1.002)
    plt.tight_layout(h_pad=0.3, w_pad=0.3)

    out_path = out_dir / f"wraparound_page_{page_idx + 1:03d}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mri_dir",  required=True, type=Path)
    p.add_argument("--out_dir",  required=True, type=Path)
    p.add_argument("--per_page", default=40,    type=int,
                   help="Subjects per output image page")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    subjects = sorted(
        (d.name, d / "mri.nii.gz")
        for d in args.mri_dir.iterdir()
        if d.is_dir() and (d / "mri.nii.gz").exists()
    )
    print(f"Found {len(subjects)} subjects with mri.nii.gz")

    pages = [subjects[i:i + args.per_page] for i in range(0, len(subjects), args.per_page)]
    print(f"Generating {len(pages)} pages ({args.per_page} subjects/page)...")

    for page_idx, page_subjects in enumerate(pages):
        make_page(page_subjects, page_idx, args.out_dir)

    print(f"\nDone. Open {args.out_dir} to review.")


if __name__ == "__main__":
    main()
