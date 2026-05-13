"""
Create schematic CM3 (method3_middle_out_prior) visuals from report discussion.

Outputs:
  - cm3_algorithm_walkthrough.png
  - cm3_good_bad_cases.png

These are synthetic 2D diagrams (not subject-specific NIfTI slices), designed to
visually explain:
  1) how CM3 removes ghost components in favorable geometry
  2) where CM3 can fail (anchor inversion, merged components, variable extent)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "cm3_visuals"


def blank(shape=(160, 120)) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def rect(mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    mask[r0:r1, c0:c1] = True


def disk(mask: np.ndarray, cr: int, cc: int, rad: int) -> None:
    rr, cc_grid = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(rr - cr) ** 2 + (cc_grid - cc) ** 2 <= rad**2] = True


def overlay_rgb(gt: np.ndarray, pred_before: np.ndarray, pred_after: np.ndarray) -> np.ndarray:
    """
    RGB overlay:
      blue  : GT
      green : kept prediction after CM3
      red   : removed voxels (before - after)
      yellow: false-negative region (GT not in after)
    """
    removed = pred_before & ~pred_after
    kept = pred_after
    fn = gt & ~pred_after

    rgb = np.zeros((*gt.shape, 3), dtype=float)
    rgb[gt, 2] = 0.55
    rgb[kept, 1] = 0.90
    rgb[removed, 0] = 0.95
    rgb[fn, 0] = 0.90
    rgb[fn, 1] = 0.75
    return rgb


def make_walkthrough_figure(out_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    titles = [
        "Step 1: Pair + Anchors",
        "Step 2: Constraint Check",
        "Step 3: Remove Violating CCs",
        "Step 4: Recompute Anchor",
    ]

    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

    # Step 1
    ax = axes[0]
    ax.text(3, 92, "Pair: i above j", fontsize=10, weight="bold")
    ax.add_patch(plt.Rectangle((25, 68), 50, 16, color="#7fb3ff", alpha=0.9))
    ax.text(28, 75, "j anchor (reference)", fontsize=9)
    ax.add_patch(plt.Rectangle((35, 18), 30, 12, color="#66cc66", alpha=0.9))
    ax.text(28, 32, "i component A (real)", fontsize=8)
    ax.add_patch(plt.Rectangle((10, 6), 20, 8, color="#ff9966", alpha=0.9))
    ax.text(3, 16, "i component B (ghost)", fontsize=8)

    # Step 2
    ax = axes[1]
    ax.text(3, 92, "_is_entirely_below(ext_i, ext_j)?", fontsize=9, weight="bold")
    ax.add_patch(plt.Rectangle((25, 68), 50, 16, color="#7fb3ff", alpha=0.9))
    ax.add_patch(plt.Rectangle((35, 18), 30, 12, color="#66cc66", alpha=0.9))
    ax.annotate("", xy=(50, 68), xytext=(50, 30), arrowprops=dict(arrowstyle="<->", lw=1.5))
    ax.text(54, 48, "ordering gap", fontsize=8)
    ax.text(5, 7, "If violated: trigger guard / normal cleanup", fontsize=8)

    # Step 3
    ax = axes[2]
    ax.text(3, 92, "Remove non-anchor components", fontsize=9, weight="bold")
    ax.add_patch(plt.Rectangle((25, 68), 50, 16, color="#7fb3ff", alpha=0.9))
    ax.add_patch(plt.Rectangle((35, 18), 30, 12, color="#66cc66", alpha=0.9))
    ax.add_patch(plt.Rectangle((10, 6), 20, 8, color="#ff3333", alpha=0.9))
    ax.text(8, 1, "red = removed violating component", fontsize=8, color="#aa2222")

    # Step 4
    ax = axes[3]
    ax.text(3, 92, "Invalidate cache, relabel CCs", fontsize=9, weight="bold")
    ax.add_patch(plt.Rectangle((25, 68), 50, 16, color="#7fb3ff", alpha=0.9))
    ax.add_patch(plt.Rectangle((35, 18), 30, 12, color="#66cc66", alpha=0.9))
    ax.text(5, 44, "Updated masks feed next pair", fontsize=9)
    ax.text(5, 32, "(middle-out ordering reduces cascades)", fontsize=8, color="0.35")

    fig.suptitle("CM3 walkthrough (schematic)", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def case_good_kidney_to_hip() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = blank()
    before = blank()
    after = blank()

    # True anatomy (pelvic structure) and a separated inferior ghost.
    rect(gt, 78, 112, 45, 78)
    rect(before, 78, 112, 45, 78)   # real component
    rect(before, 18, 34, 18, 36)    # ghost component
    rect(after, 78, 112, 45, 78)    # ghost removed
    return gt, before, after


def case_bad_anchor_inversion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = blank()
    before = blank()
    after = blank()

    # Real humerus-like component (smaller) + ghost blob (larger) -> wrong anchor.
    disk(gt, 92, 76, 14)
    disk(before, 92, 76, 14)   # real
    disk(before, 40, 48, 18)   # ghost wins as anchor
    disk(after, 40, 48, 18)    # CM3 keeps ghost, removes real (failure)
    return gt, before, after


def case_bad_merged_component() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = blank()
    before = blank()
    after = blank()

    # Real + ghost connected by a bridge -> one CC, CM3 cannot split.
    rect(gt, 82, 108, 58, 84)
    rect(before, 82, 108, 58, 84)
    rect(before, 52, 70, 34, 52)    # ghost
    rect(before, 68, 84, 50, 62)    # connecting bridge
    after[:] = before               # unchanged failure mode
    return gt, before, after


def case_bad_variable_extent() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = blank()
    before = blank()
    after = blank()

    # Small-bowel-like multi-loop anatomy; superior loop is real but removed.
    rect(gt, 66, 82, 40, 62)
    rect(gt, 84, 100, 55, 78)
    rect(gt, 100, 116, 36, 56)

    before[:] = gt
    after[:] = before
    after[66:82, 40:62] = False     # falsely removed superior loop
    return gt, before, after


def draw_case(ax_before, ax_after, gt, before, after, title: str, subtitle: str) -> None:
    ax_before.imshow(overlay_rgb(gt, before, before), origin="lower", interpolation="nearest")
    ax_after.imshow(overlay_rgb(gt, before, after), origin="lower", interpolation="nearest")
    ax_before.set_title(f"{title}\nBefore", fontsize=9)
    ax_after.set_title("After CM3", fontsize=9)
    ax_before.text(2, 2, subtitle, fontsize=7, color="w", va="bottom", ha="left")
    ax_after.text(2, 2, subtitle, fontsize=7, color="w", va="bottom", ha="left")
    ax_before.axis("off")
    ax_after.axis("off")


def make_cases_figure(out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(8.5, 14))

    cases = [
        (
            "Good: kidney_to_hip",
            "FP ghost is spatially separated; CM3 removes it.",
            case_good_kidney_to_hip(),
        ),
        (
            "Bad: anchor inversion (humerus_right-like)",
            "Ghost chosen as anchor; real anatomy removed.",
            case_bad_anchor_inversion(),
        ),
        (
            "Bad: merged ghost+real CC",
            "Single connected blob, CM3 cannot split.",
            case_bad_merged_component(),
        ),
        (
            "Bad: variable S-I extent (small_bowel-like)",
            "Legitimate superior loop removed as 'violating'.",
            case_bad_variable_extent(),
        ),
    ]

    for row, (title, subtitle, (gt, before, after)) in enumerate(cases):
        draw_case(axes[row, 0], axes[row, 1], gt, before, after, title, subtitle)

    handles = [
        plt.Line2D([0], [0], color="#0000aa", lw=6, label="GT (blue)"),
        plt.Line2D([0], [0], color="#00cc00", lw=6, label="Kept prediction (green)"),
        plt.Line2D([0], [0], color="#ff3333", lw=6, label="Removed by CM3 (red)"),
        plt.Line2D([0], [0], color="#ffcc00", lw=6, label="False negative vs GT (yellow)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("CM3 good and failure cases (2D schematic segmentations)", fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_walk = OUT_DIR / "cm3_algorithm_walkthrough.png"
    out_cases = OUT_DIR / "cm3_good_bad_cases.png"
    make_walkthrough_figure(out_walk)
    make_cases_figure(out_cases)
    print(f"Saved: {out_walk}")
    print(f"Saved: {out_cases}")


if __name__ == "__main__":
    main()
