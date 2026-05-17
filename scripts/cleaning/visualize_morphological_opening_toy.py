"""
Toy figure illustrating morphological opening on a 2-D binary mask.

The input has two connected blobs joined by a thin neck:
  - a large main body (the true structure)
  - a small satellite blob connected by a neck thinner than the structuring element

After opening (erosion then dilation):
  - the thin neck is severed → two disconnected components
  - LCC selection keeps the large body and discards the satellite
  - the LCC itself is slightly smaller (opening rounds its boundary)

Output:
  results/cm4_visuals/morphological_opening_toy.png

Usage:
  python scripts/cleaning/visualize_morphological_opening_toy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_PATH = PROJECT_ROOT / "results" / "cm4_visuals" / "morphological_opening_toy.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build synthetic mask
# ---------------------------------------------------------------------------
H, W = 28, 44
mask = np.zeros((H, W), dtype=bool)

# Large main body  (rows 4-23, cols 2-28)
mask[4:24, 2:29] = True

# Thin neck connecting main body to satellite (1 px wide — narrower than SE)
mask[13:14, 29:35] = True

# Small satellite blob (rows 10-17, cols 35-42)
mask[10:18, 35:43] = True

# Round off corners a bit for a more natural look
for r, c in [(4, 2), (4, 3), (5, 2), (23, 2), (23, 3), (22, 2),
             (4, 27), (4, 28), (5, 28), (23, 27), (23, 28), (22, 28)]:
    mask[r, c] = False

# ---------------------------------------------------------------------------
# Morphological opening  (disk-like SE, radius 3)
# ---------------------------------------------------------------------------
radius = 3
se = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=bool)
for dr in range(-radius, radius + 1):
    for dc in range(-radius, radius + 1):
        if dr * dr + dc * dc <= radius * radius:
            se[dr + radius, dc + radius] = True

eroded  = binary_erosion(mask, structure=se)
opened  = binary_dilation(eroded, structure=se)

# ---------------------------------------------------------------------------
# LCC before and after
# ---------------------------------------------------------------------------
def lcc(m: np.ndarray) -> np.ndarray:
    labeled, n = label(m)
    if n == 0:
        return np.zeros_like(m)
    sizes = [(labeled == k).sum() for k in range(1, n + 1)]
    best = int(np.argmax(sizes)) + 1
    return labeled == best

lcc_before = lcc(mask)
lcc_after  = lcc(opened)

# ---------------------------------------------------------------------------
# Colour maps  (RGBA, float 0-1)
# ---------------------------------------------------------------------------
BLUE_FULL   = np.array([0.33, 0.60, 0.93, 1.0])   # main mask fill
BLUE_LCC    = np.array([0.10, 0.36, 0.78, 1.0])   # LCC highlight
RED_SAT     = np.array([0.88, 0.28, 0.25, 1.0])   # satellite / lost part
ORANGE_LOST = np.array([0.95, 0.60, 0.15, 1.0])   # eroded-away boundary

def _rgba_image(fg: np.ndarray, lcc_mask: np.ndarray,
                non_lcc: np.ndarray | None = None,
                lost: np.ndarray | None = None) -> np.ndarray:
    """Build an RGBA image from boolean layers."""
    img = np.ones((H, W, 4))   # white background
    # foreground fill
    img[fg] = BLUE_FULL
    # LCC highlight on top
    img[lcc_mask] = BLUE_LCC
    # non-LCC components
    if non_lcc is not None:
        img[non_lcc] = RED_SAT
    # pixels lost by opening
    if lost is not None:
        img[lost] = ORANGE_LOST
    return img

# Panel 1: raw mask, LCC highlighted, satellite red
non_lcc_before = mask & ~lcc_before
img_before = _rgba_image(mask, lcc_before, non_lcc=non_lcc_before)

# Panel 2: opened mask, LCC highlighted; satellite remnant red; lost boundary orange
lost_pixels    = mask & ~opened
non_lcc_after  = opened & ~lcc_after
img_after      = _rgba_image(opened, lcc_after, non_lcc=non_lcc_after, lost=lost_pixels)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
fig.patch.set_facecolor("#f8f8f8")

for ax, img, title, subtitle in [
    (axes[0], img_before,
     "Input mask",
     "LCC = large body  |  satellite connected via thin neck"),
    (axes[1], img_after,
     "After morphological opening  +  LCC selection",
     "Neck severed → satellite discarded  |  LCC boundary shrinks"),
]:
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(subtitle, fontsize=8.5, color="#444444")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

# Shared legend
legend_handles = [
    mpatches.Patch(color=BLUE_LCC,    label="LCC (kept)"),
    mpatches.Patch(color=RED_SAT,     label="Non-LCC component (discarded)"),
    mpatches.Patch(color=ORANGE_LOST, label="Pixels removed by opening"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {OUT_PATH}")
