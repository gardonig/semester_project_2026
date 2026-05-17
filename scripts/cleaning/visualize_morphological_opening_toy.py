"""
Toy 3-panel figure illustrating morphological opening on two separate blobs.

Two disconnected structures from the start:
  - a large blob (survives erosion)
  - a small blob (fully erased by erosion — too small for the structuring element)

Sequence:
  1. Original mask  — both blobs present
  2. After erosion  — small blob gone, large blob shrunk
  3. After dilation — large blob restored (LCC); small blob never comes back

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
# Structuring element — disk, radius 4
# ---------------------------------------------------------------------------
RADIUS = 4
se = np.zeros((2 * RADIUS + 1, 2 * RADIUS + 1), dtype=bool)
for dr in range(-RADIUS, RADIUS + 1):
    for dc in range(-RADIUS, RADIUS + 1):
        if dr * dr + dc * dc <= RADIUS * RADIUS:
            se[dr + RADIUS, dc + RADIUS] = True

# ---------------------------------------------------------------------------
# Build synthetic mask  (two separate blobs, no connection)
# ---------------------------------------------------------------------------
H, W = 34, 52
mask = np.zeros((H, W), dtype=bool)

# Large blob — wide enough to survive erosion by radius 4
mask[4:30, 2:30] = True
# round corners
for r, c in [(4,2),(4,3),(4,4),(5,2),(5,3),(6,2),
             (4,28),(4,27),(4,26),(5,28),(5,27),(6,28),
             (29,2),(29,3),(29,4),(28,2),(28,3),(27,2),
             (29,28),(29,27),(29,26),(28,28),(28,27),(27,28)]:
    mask[r, c] = False

# Small blob — 5×5, fully erased by a radius-4 erosion
mask[6:11, 36:41] = True

# ---------------------------------------------------------------------------
# Erosion and dilation
# ---------------------------------------------------------------------------
eroded = binary_erosion(mask, structure=se)
opened = binary_dilation(eroded, structure=se)

# ---------------------------------------------------------------------------
# Identify the two components
# ---------------------------------------------------------------------------
labeled_orig, _ = label(mask)
comp1 = labeled_orig == 1
comp2 = labeled_orig == 2
if comp1.sum() < comp2.sum():
    comp1, comp2 = comp2, comp1   # comp1 = large, comp2 = small

# ---------------------------------------------------------------------------
# Colours — single colour for the mask, grey for eroded-away pixels
# ---------------------------------------------------------------------------
BLUE      = np.array([0.20, 0.47, 0.87, 1.0])
GREY_LOST = np.array([0.78, 0.78, 0.78, 1.0])

BG = np.ones((H, W, 4))   # white background

def _make(fg: np.ndarray, faded: np.ndarray | None = None) -> np.ndarray:
    img = BG.copy()
    if faded is not None:
        img[faded] = GREY_LOST
    img[fg] = BLUE
    return img

# Panel 1: both blobs, same colour
img1 = _make(mask)

# Panel 2: after erosion — small gone, large shrunk; eroded boundary shown grey
img2 = _make(eroded, faded=(mask & ~eroded))

# Panel 3: after dilation (opening) — only LCC survives
img3 = _make(opened)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(5, 11))
fig.patch.set_facecolor("#f7f7f7")

panels = [
    (img1, "1 — Input mask",
     "Both structures present"),
    (img2, "2 — After erosion",
     "Small structure erased  |  large structure shrinks"),
    (img3, "3 — After dilation  →  LCC",
     "Large structure restored  |  small structure gone"),
]

for ax, (img, title, sub) in zip(axes, panels):
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel(sub, fontsize=8.5, color="#444")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

legend_handles = [
    mpatches.Patch(color=BLUE,      label="Segmentation mask"),
    mpatches.Patch(color=GREY_LOST, label="Pixels removed by erosion"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

fig.tight_layout(rect=[0, 0.1, 1, 1])
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {OUT_PATH}")
