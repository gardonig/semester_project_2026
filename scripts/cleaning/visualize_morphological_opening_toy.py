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
RADIUS = 2
se = np.zeros((2 * RADIUS + 1, 2 * RADIUS + 1), dtype=bool)
for dr in range(-RADIUS, RADIUS + 1):
    for dc in range(-RADIUS, RADIUS + 1):
        if dr * dr + dc * dc <= RADIUS * RADIUS:
            se[dr + RADIUS, dc + RADIUS] = True

# ---------------------------------------------------------------------------
# Build synthetic mask  (two separate blobs, no connection)
# ---------------------------------------------------------------------------
H, W = 72, 112
mask = np.zeros((H, W), dtype=bool)

# Heart blob — implicit equation (x²+y²−1)³ − x²y³ ≤ 0
# Scaled so the heart is ~40 px wide; thick enough to survive radius-4 erosion
cy, cx = 38, 40
s = 19.0
rows, cols = np.mgrid[0:H, 0:W]
x = (cols - cx) / s
y = -(rows - cy) / s   # flip so y-up gives standard upward-pointing heart
mask |= (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0

# Small circular blob — radius 1.9 px (9 pixels), fully erased by radius-2 erosion
# (SE of radius 2 cannot fit inside: nearest boundary is only ~2 px from center)
_br, _bc = 20, 82
for _dr in range(-2, 3):
    for _dc in range(-2, 3):
        if _dr * _dr + _dc * _dc <= 3.61:   # radius ≈ 1.9
            if 0 <= _br + _dr < H and 0 <= _bc + _dc < W:
                mask[_br + _dr, _bc + _dc] = True

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
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.patch.set_facecolor("#f7f7f7")
fig.suptitle("Morphological Opening", fontsize=18, fontweight="bold", y=1.02)

panels = [
    (img1, "Input"),
    (img2, "Eroded"),
    (img3, "Opened"),
]

for ax, (img, title) in zip(axes, panels):
    ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

legend_handles = [
    mpatches.Patch(color=BLUE,      label="Mask"),
    mpatches.Patch(color=GREY_LOST, label="Eroded away"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=13, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {OUT_PATH}")
