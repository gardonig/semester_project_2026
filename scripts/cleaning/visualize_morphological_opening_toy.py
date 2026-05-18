"""3-panel toy figure: morphological opening removes the small blob, preserves the large one."""

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

RADIUS = 8
se = np.zeros((2 * RADIUS + 1, 2 * RADIUS + 1), dtype=bool)
for dr in range(-RADIUS, RADIUS + 1):
    for dc in range(-RADIUS, RADIUS + 1):
        if dr * dr + dc * dc <= RADIUS * RADIUS:
            se[dr + RADIUS, dc + RADIUS] = True

H, W = 120, 190
mask = np.zeros((H, W), dtype=bool)

cy, cx = 62, 62
s = 28.0
rows, cols = np.mgrid[0:H, 0:W]
x = (cols - cx) / s
y = -(rows - cy) / s
mask |= (x**2 + y**2 - 1)**3 - x**2 * y**3 <= 0

_br, _bc = 50, 145
for _dr in range(-7, 8):
    for _dc in range(-7, 8):
        if _dr * _dr + _dc * _dc <= 49:   # radius = 7
            if 0 <= _br + _dr < H and 0 <= _bc + _dc < W:
                mask[_br + _dr, _bc + _dc] = True

eroded = binary_erosion(mask, structure=se)
opened = binary_dilation(eroded, structure=se)

labeled_orig, _ = label(mask)
comp1 = labeled_orig == 1
comp2 = labeled_orig == 2
if comp1.sum() < comp2.sum():
    comp1, comp2 = comp2, comp1

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

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.patch.set_facecolor("#f7f7f7")
fig.suptitle("Morphological Opening", fontsize=18, fontweight="bold", y=1.02)

panels = [
    (img1, "Input"),
    (img2, "Eroded"),
    (img3, "Dilated (Opened)"),
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
