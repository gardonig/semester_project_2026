"""
Toy 2-D figure illustrating the poset-based cleaning method.

Two scenarios on a stylised body cross-section:
  1. Normal firing  — small lung fragments above the brain are removed
                      (constraint: Brain above Lungs)
  2. Hard violation — the brain segmentation is displaced entirely below
                      the lungs and is removed outright

Output: results/cm4_visuals/poset_cleaning_toy.png
Usage:  python scripts/cleaning/visualize_poset_cleaning_toy.py
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = PROJECT_ROOT / "results" / "cm4_visuals" / "poset_cleaning_toy.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
C_BRAIN  = "#D4A0A0"
C_LUNG   = "#82AACC"
C_GHOST  = "#C4C4C4"
C_EDGE   = "#445566"
C_VIOL   = "#C83030"
C_OK     = "#2A9A5A"
C_BOUND  = "#D07020"
C_BG     = "#F8F7F4"
C_FIG    = "#ECEAE4"

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _ellipse(ax, cx, cy, w, h, fc, ec=C_EDGE, lw=1.8, ls="-",
             alpha=1.0, angle=0, z=4):
    ax.add_patch(Ellipse((cx, cy), w, h, angle=angle,
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         linestyle=ls, alpha=alpha, zorder=z))


def draw_brain(ax, cx, cy, fc=C_BRAIN, ec=C_EDGE, lw=1.8, ls="-", alpha=1.0):
    _ellipse(ax, cx, cy, 0.28, 0.18, fc, ec=ec, lw=lw, ls=ls, alpha=alpha)
    tc = "#443333" if alpha > 0.6 else "#AAAAAA"
    ax.text(cx, cy, "Brain", ha="center", va="center", fontsize=10,
            fontweight="bold", color=tc, alpha=alpha, zorder=6)


def draw_lungs(ax, cx, cy, fc=C_LUNG, ec=C_EDGE, lw=1.8, ls="-", alpha=1.0):
    for sign, side in [(-1, "L"), (1, "R")]:
        _ellipse(ax, cx + sign*0.185, cy, 0.21, 0.33, fc,
                 ec=ec, lw=lw, ls=ls, alpha=alpha, angle=-sign*8)
        ax.text(cx + sign*0.185, cy, f"{side}. Lung",
                ha="center", va="center", fontsize=9,
                color="#334455", alpha=alpha, zorder=6)


def draw_artifact_blob(ax, cx, cy, w=0.10, h=0.08):
    _ellipse(ax, cx, cy, w, h, C_LUNG, ec=C_VIOL, lw=2.2, ls="--")


def constraint_line(ax, y, label=""):
    ax.axhline(y, color=C_BOUND, linewidth=1.8, linestyle="--", alpha=0.85, zorder=3)
    if label:
        ax.text(0.97, y + 0.016, label, ha="right", va="bottom",
                fontsize=8, color=C_BOUND, style="italic", zorder=7)


def forbidden_shade(ax, y_lo, y_hi, color=C_BOUND):
    ax.fill_between([0, 1], y_lo, y_hi, color=color, alpha=0.07, zorder=1)


def rule_box(ax, cx, cy, text):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5,
            color=C_BOUND, style="italic", zorder=7,
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFF8EC",
                      ec=C_BOUND, linewidth=1.2, alpha=0.92))


def bottom_note(ax, text, color):
    ax.text(0.5, 0.025, text, ha="center", va="bottom", fontsize=9,
            color=color, style="italic", zorder=7)


def setup(ax, title, title_color="#333333"):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(C_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#C0C0C0")
        sp.set_linewidth(1.2)
    ax.set_title(title, fontsize=12, fontweight="bold",
                 color=title_color, pad=8)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(13, 9))
fig.patch.set_facecolor(C_FIG)
gs = GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.09,
              left=0.04, right=0.96, top=0.88, bottom=0.09)
ax = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]
for row in ax:
    for a in row:
        setup(a, "")

fig.suptitle("Poset-Based Cleaning", fontsize=20, fontweight="bold",
             y=0.95, color="#222222")

# ---------------------------------------------------------------------------
# Row 0 — Normal Firing
# ---------------------------------------------------------------------------
BY   = 0.79          # brain centre y
LY   = 0.46          # lung centre y
BFLOOR = BY - 0.18/2 # brain inferior boundary ≈ 0.70

# Panel (0,0) — Input
a = ax[0][0]
setup(a, "Normal Firing — Input", title_color="#2A5070")
draw_brain(a, 0.50, BY)
draw_lungs(a, 0.50, LY)
draw_artifact_blob(a, 0.22, 0.92, w=0.11, h=0.08)
draw_artifact_blob(a, 0.76, 0.90, w=0.09, h=0.07)
forbidden_shade(a, BFLOOR, 1.0)
constraint_line(a, BFLOOR, "brain floor")
a.annotate("", xy=(0.22, 0.87), xytext=(0.22, 0.94),
           arrowprops=dict(arrowstyle="->", color=C_VIOL, lw=1.8))
a.annotate("", xy=(0.76, 0.85), xytext=(0.76, 0.93),
           arrowprops=dict(arrowstyle="->", color=C_VIOL, lw=1.8))
rule_box(a, 0.50, 0.605, "Brain  ↑  Lung")
bottom_note(a, "Lung fragments found above brain floor", C_VIOL)

# Panel (0,1) — Cleaned
a = ax[0][1]
setup(a, "Normal Firing — Cleaned", title_color="#2A5070")
draw_brain(a, 0.50, BY)
draw_lungs(a, 0.50, LY)
bottom_note(a, "Artifacts removed  ✓", C_OK)

# ---------------------------------------------------------------------------
# Row 1 — Hard Violation
# ---------------------------------------------------------------------------
LY2    = 0.70         # lung centre y
LFLOOR = LY2 - 0.33/2  # lung inferior boundary ≈ 0.535
BDY    = 0.26          # displaced brain centre y

# Panel (1,0) — Input
a = ax[1][0]
setup(a, "Hard Violation — Input", title_color="#704428")
draw_lungs(a, 0.50, LY2)
draw_brain(a, 0.50, BDY, ec=C_VIOL, lw=2.5)
forbidden_shade(a, 0.0, LFLOOR)
constraint_line(a, LFLOOR, "lung floor")
a.annotate("", xy=(0.50, LFLOOR + 0.03), xytext=(0.50, BDY + 0.10),
           arrowprops=dict(arrowstyle="->", color=C_VIOL, lw=2.0))
rule_box(a, 0.50, 0.595, "Brain  ↑  Lung")
bottom_note(a, "Brain CoM entirely below lungs → hard violation", C_VIOL)

# Panel (1,1) — Cleaned
a = ax[1][1]
setup(a, "Hard Violation — Cleaned", title_color="#704428")
draw_lungs(a, 0.50, LY2)
draw_brain(a, 0.50, BDY, fc=C_GHOST, ec="#AAAAAA", lw=1.5, ls="--", alpha=0.45)
bottom_note(a, "Displaced brain removed  ✓", C_OK)

# ---------------------------------------------------------------------------
# Arrows between before/after panels (figure-level)
# ---------------------------------------------------------------------------
for y_fig in [0.665, 0.235]:
    fig.text(0.503, y_fig, "→", ha="center", va="center",
             fontsize=30, color="#AAAAAA", fontweight="bold")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_handles = [
    mpatches.Patch(fc=C_BRAIN, ec=C_EDGE,  lw=1.2, label="Brain"),
    mpatches.Patch(fc=C_LUNG,  ec=C_EDGE,  lw=1.2, label="Lung"),
    mpatches.Patch(fc=C_LUNG,  ec=C_VIOL,  lw=2.0,
                   label="Artifact blob", linestyle="--"),
    mpatches.Patch(fc=C_GHOST, ec="#AAAAAA", lw=1.2,
                   label="Removed", alpha=0.5),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=11, framealpha=0.92, bbox_to_anchor=(0.5, 0.01))

fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {OUT_PATH}")
