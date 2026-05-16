"""
Toy 3D volumes illustrating every branch of ``method4_center_conflict``.

Saves one multi-panel PNG with synthetic Lung (i) / Hip (j) masks on a small grid;
SI axis is the last dimension (index grows toward feet in +1 convention used here).

Output:
  results/cm4_visuals/method4_center_conflict_toy_walkthrough.png

Usage:
  python scripts/cleaning/visualize_cm4_method4_toy_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "cleaning"))

from anatomy_poset.core.axis_models import Structure
from anatomy_poset.core.io import PosetFromJson

from evaluate_cleaning_methods import method4_center_conflict

OUT_PATH = PROJECT_ROOT / "results" / "cm4_visuals" / "method4_center_conflict_toy_walkthrough.png"

# 3D grid: XY small, Z = SI for easy reading
SHAPE = (4, 4, 42)
SI_AX = 2
SI_SIGN = +1
THRESH = 1.0


def _poset_two_node() -> PosetFromJson:
    """Lung (index 0) above Hip (index 1) in vertical poset."""
    lung = Structure("Lung", 60.0, 0.0, 0.0)
    hip = Structure("Hip", 30.0, 0.0, 0.0)
    n = 2
    z = [[None] * n for _ in range(n)]
    z[0][1] = 1.0  # Lung above Hip
    empty = [[None] * n for _ in range(n)]
    return PosetFromJson(
        structures=[lung, hip],
        matrix_vertical=z,
        matrix_mediolateral=empty,
        matrix_anteroposterior=empty,
    )


def _blob(shape: tuple[int, int, int], z_lo: int, z_hi: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[:, :, z_lo : z_hi + 1] = True
    return m


def _combine(parts: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(SHAPE, dtype=bool)
    for p in parts:
        out |= p
    return out


def _si_projection(mask: np.ndarray) -> np.ndarray:
    """Collapse XY → length-Z boolean presence."""
    return mask.any(axis=(0, 1))


def _si_strip_rgb(lung: np.ndarray, hip: np.ndarray, band_width: int = 14) -> np.ndarray:
    """Z×W×3 image: **rows = SI (Z index)**, columns = thick line for visibility.

    Lung red, Hip blue, overlap magenta. Use with ``imshow(..., origin="lower")`` so
    image row ``z`` is SI index ``z`` (small Z at bottom of panel, large Z at top).
    """
    pl = _si_projection(lung)
    ph = _si_projection(hip)
    zlen = pl.shape[0]
    img = np.ones((zlen, band_width, 3), dtype=np.float32)
    for z in range(zlen):
        l, h = bool(pl[z]), bool(ph[z])
        if l and h:
            col = (0.85, 0.25, 0.95)
        elif l:
            col = (0.95, 0.25, 0.22)
        elif h:
            col = (0.22, 0.45, 0.98)
        else:
            col = (1.0, 1.0, 1.0)
        img[z, :, :] = col
    return img


def _run_case(
    title: str,
    caption: str,
    lung0: np.ndarray,
    hip0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    preds = {"Lung": lung0.copy(), "Hip": hip0.copy()}
    poset = _poset_two_node()
    cleaned, removed = method4_center_conflict(preds, poset, SI_AX, SI_SIGN, THRESH)
    note = f"removed vox: Lung={removed['Lung']}, Hip={removed['Hip']}"
    return lung0, hip0, cleaned["Lung"], cleaned["Hip"], f"{caption}\n{note}"


def main() -> None:
    cases: list[tuple[str, str, np.ndarray, np.ndarray, str]] = []

    # Case 1 — hard violation, Lung closer to mid-plane → Hip stripped entirely
    lung1 = _blob(SHAPE, 19, 20)  # max 20 < Hip min 22; Lung nearer z=N/2 than Hip
    hip1 = _blob(SHAPE, 22, 24)
    cases.append(
        (
            "1 · Hard violation — Lung wins (closer to SI mid-plane)",
            f"Poset: Lung above Hip. LCC extents impossible (Lung max < Hip min).\n"
            f"SI mid index = {SHAPE[SI_AX] / 2:.1f}. Lung LCC is at least as central as Hip →\n"
            f"Hip is the violator: remove Hip (including its LCC; protect_anchor=False).",
            lung1,
            hip1,
        )
    )

    # Case 2 — hard violation, Hip closer to mid-plane → Lung is stripped
    lung2 = _blob(SHAPE, 0, 2)
    hip2 = _blob(SHAPE, 10, 12)
    cases.append(
        (
            "2 · Hard violation — Hip wins",
            "Same impossible stacking, but Hip LCC sits nearer the mid-plane →\n"
            "Lung is the violator; non-protected removal clears offending Lung components.",
            lung2,
            hip2,
        )
    )

    # Case 3 — normal: Lung has a small inferior satellite; Hip LCC sets below_limit
    lung3 = _combine([_blob(SHAPE, 20, 23), _blob(SHAPE, 2, 3)])  # satellite 2–3, LCC 20–23
    hip3 = _blob(SHAPE, 8, 10)
    cases.append(
        (
            "3 · Normal — trim Lung below Hip inferior bound",
            "No hard violation (Lung LCC not entirely below Hip). below_limit = min SI of Hip LCC.\n"
            "Non-anchor Lung components with max SI < that limit are removed; Lung LCC kept.",
            lung3,
            hip3,
        )
    )

    # Case 4 — normal: Hip has superior ghost above Lung superior bound
    lung4 = _blob(SHAPE, 5, 8)
    hip4 = _combine([_blob(SHAPE, 3, 5), _blob(SHAPE, 10, 11)])  # LCC 3–5, ghost 10–11
    cases.append(
        (
            "4 · Normal — trim Hip above Lung superior bound",
            "above_limit = max SI of Lung LCC. Non-anchor Hip components with min SI > limit removed;\n"
            "Hip LCC stays if it does not violate.",
            lung4,
            hip4,
        )
    )

    # Case 5 — compliant single blobs only; no removals
    lung5 = _blob(SHAPE, 22, 26)
    hip5 = _blob(SHAPE, 8, 12)
    cases.append(
        (
            "5 · Compliant pair — no voxels removed",
            "Single LCC each, anatomically consistent extents for this poset edge →\n"
            "violations array empty; masks unchanged.",
            lung5,
            hip5,
        )
    )

    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 2, figsize=(8, 2.8 * n_cases))
    if n_cases == 1:
        axes = np.array([axes])

    for row, (short_title, long_cap, l0, h0) in enumerate(cases):
        lb, hb, la, ha, footer = _run_case(short_title, long_cap, l0, h0)
        z_n = SHAPE[SI_AX]
        for c in range(2):
            ax = axes[row, c]
            strip = _si_strip_rgb(lb, hb) if c == 0 else _si_strip_rgb(la, ha)
            ax.imshow(strip, aspect="equal", interpolation="nearest", origin="lower")
            ax.set_xlim(-0.5, strip.shape[1] - 0.5)
            ax.set_ylim(-0.5, z_n - 0.5)
            ax.set_xticks([])
            ax.set_yticks([0, z_n // 2, z_n - 1])
            ax.set_yticklabels(["0", str(z_n // 2), str(z_n - 1)], fontsize=8)
            if c == 0:
                ax.set_ylabel("SI index (Z)", fontsize=9)
            else:
                ax.set_ylabel("")
        axes[row, 0].set_title("Before cleaning", fontsize=10)
        axes[row, 1].set_title("After method4_center_conflict", fontsize=10)

        axes[row, 0].text(
            -0.02,
            1.18,
            short_title + "\n" + footer + "\n" + long_cap,
            transform=axes[row, 0].transAxes,
            fontsize=8.5,
            va="bottom",
            ha="left",
            linespacing=1.22,
        )

    fig.suptitle(
        "Toy walkthrough: method4_center_conflict\n"
        "SI = vertical (Z, last axis), si_sign = +1  |  Red = Lung, Blue = Hip, Magenta = overlap",
        fontsize=11,
        y=0.995,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.07, hspace=0.72, wspace=0.2)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
