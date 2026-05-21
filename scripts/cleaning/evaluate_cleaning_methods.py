"""
Evaluate poset-based cleaning across MRI wrap-around artifact conditions.

Methods
-------
1. Unidirectional (preliminary method M1): for each pair (i above j), remove non-LCC
   components of i that sit entirely below j's LCC inferior boundary.

2. Symmetric (preliminary method M2): same as (1), plus remove non-LCC components of j
   that sit entirely above i's LCC superior boundary.  Catches both artifact directions.

3. Middle-out + spatial prior (poset-based cleaning — published method): like (2), but
   selects the "real" component using the atlas CoM prior (closest centroid to expected
   atlas position) rather than pure LCC.  Processes constraint pairs ordered from
   most-central structures outward, so already-cleaned central anchors inform the periphery.

4. Center-conflict resolver (experimental CM4): when a hard-order violation is detected
   for pair (i above j), whichever LCC mask has a voxel closer to the SI mid-plane
   (horizontal centre line, index N/2) wins; the other structure is treated as violating
   and is removed for that pair (including its LCC).

Usage
-----
    python scripts/cleaning/evaluate_cleaning_methods.py \
        --data_dir  data/datasets/TotalsegmentatorMRI_dataset_v200 \
        --exp_dir   data/wraparound_experiments/wraparound \
        --poset     data/posets/empirical/totalseg_mri_empirical_poset.json \
        --com       data/structures/totalseg_v2_com.json \
        --subject   s0175 \
        --out_dir   data/wraparound_experiments/wraparound_cleaning_eval \
        --threshold 0.95
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.orientations import aff2axcodes
from scipy.ndimage import label as cc_label

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from anatomy_poset.core.io import load_poset_from_json, PosetFromJson

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_si_info(affine) -> Tuple[int, int]:
    codes = aff2axcodes(affine)
    si_ax = next(i for i, c in enumerate(codes) if c in ("S", "I"))
    si_sign = +1 if codes[si_ax] == "S" else -1
    return si_ax, si_sign


def axis_extent(mask: np.ndarray, si_ax: int) -> Optional[Tuple[int, int]]:
    other = tuple(ax for ax in range(mask.ndim) if ax != si_ax)
    proj = mask.any(axis=other)
    idx = np.where(proj)[0]
    return (int(idx.min()), int(idx.max())) if len(idx) > 0 else None


def centroid_1d(mask: np.ndarray, si_ax: int) -> Optional[float]:
    idx = np.where(mask.any(axis=tuple(ax for ax in range(mask.ndim) if ax != si_ax)))[0]
    return float(idx.mean()) if len(idx) > 0 else None


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return float(2.0 * (p & g).sum() / denom) if denom > 0 else 1.0


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum())
    pp = int(p.sum())
    return tp / pp if pp > 0 else 1.0


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum())
    pos = int(g.sum())
    return tp / pos if pos > 0 else 1.0


def f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """Voxel-wise F1 from precision and recall of ``pred`` vs ``gt``."""
    p = precision(pred, gt)
    r = recall(pred, gt)
    s = p + r
    return float(2.0 * p * r / s) if s > 0 else 0.0


def tp_fp_fn(pred: np.ndarray, gt: np.ndarray):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int((p & g).sum())
    fp = int(p.sum()) - tp
    fn = int(g.sum()) - tp
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def get_vertebrae_landmarks(seg_dir: Path, si_ax: int, si_sign: int,
                            ) -> Tuple[Optional[float], Optional[float]]:
    """Return (vert_sup_vox, vert_inf_vox) from the combined vertebrae mask in seg_dir.

    vert_sup_vox: global voxel index of the anatomically superior edge of vertebrae (≈ C1 top)
    vert_inf_vox: global voxel index of the anatomically inferior edge (≈ L5 bottom)

    Returns (None, None) if the file is missing or empty.
    """
    vert_path = seg_dir / "vertebrae.nii.gz"
    if not vert_path.exists():
        return None, None
    mask = np.asarray(nib.load(str(vert_path)).dataobj, dtype=bool)
    if not mask.any():
        return None, None
    other = tuple(ax for ax in range(mask.ndim) if ax != si_ax)
    proj = mask.any(axis=other)
    idx = np.where(proj)[0]
    if len(idx) == 0:
        return None, None
    lo, hi = int(idx.min()), int(idx.max())
    if si_sign == +1:
        return float(hi), float(lo)   # sup=high index, inf=low index
    else:
        return float(lo), float(hi)   # sup=low index (small=superior), inf=high index


# ---------------------------------------------------------------------------
# Connected component selection
# ---------------------------------------------------------------------------

def get_components(mask: np.ndarray):
    """Returns (labeled_array, n_components, sizes_array, lcc_label)."""
    if not mask.any():
        return None, 0, None, None
    labeled, n = cc_label(mask)
    if n == 0:
        return labeled, 0, None, None
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    lcc_label = int(sizes.argmax())
    return labeled, n, sizes, lcc_label


def select_by_lcc(mask: np.ndarray) -> np.ndarray:
    labeled, n, sizes, lcc = get_components(mask)
    if n == 0:
        return mask.copy()
    return (labeled == lcc).astype(bool)


def select_by_prior(mask: np.ndarray, si_ax: int, expected_local: float,
                    size_dominance: float = 5.0) -> np.ndarray:
    """Pick the component closest to expected_local, but only override the LCC
    when a competing component is within size_dominance× of the LCC.
    This prevents the prior from accidentally picking a tiny noise CC and
    marking the true large component for removal."""
    labeled, n, sizes, lcc = get_components(mask)
    if n <= 1:
        return mask.copy()

    # Sort labels by size descending (excluding background=0)
    sorted_labels = np.argsort(-sizes[1:]) + 1   # labels in size order
    top2_sizes = [sizes[sorted_labels[0]], sizes[sorted_labels[1]]]

    # If the LCC is dominant (>5× the 2nd largest), trust it unconditionally
    if top2_sizes[0] >= size_dominance * top2_sizes[1]:
        return (labeled == lcc).astype(bool)

    # Multiple similarly-sized CCs: use prior to pick the most plausible one
    best_label, best_dist = lcc, float("inf")
    for lab in range(1, n + 1):
        c = centroid_1d((labeled == lab), si_ax)
        if c is not None:
            dist = abs(c - expected_local)
            if dist < best_dist:
                best_dist, best_label = dist, lab
    return (labeled == best_label).astype(bool)


# ---------------------------------------------------------------------------
# Preliminary method M1: unidirectional
# (internal development variant — not published)
# ---------------------------------------------------------------------------

def method1_unidirectional(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    cleaned = {n: m.copy() for n, m in predictions.items()}
    removed = {n: 0 for n in predictions}
    cc_cache: Dict[str, tuple] = {}

    pairs = _get_pairs(poset, threshold)
    for name_i, name_j in pairs:
        if name_i not in cleaned or name_j not in cleaned:
            continue
        if name_j not in cc_cache:
            labeled, n, sizes, lcc = get_components(cleaned[name_j])
            cc_cache[name_j] = (labeled, n, sizes, lcc)
        _, _, _, lcc = cc_cache[name_j]
        if lcc is None:
            continue
        labeled_j = cc_cache[name_j][0]
        anchor = (labeled_j == lcc).astype(bool)
        ext_j = axis_extent(anchor, si_ax)
        if ext_j is None:
            continue
        _remove_violated_components(cleaned, removed, name_i, si_ax, si_sign,
                                    below_limit=ext_j[0], above_limit=None,
                                    cc_cache=cc_cache)
    return cleaned, removed


# ---------------------------------------------------------------------------
# Preliminary method M2: symmetric
# (internal development variant — not published)
# ---------------------------------------------------------------------------

def method2_symmetric(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    cleaned = {n: m.copy() for n, m in predictions.items()}
    removed = {n: 0 for n in predictions}
    cc_cache: Dict[str, tuple] = {}

    pairs = _get_pairs(poset, threshold)
    for name_i, name_j in pairs:
        if name_i not in cleaned or name_j not in cleaned:
            continue

        def lcc_extent(name):
            if name not in cc_cache:
                cc_cache[name] = get_components(cleaned[name])
            labeled, n, sizes, lcc = cc_cache[name]
            if lcc is None:
                return None
            return axis_extent((labeled == lcc).astype(bool), si_ax)

        ext_j = lcc_extent(name_j)
        ext_i = lcc_extent(name_i)

        if ext_j is not None:
            _remove_violated_components(cleaned, removed, name_i, si_ax, si_sign,
                                        below_limit=ext_j[0], above_limit=None,
                                        cc_cache=cc_cache)
        if ext_i is not None:
            _remove_violated_components(cleaned, removed, name_j, si_ax, si_sign,
                                        below_limit=None, above_limit=ext_i[1],
                                        cc_cache=cc_cache)
    return cleaned, removed


# ---------------------------------------------------------------------------
# Poset-based cleaning: middle-out + spatial prior (published method)
# ---------------------------------------------------------------------------

def method3_middle_out_prior(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Middle-out cleaning with center-preference and constraint-consistency.

    No atlas, no crop coordinates, no external prior.  The only inputs are the
    predicted masks, the poset, and the image orientation.

    Component selection: when two similarly-sized components compete, prefer the
    one whose centroid is closest to the image centre (N/2).  Wrap-around ghosts
    appear at the edges; real anatomy occupies the centre.

    Constraint-consistency guard: for pair (i above j), if i's LCC is entirely
    below j's LCC, i's LCC is displaced from its expected (superior) position and
    is therefore the ghost.  All i-components below j's inferior boundary are
    removed (including the LCC itself — protect_anchor=False).  Any real fragment
    of i that already sits above j is preserved and becomes the new LCC for
    subsequent pairs.

    Pair ordering: pairs whose combined observed-LCC midpoint is closest to the
    image centre are processed first, establishing trustworthy central anchors
    before cleaning peripheral structures.
    """
    cleaned = {n: m.copy() for n, m in predictions.items()}
    removed = {n: 0 for n in predictions}

    # Derive image height from predictions — no external crop knowledge needed
    N = next(iter(predictions.values())).shape[si_ax]
    center = N / 2.0

    cc_cache: Dict[str, tuple] = {}

    def _lcc_midpoint(name: str) -> float:
        """Midpoint of LCC extent along si_ax, or image centre if absent."""
        if name not in cleaned or not cleaned[name].any():
            return center
        if name not in cc_cache:
            cc_cache[name] = get_components(cleaned[name])
        labeled, _, sizes, lcc = cc_cache[name]
        if lcc is None:
            return center
        ext = axis_extent((labeled == lcc).astype(bool), si_ax)
        return (ext[0] + ext[1]) / 2.0 if ext is not None else center

    # Order pairs: most central (trustworthy) first
    pairs = _get_pairs(poset, threshold)
    pairs_sorted = sorted(
        pairs,
        key=lambda p: abs((_lcc_midpoint(p[0]) + _lcc_midpoint(p[1])) / 2.0 - center)
    )

    def _get_anchor(name: str):
        """Return (anchor_mask, extent) using the LCC as the candidate anchor.

        The middle-out ordering ensures central structures are already cleaned
        by the time peripheral ones are processed, making the LCC reliable.
        The constraint-consistency guard below handles the remaining case where
        the LCC is actually the ghost (sets anchor=None).
        """
        if not cleaned[name].any():
            return None, None
        if name not in cc_cache:
            cc_cache[name] = get_components(cleaned[name])
        labeled, _, sizes, lcc = cc_cache[name]
        if lcc is None:
            return None, None
        anchor = (labeled == lcc).astype(bool)
        return anchor, axis_extent(anchor, si_ax)

    def _is_entirely_below(ext_a, ext_b) -> bool:
        """True if structure a is anatomically entirely below structure b."""
        if si_sign == +1:
            return ext_a[1] < ext_b[0]   # a's superior edge < b's inferior edge
        else:
            return ext_a[0] > ext_b[1]   # I-axis: a's superior edge > b's inferior edge

    for name_i, name_j in pairs_sorted:
        if name_i not in cleaned or name_j not in cleaned:
            continue

        anchor_i, ext_i = _get_anchor(name_i)
        anchor_j, ext_j = _get_anchor(name_j)

        # Constraint-consistency guard:
        # Pair says i above j, so i's CoM is superior to j's CoM.  If i's LCC is
        # entirely below j's LCC the constraint is violated and i's LCC is displaced
        # from its expected (superior) position → i's LCC is the ghost.
        # Remove all i-components below j's inferior boundary with protect_anchor=False
        # so the ghost LCC is not spared.  Any real fragment of i that sits above j
        # is left intact.  Both extents are then cleared to skip the normal steps.
        if ext_i is not None and ext_j is not None:
            if _is_entirely_below(ext_i, ext_j):
                _remove_violated_components(cleaned, removed, name_i, si_ax, si_sign,
                                            below_limit=ext_j[0], above_limit=None,
                                            cc_cache=cc_cache, protect_anchor=False)
                anchor_i, ext_i = None, None   # skip normal steps for this pair
                anchor_j, ext_j = None, None

        if ext_j is not None:
            _remove_violated_components(cleaned, removed, name_i, si_ax, si_sign,
                                        below_limit=ext_j[0], above_limit=None,
                                        cc_cache=cc_cache, anchor_override=anchor_i)
        if ext_i is not None:
            _remove_violated_components(cleaned, removed, name_j, si_ax, si_sign,
                                        below_limit=None, above_limit=ext_i[1],
                                        cc_cache=cc_cache, anchor_override=anchor_j)

    return cleaned, removed


# ---------------------------------------------------------------------------
# Method CM4: center-conflict resolver
# ---------------------------------------------------------------------------

def _min_abs_si_distance_to_midplane(
    mask: np.ndarray, si_ax: int, center: float
) -> float:
    """Minimum |SI_index − center| over foreground voxels; +inf if mask empty."""
    if mask is None or not mask.any():
        return float("inf")
    coords = np.argwhere(mask)
    si_vals = coords[:, si_ax].astype(np.float64)
    return float(np.min(np.abs(si_vals - center)))


def method4_center_conflict(
    predictions: Dict[str, np.ndarray],
    poset: PosetFromJson,
    si_ax: int,
    si_sign: int,
    threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """CM4: conflict resolution by proximity of mask voxels to the SI mid-plane.

    For pair (i above j), if i is entirely below j (hard violation), compare each LCC
    mask's minimum absolute SI distance to the image centre index N/2. The structure with
    the smaller value dominates; the other is treated as violating and removed
    (protect_anchor=False for the violating side).
    """
    cleaned = {n: m.copy() for n, m in predictions.items()}
    removed = {n: 0 for n in predictions}

    N = next(iter(predictions.values())).shape[si_ax]
    center = N / 2.0
    cc_cache: Dict[str, tuple] = {}

    def _lcc_midpoint(name: str) -> float:
        if name not in cleaned or not cleaned[name].any():
            return center
        if name not in cc_cache:
            cc_cache[name] = get_components(cleaned[name])
        labeled, _, _, lcc = cc_cache[name]
        if lcc is None:
            return center
        ext = axis_extent((labeled == lcc).astype(bool), si_ax)
        return (ext[0] + ext[1]) / 2.0 if ext is not None else center

    def _get_anchor(name: str):
        if not cleaned[name].any():
            return None, None
        if name not in cc_cache:
            cc_cache[name] = get_components(cleaned[name])
        labeled, _, _, lcc = cc_cache[name]
        if lcc is None:
            return None, None
        anchor = (labeled == lcc).astype(bool)
        return anchor, axis_extent(anchor, si_ax)

    def _is_entirely_below(ext_a, ext_b) -> bool:
        if si_sign == +1:
            return ext_a[1] < ext_b[0]
        return ext_a[0] > ext_b[1]

    pairs = _get_pairs(poset, threshold)
    pairs_sorted = sorted(
        pairs,
        key=lambda p: abs((_lcc_midpoint(p[0]) + _lcc_midpoint(p[1])) / 2.0 - center),
    )

    for name_i, name_j in pairs_sorted:
        if name_i not in cleaned or name_j not in cleaned:
            continue

        anchor_i, ext_i = _get_anchor(name_i)
        anchor_j, ext_j = _get_anchor(name_j)

        # Conflict resolution at hard violation: dominant side = LCC voxels closest to SI mid-plane.
        if (
            ext_i is not None
            and ext_j is not None
            and anchor_i is not None
            and anchor_j is not None
            and _is_entirely_below(ext_i, ext_j)
        ):
            dist_i = _min_abs_si_distance_to_midplane(anchor_i, si_ax, center)
            dist_j = _min_abs_si_distance_to_midplane(anchor_j, si_ax, center)

            if dist_i <= dist_j:
                # i is more central/plausible -> j is violating.
                _remove_violated_components(
                    cleaned,
                    removed,
                    name_j,
                    si_ax,
                    si_sign,
                    below_limit=None,
                    above_limit=ext_i[1],
                    cc_cache=cc_cache,
                    protect_anchor=False,
                )
            else:
                # j is more central/plausible -> i is violating.
                _remove_violated_components(
                    cleaned,
                    removed,
                    name_i,
                    si_ax,
                    si_sign,
                    below_limit=ext_j[0],
                    above_limit=None,
                    cc_cache=cc_cache,
                    protect_anchor=False,
                )
            anchor_i, ext_i = None, None
            anchor_j, ext_j = None, None

        if ext_j is not None:
            _remove_violated_components(
                cleaned,
                removed,
                name_i,
                si_ax,
                si_sign,
                below_limit=ext_j[0],
                above_limit=None,
                cc_cache=cc_cache,
                anchor_override=anchor_i,
            )
        if ext_i is not None:
            _remove_violated_components(
                cleaned,
                removed,
                name_j,
                si_ax,
                si_sign,
                below_limit=None,
                above_limit=ext_i[1],
                cc_cache=cc_cache,
                anchor_override=anchor_j,
            )

    return cleaned, removed


# ---------------------------------------------------------------------------
# Shared removal helper
# ---------------------------------------------------------------------------

def _get_pairs(poset: PosetFromJson, threshold: float) -> List[Tuple[str, str]]:
    n = len(poset.structures)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cell = poset.matrix_vertical[i][j]
            if cell is not None and cell >= threshold:
                pairs.append((poset.structures[i].name, poset.structures[j].name))
    return pairs


def _remove_violated_components(
    cleaned: Dict[str, np.ndarray],
    removed: Dict[str, int],
    name: str,
    si_ax: int,
    si_sign: int,
    below_limit: Optional[int],
    above_limit: Optional[int],
    cc_cache: Dict[str, tuple],
    anchor_override: Optional[np.ndarray] = None,
    protect_anchor: bool = True,
) -> bool:
    """Returns True if any voxels were removed (caller should invalidate cc_cache[name]).

    protect_anchor=False: remove ALL components that violate the limit, including the LCC.
    Used when the LCC itself is the ghost and must be cleared.
    """
    mask = cleaned[name]
    if not mask.any():
        return False

    if name not in cc_cache:
        labeled, n, sizes, lcc_label = get_components(mask)
        cc_cache[name] = (labeled, n, sizes, lcc_label)
    labeled, n, sizes, lcc_label = cc_cache[name]
    if n == 0:
        return False

    if not protect_anchor:
        anchor_label = -1   # protect nothing — ghost LCC included
    elif anchor_override is not None:
        overlap = np.bincount(labeled[anchor_override].ravel(), minlength=n + 1)
        overlap[0] = 0
        anchor_label = int(overlap.argmax()) if overlap.max() > 0 else lcc_label
    else:
        anchor_label = lcc_label

    changed = False
    for comp_label in range(1, n + 1):
        if comp_label == anchor_label:
            continue
        comp = labeled == comp_label
        ext = axis_extent(comp, si_ax)
        if ext is None:
            continue
        c_min, c_max = ext

        violated = False
        if below_limit is not None:
            if si_sign == +1:
                violated = violated or (c_max < below_limit)
            else:
                violated = violated or (c_min > below_limit)
        if above_limit is not None:
            if si_sign == +1:
                violated = violated or (c_min > above_limit)
            else:
                violated = violated or (c_max < above_limit)

        if violated:
            removed[name] += int(comp.sum())
            cleaned[name][comp] = False
            changed = True

    if changed:
        cc_cache.pop(name, None)   # invalidate so next call re-labels
    return changed


# ---------------------------------------------------------------------------
# Crop GT to prediction space
# ---------------------------------------------------------------------------

def crop_gt(gt_full: np.ndarray, si_ax: int, si_sign: int,
            crop_lo: int, crop_hi: int) -> np.ndarray:
    slices = [slice(None)] * gt_full.ndim
    slices[si_ax] = slice(crop_lo, crop_hi)
    return gt_full[tuple(slices)]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

CROPS = [
    ("brain_to_heart",  "brain",       "heart"),
    ("heart_to_kidney", "heart",       "kidney_left"),
    ("kidney_to_hip",   "kidney_left", "hip_left"),
]
FALLBACKS = {"kidney_left": "kidney_right", "hip_left": "hip_right"}
# Default sweep — overridden by --d_fracs / --r_vals CLI args
DEFAULT_D_FRACS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DEFAULT_R_VALS  = (0.25, 0.50, 0.75, 1.00)

def build_tags(d_fracs, r_vals):
    return [f"d{int(d*100):03d}_r{int(r*100):03d}" for d in d_fracs for r in r_vals]

TAGS = build_tags(DEFAULT_D_FRACS, DEFAULT_R_VALS)

MARGIN = 5


def compute_crop_window(gt_dir: Path, anchors: Tuple[str, str],
                        si_ax: int, full_height: int) -> Tuple[int, int]:
    lo_vals, hi_vals = [], []
    for name in anchors:
        p = gt_dir / f"{name}.nii.gz"
        if not p.exists():
            name = FALLBACKS.get(name, name)
            p = gt_dir / f"{name}.nii.gz"
        if not p.exists():
            continue
        d = np.asarray(nib.load(str(p)).dataobj).astype(bool)
        ext = axis_extent(d, si_ax)
        if ext:
            lo_vals.append(ext[0])
            hi_vals.append(ext[1])
    lo = max(0, min(lo_vals) - MARGIN)
    hi = min(full_height, max(hi_vals) + MARGIN)
    return lo, hi


def run_evaluation(args, tags: Optional[List[str]] = None) -> List[dict]:
    poset = load_poset_from_json(args.poset)
    data_dir = Path(args.data_dir)
    exp_dir  = Path(args.exp_dir)

    subjects = getattr(args, "subjects", None) or [args.subject]
    all_rows_combined = []
    for subj in subjects:
        print(f"\n{'#'*70}")
        print(f"  SUBJECT: {subj}")
        print(f"{'#'*70}")
        rows = _run_one_subject(args, subj, data_dir, exp_dir, poset, tags=tags)
        all_rows_combined.extend(rows)
    return all_rows_combined


def _run_one_subject(args, subj, data_dir, exp_dir, poset,
                     tags: Optional[List[str]] = None) -> List[dict]:
    gt_dir   = data_dir / subj / "segmentations"

    # Load full MRI for geometry and GT crop window computation
    mri_full = nib.load(str(data_dir / subj / "mri.nii.gz"))
    si_ax, si_sign = get_si_info(mri_full.affine)
    full_height = mri_full.shape[si_ax]   # used only for GT crop bounds, not passed to poset-based cleaning

    # Evaluable structure names (GT ∩ poset) — used for Dice evaluation
    poset_names  = {s.name for s in poset.structures}
    gt_names     = {p.stem.replace(".nii", "") for p in gt_dir.glob("*.nii.gz")}
    eval_names   = sorted(poset_names & gt_names)

    all_rows = []

    for crop_name, anc_a, anc_b in CROPS:
        crop_lo, crop_hi = compute_crop_window(gt_dir, (anc_a, anc_b), si_ax, full_height)
        N = crop_hi - crop_lo
        print(f"\n{'='*70}")
        print(f"  Crop: {crop_name}  lo={crop_lo} hi={crop_hi} N={N}")

        # Pre-load and crop GT masks (constant across tags)
        gt_masks = {}
        for name in eval_names:
            g = np.asarray(nib.load(str(gt_dir / f"{name}.nii.gz")).dataobj).astype(bool)
            gt_crop = crop_gt(g, si_ax, si_sign, crop_lo, crop_hi)
            if gt_crop.any():
                gt_masks[name] = gt_crop
        eval_here = sorted(gt_masks)
        print(f"  Evaluable structures in crop: {len(eval_here)}")

        for tag in (tags or TAGS):
            d_str, r_str = tag.split("_")
            d_frac = int(d_str[1:]) / 100.0
            r_val  = int(r_str[1:]) / 100.0
            pred_dir = exp_dir / subj / crop_name / tag / "segmentations"
            if not pred_dir.exists():
                print(f"    [skip] {tag}: no segmentations directory")
                continue

            # Load ALL predictions (needed so cleaning can use full poset context)
            all_preds: Dict[str, np.ndarray] = {}
            affine = None
            for pred_file in sorted(pred_dir.glob("*.nii.gz")):
                name = pred_file.stem.replace(".nii", "")
                img = nib.load(str(pred_file))
                data = np.asarray(img.dataobj).astype(bool)
                if data.any():
                    all_preds[name] = data
                    if affine is None:
                        affine = img.affine

            if not all_preds:
                print(f"    [skip] {tag}: no predictions")
                continue

            pred_si_ax, pred_si_sign = get_si_info(affine)

            # Apply selected cleaning method.
            # Preliminary methods M1/M2 commented out for speed; re-enable if comparing variants.
            # c1, r1 = method1_unidirectional(all_preds, poset, pred_si_ax, pred_si_sign, args.threshold)
            # c2, r2 = method2_symmetric(all_preds, poset, pred_si_ax, pred_si_sign, args.threshold)
            if args.method == "cm4":
                c3, r3 = method4_center_conflict(all_preds, poset, pred_si_ax, pred_si_sign, args.threshold)
            else:
                c3, r3 = method3_middle_out_prior(all_preds, poset, pred_si_ax, pred_si_sign, args.threshold)
            # Stubs so row-building code below still works
            c1, r1 = all_preds, {n: 0 for n in all_preds}
            c2, r2 = all_preds, {n: 0 for n in all_preds}

            # Evaluate Dice for GT-evaluable structures
            improved = [0, 0, 0]
            degraded = [0, 0, 0]
            fp_removed = [0, 0, 0]   # false positive voxels removed (no GT in crop)
            tp_removed = [0, 0, 0]   # true positive voxels removed (with GT)
            tag_rows = []

            for name in sorted(all_preds):
                has_gt = name in gt_masks and all_preds[name].shape == gt_masks[name].shape

                if has_gt:
                    gt = gt_masks[name]
                    pred0 = all_preds[name]
                    d0 = dice(pred0, gt)
                    d3 = dice(c3[name], gt)
                    f0 = f1(pred0, gt)
                    f3 = f1(c3[name], gt)
                    p0 = precision(pred0, gt)
                    p3 = precision(c3[name], gt)
                    rc = recall(pred0, gt)  # recall unchanged by cleaning
                    tp0, fp0, fn0 = tp_fp_fn(pred0, gt)
                    tp3, _, _ = tp_fp_fn(c3[name], gt)
                    tp_removed_pc = max(0, tp0 - tp3)
                    tp_removed[2] += tp_removed_pc

                    delta3 = d3 - d0
                    delta_f1 = f3 - f0
                    if delta_f1 > 0.0001:
                        improved[2] += 1
                    elif delta_f1 < -0.0001:
                        degraded[2] += 1

                    # NOTE: older CSV files on disk used the column names
                    # dice_m3, delta_m3, precision_m3, delta_prec_m3, vox_removed_m3.
                    # Regenerate with this script to get the current names below.
                    tag_rows.append({
                        "subject": subj,
                        "crop": crop_name,
                        "d_frac": d_frac,
                        "r_val": r_val,
                        "tag": tag,
                        "structure": name,
                        "has_gt": True,
                        "dice_before":      round(d0, 5),
                        "dice_pc":          round(d3, 5),
                        "delta_pc":         round(d3 - d0, 5),
                        "f1_before":        round(f0, 5),
                        "f1_pc":            round(f3, 5),
                        "delta_f1":         round(delta_f1, 5),
                        "precision_before": round(p0, 5),
                        "precision_pc":     round(p3, 5),
                        "delta_prec_pc":    round(p3 - p0, 5),
                        "recall_before":    round(rc, 5),
                        "tp_before":        tp0,
                        "fp_before":        fp0,
                        "fn_before":        fn0,
                        "vox_before":       int(pred0.sum()),
                        "vox_removed_pc":   r3.get(name, 0),
                        "tp_removed_pc":    tp_removed_pc,
                    })
                else:
                    fp_removed[2] += r3.get(name, 0)

            if tag_rows:
                md  = np.mean([r["delta_f1"]     for r in tag_rows])
                mp  = np.mean([r["delta_prec_pc"] for r in tag_rows])
                print(f"    {tag}  n={len(tag_rows):2d} | "
                      f"ΔF1={md:+.4f} | "
                      f"ΔPrec={mp:+.4f} | "
                      f"improved={improved[2]} degraded={degraded[2]} "
                      f"TP_removed={tp_removed[2]} FP_removed={fp_removed[2]}")
            all_rows.extend(tag_rows)

    return all_rows


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def make_report(df, out_dir: Path, n_total: int, has_prec: bool,
                thresh: float = 0.0001, poset_threshold: float = 0.95) -> None:
    import pandas as pd

    def pct(n): return f"{100*n/n_total:.1f}%"

    def section_table(grp_col, label):
        rows_md = []
        groups = sorted(df[grp_col].unique())
        header = f"| {label} | Mean ΔDice | Median ΔDice | Imp↑ | Deg↓ | Net |"
        if has_prec:
            header += " Mean ΔPrec | Prec Imp↑ | Prec Deg↓ | Prec Net |"
        rows_md.append(header)
        sep = "|---|---|---|---|---|---|"
        if has_prec:
            sep += "---|---|---|---|"
        rows_md.append(sep)
        for g in groups:
            sub = df[df[grp_col] == g]
            n = len(sub)
            imp  = (sub["delta_pc"] >  thresh).sum()
            deg  = (sub["delta_pc"] < -thresh).sum()
            row  = (f"| {g} | {sub['delta_pc'].mean():+.5f} | {sub['delta_pc'].median():+.5f} |"
                    f" {imp} ({100*imp/n:.1f}%) | {deg} ({100*deg/n:.1f}%) | {imp-deg:+d} |")
            if has_prec:
                pimp = (sub["delta_prec_pc"] >  thresh).sum()
                pdeg = (sub["delta_prec_pc"] < -thresh).sum()
                row += (f" {sub['delta_prec_pc'].mean():+.5f} |"
                        f" {pimp} ({100*pimp/n:.1f}%) | {pdeg} ({100*pdeg/n:.1f}%) | {pimp-pdeg:+d} |")
            rows_md.append(row)
        return "\n".join(rows_md)

    lines = [
        "# Poset-Based Cleaning Evaluation Report",
        "",
        f"**Total structure×condition pairs:** {n_total}  ",
        f"**Poset constraint threshold:** {poset_threshold} "
        f"({'all constraints' if poset_threshold == 1.0 else f'≥{int(poset_threshold*100)}% probability'})  ",
        "",
        "---",
        "## Overall",
        "",
        f"| Metric | Mean | Median | Std | Improved | Degraded | Net |",
        f"|--------|------|--------|-----|----------|----------|-----|",
    ]
    for col, name in [("delta_pc", "Δ Dice")] + ([("delta_prec_pc", "Δ Precision")] if has_prec else []):
        imp = (df[col] > thresh).sum()
        deg = (df[col] < -thresh).sum()
        lines.append(
            f"| {name} | {df[col].mean():+.5f} | {df[col].median():+.5f} | {df[col].std():.5f}"
            f" | {imp} ({pct(imp)}) | {deg} ({pct(deg)}) | {imp-deg:+d} |"
        )

    lines += ["", "---", "## By shift fraction (d)", "", section_table("d_frac", "d")]
    lines += ["", "---", "## By ghost intensity (r)", "", section_table("r_val", "r")]
    lines += ["", "---", "## By crop region", "", section_table("crop", "crop")]

    lines += ["", "---", "## Per structure (sorted by net Dice improvement)", ""]
    struct_col = ["structure", "delta_pc"]
    if has_prec:
        struct_col.append("delta_prec_pc")
    sg = df.groupby("structure")[struct_col[1:]].agg(
        mean_dice=("delta_pc", "mean"),
        imp_dice=("delta_pc", lambda x: (x > thresh).sum()),
        deg_dice=("delta_pc", lambda x: (x < -thresh).sum()),
    )
    if has_prec:
        sg["mean_prec"] = df.groupby("structure")["delta_prec_pc"].mean()
        sg["imp_prec"]  = df.groupby("structure")["delta_prec_pc"].apply(lambda x: (x > thresh).sum())
        sg["deg_prec"]  = df.groupby("structure")["delta_prec_pc"].apply(lambda x: (x < -thresh).sum())
    sg["net_dice"] = sg["imp_dice"] - sg["deg_dice"]
    sg = sg.sort_values("net_dice", ascending=False)

    hdr = "| Structure | Mean ΔDice | Imp↑ | Deg↓ | Net |"
    sep = "|-----------|-----------|------|------|-----|"
    if has_prec:
        hdr += " Mean ΔPrec | Prec Imp↑ | Prec Deg↓ |"
        sep += "-----------|-----------|-----------|"
    lines += [hdr, sep]
    for struct, row in sg.iterrows():
        n_s = len(df[df["structure"] == struct])
        r = (f"| {struct} | {row['mean_dice']:+.5f} |"
             f" {int(row['imp_dice'])} ({100*row['imp_dice']/n_s:.1f}%) |"
             f" {int(row['deg_dice'])} ({100*row['deg_dice']/n_s:.1f}%) |"
             f" {int(row['net_dice']):+d} |")
        if has_prec:
            r += (f" {row['mean_prec']:+.5f} |"
                  f" {int(row['imp_prec'])} ({100*row['imp_prec']/n_s:.1f}%) |"
                  f" {int(row['deg_prec'])} ({100*row['deg_prec']/n_s:.1f}%) |")
        lines.append(r)

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report → {report_path}")


# ---------------------------------------------------------------------------
# Plotting  (poset-based cleaning only — preliminary methods M1/M2 commented out)
# ---------------------------------------------------------------------------

def make_plots(rows: List[dict], out_dir: Path, poset_threshold: float = 0.95) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    PC_COLOR  = "#55A868"
    R_COLORS  = {0.25: "#a8d5e2", 0.50: "#4a9fb5", 0.75: "#1f6d84", 1.00: "#0a3e50"}
    D_COLORS  = {0.05: "#fddbc7", 0.10: "#f4a582", 0.15: "#d6604d",
                 0.20: "#b2182b", 0.25: "#8b0000", 0.30: "#67001f",
                 0.35: "#ffeda0", 0.40: "#feb24c", 0.45: "#f03b20", 0.50: "#bd0026"}

    r_vals  = sorted(df["r_val"].unique())
    d_fracs = sorted(df["d_frac"].unique())

    # helper: grouped bar chart
    def grouped_bar(ax, pivot, group_colors, xlabel, ylabel, title, col_prefix="r"):
        """pivot: DataFrame where rows=x-groups, columns=bar-groups."""
        n_groups = len(pivot)
        n_bars   = len(pivot.columns)
        width    = 0.8 / n_bars
        x        = np.arange(n_groups)
        for k, col in enumerate(pivot.columns):
            offset = (k - n_bars / 2 + 0.5) * width
            color  = group_colors.get(col, "#888888")
            ax.bar(x + offset, pivot[col].values, width=width * 0.9,
                   color=color, alpha=0.85, label=f"{col_prefix}={col}" if isinstance(col, float) else str(col))
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in pivot.index], fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)

    # --- 1. ΔDice by d, grouped bars per r ---
    pivot_d_r = df.groupby(["d_frac", "r_val"])["delta_pc"].mean().unstack("r_val")
    fig, ax = plt.subplots(figsize=(12, 4))
    grouped_bar(ax, pivot_d_r, R_COLORS, "d (shift fraction)", "Mean Δ Dice (poset-based cleaning)",
                "Poset-based cleaning: Δ Dice by shift fraction, split by ghost intensity r")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_delta_by_d.png", dpi=150)
    plt.close()

    # --- 2. ΔDice by r, grouped bars per d ---
    pivot_r_d = df.groupby(["r_val", "d_frac"])["delta_pc"].mean().unstack("d_frac")
    fig, ax = plt.subplots(figsize=(10, 4))
    grouped_bar(ax, pivot_r_d, D_COLORS, "r (ghost intensity)", "Mean Δ Dice (poset-based cleaning)",
                "Poset-based cleaning: Δ Dice by ghost intensity, split by shift fraction d", col_prefix="d")
    plt.tight_layout()
    plt.savefig(out_dir / "bar_delta_by_r.png", dpi=150)
    plt.close()

    # --- 3. ΔDice by crop, grouped bars per r ---
    pivot_crop_r = df.groupby(["crop", "r_val"])["delta_pc"].mean().unstack("r_val")
    fig, ax = plt.subplots(figsize=(9, 4))
    grouped_bar(ax, pivot_crop_r, R_COLORS, "Crop region", "Mean Δ Dice (poset-based cleaning)",
                "Poset-based cleaning: Δ Dice by crop region, split by ghost intensity r")
    ax.set_xticklabels([c.replace("_to_", "→") for c in pivot_crop_r.index], fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "bar_delta_by_crop.png", dpi=150)
    plt.close()

    # --- 4. Box plot: ΔDice and ΔPrecision for poset-based cleaning ---
    has_prec = "delta_prec_pc" in df.columns
    ncols = 2 if has_prec else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes = [axes] if ncols == 1 else axes
    for ax, col, ylabel in zip(
        axes,
        ["delta_pc"] + (["delta_prec_pc"] if has_prec else []),
        ["Δ Dice (after − before)"] + (["Δ Precision (after − before)"] if has_prec else []),
    ):
        bp = ax.boxplot([df[col].values], patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(PC_COLOR); bp["boxes"][0].set_alpha(0.7)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticklabels(["poset-based cleaning"], fontsize=9)
        ax.set_ylabel(ylabel)
    plt.suptitle("Poset-based cleaning improvement distribution — all conditions & structures")
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_delta_dice.png", dpi=150)
    plt.close()

    def make_heatmap(ax, pivot, title, cbar_label):
        vmax = max(abs(pivot.values).max(), 1e-6)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"r={v}" for v in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"d={v}" for v in pivot.index], fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:+.4f}", ha="center", va="center", fontsize=6,
                        color="black" if abs(val) < 0.6 * vmax else "white")
        plt.colorbar(im, ax=ax, label=cbar_label)

    # --- 5. Side-by-side heatmaps: ΔDice and ΔPrecision ---
    pivot_dice_heat = df.groupby(["d_frac", "r_val"])["delta_pc"].mean().unstack("r_val")
    if has_prec:
        pivot_prec_heat = df.groupby(["d_frac", "r_val"])["delta_prec_pc"].mean().unstack("r_val")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        make_heatmap(axes[0], pivot_dice_heat, "Poset-based cleaning: Mean Δ Dice per (d, r)", "Mean Δ Dice")
        make_heatmap(axes[1], pivot_prec_heat, "Poset-based cleaning: Mean Δ Precision per (d, r)", "Mean Δ Precision")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        make_heatmap(ax, pivot_dice_heat, "Poset-based cleaning: Mean Δ Dice per (d, r)", "Mean Δ Dice")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_delta_d_r.png", dpi=150)
    plt.close()

    if has_prec:
        # --- 6. ΔPrecision by d, grouped bars per r ---
        pivot_prec_d_r = df.groupby(["d_frac", "r_val"])["delta_prec_pc"].mean().unstack("r_val")
        fig, ax = plt.subplots(figsize=(12, 4))
        grouped_bar(ax, pivot_prec_d_r, R_COLORS, "d (shift fraction)", "Mean Δ Precision (poset-based cleaning)",
                    "Poset-based cleaning: Δ Precision by shift fraction, split by ghost intensity r")
        plt.tight_layout()
        plt.savefig(out_dir / "bar_prec_by_d.png", dpi=150)
        plt.close()

        # --- 7. ΔPrecision by r, grouped bars per d ---
        pivot_prec_r_d = df.groupby(["r_val", "d_frac"])["delta_prec_pc"].mean().unstack("d_frac")
        fig, ax = plt.subplots(figsize=(10, 4))
        grouped_bar(ax, pivot_prec_r_d, D_COLORS, "r (ghost intensity)", "Mean Δ Precision (poset-based cleaning)",
                    "Poset-based cleaning: Δ Precision by ghost intensity, split by shift fraction d", col_prefix="d")
        plt.tight_layout()
        plt.savefig(out_dir / "bar_prec_by_r.png", dpi=150)
        plt.close()

        # --- 8. ΔPrecision by crop, grouped bars per r ---
        pivot_prec_crop = df.groupby(["crop", "r_val"])["delta_prec_pc"].mean().unstack("r_val")
        fig, ax = plt.subplots(figsize=(9, 4))
        grouped_bar(ax, pivot_prec_crop, R_COLORS, "Crop region", "Mean Δ Precision (poset-based cleaning)",
                    "Poset-based cleaning: Δ Precision by crop region, split by ghost intensity r")
        ax.set_xticklabels([c.replace("_to_", "→") for c in pivot_prec_crop.index], fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "bar_prec_by_crop.png", dpi=150)
        plt.close()

        # --- 9. Scatter: ΔDice vs ΔPrecision per structure ---
        mean_per_struct = df.groupby("structure")[["delta_pc", "delta_prec_pc"]].mean().reset_index()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(mean_per_struct["delta_pc"], mean_per_struct["delta_prec_pc"],
                   s=20, alpha=0.6, color=PC_COLOR)
        lim_x = max(abs(mean_per_struct["delta_pc"]).max() * 1.1, 0.01)
        lim_y = max(abs(mean_per_struct["delta_prec_pc"]).max() * 1.1, 0.01)
        ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y)
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel("Mean Δ Dice (poset-based cleaning)", fontsize=10)
        ax.set_ylabel("Mean Δ Precision (poset-based cleaning)", fontsize=10)
        ax.set_title("Δ Dice vs Δ Precision per structure (poset-based cleaning)", fontsize=10)
        for _, row in mean_per_struct.iterrows():
            if abs(row["delta_pc"]) > 0.005 or abs(row["delta_prec_pc"]) > 0.01:
                ax.annotate(row["structure"], (row["delta_pc"], row["delta_prec_pc"]),
                            fontsize=5, alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_dir / "scatter_dice_vs_prec.png", dpi=150)
        plt.close()

    # --- improved/degraded count helper ---
    THRESH = 0.0001

    def count_stacked_bar(ax, grp_col, delta_col, title, xlabel):
        """Stacked bar: green=improved, red=degraded, gray=neutral, per group."""
        groups = sorted(df[grp_col].unique())
        improved = [( df[df[grp_col]==g][delta_col] >  THRESH).sum() for g in groups]
        degraded = [( df[df[grp_col]==g][delta_col] < -THRESH).sum() for g in groups]
        neutral  = [( df[df[grp_col]==g][delta_col].between(-THRESH, THRESH)).sum() for g in groups]
        x = np.arange(len(groups))
        ax.bar(x, improved, color="#55A868", alpha=0.85, label="improved")
        ax.bar(x, neutral,  bottom=improved, color="#cccccc", alpha=0.6, label="neutral")
        ax.bar(x, degraded, bottom=[i+n for i,n in zip(improved,neutral)],
               color="#C44E52", alpha=0.85, label="degraded")
        ax.set_xticks(x)
        xlabels = [str(g).replace("_to_","→") for g in groups]
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("# structure×condition pairs", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    def count_diverging_bar(ax, grp_col, delta_col, title, xlabel):
        """Diverging bar: improved above 0, degraded below 0, net annotated."""
        groups = sorted(df[grp_col].unique())
        improved = np.array([(df[df[grp_col]==g][delta_col] >  THRESH).sum() for g in groups], float)
        degraded = np.array([(df[df[grp_col]==g][delta_col] < -THRESH).sum() for g in groups], float)
        x = np.arange(len(groups))
        ax.bar(x,  improved, color="#55A868", alpha=0.85, label="improved")
        ax.bar(x, -degraded, color="#C44E52", alpha=0.85, label="degraded")
        ax.axhline(0, color="black", linewidth=0.8)
        for xi, imp, deg in zip(x, improved, degraded):
            net = int(imp - deg)
            ax.text(xi, max(imp, 1) + 0.5, f"net\n{net:+d}", ha="center", va="bottom",
                    fontsize=6, color="black")
        ax.set_xticks(x)
        xlabels = [str(g).replace("_to_","→") for g in groups]
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("# pairs", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)

    for delta_col, metric, fname_suffix in [
        ("delta_pc",      "Dice",      "dice"),
        ("delta_prec_pc", "Precision", "prec"),
    ]:
        if delta_col not in df.columns:
            continue

        # stacked counts by d, r, crop
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        count_stacked_bar(axes[0], "d_frac", delta_col,
                          f"Δ{metric} counts by shift d", "d (shift fraction)")
        count_stacked_bar(axes[1], "r_val",  delta_col,
                          f"Δ{metric} counts by ghost r", "r (ghost intensity)")
        count_stacked_bar(axes[2], "crop",   delta_col,
                          f"Δ{metric} counts by crop",    "Crop")
        plt.suptitle(f"Poset-based cleaning: improved / neutral / degraded counts ({metric})", y=1.01)
        plt.tight_layout()
        plt.savefig(out_dir / f"counts_stacked_{fname_suffix}.png", dpi=150)
        plt.close()

        # diverging counts by d
        fig, ax = plt.subplots(figsize=(12, 4))
        count_diverging_bar(ax, "d_frac", delta_col,
                            f"Poset-based cleaning: net Δ{metric} improvement count by d", "d (shift fraction)")
        plt.tight_layout()
        plt.savefig(out_dir / f"counts_diverging_{fname_suffix}_by_d.png", dpi=150)
        plt.close()

    # per-structure degradation breakdown
    struct_counts = df.groupby("structure").apply(
        lambda g: pd.Series({
            "improved_dice":  (g["delta_pc"]      >  THRESH).sum(),
            "degraded_dice":  (g["delta_pc"]      < -THRESH).sum(),
            "improved_prec":  (g["delta_prec_pc"] >  THRESH).sum() if "delta_prec_pc" in g else 0,
            "degraded_prec":  (g["delta_prec_pc"] < -THRESH).sum() if "delta_prec_pc" in g else 0,
        })
    ).reset_index()
    struct_counts["net_dice"] = struct_counts["improved_dice"] - struct_counts["degraded_dice"]
    struct_counts_sorted = struct_counts.sort_values("net_dice")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, imp_col, deg_col, title in [
        (axes[0], "improved_dice", "degraded_dice", "Dice"),
        (axes[1], "improved_prec", "degraded_prec", "Precision"),
    ]:
        y = np.arange(len(struct_counts_sorted))
        ax.barh(y,  struct_counts_sorted[imp_col].values, color="#55A868", alpha=0.85, label="improved")
        ax.barh(y, -struct_counts_sorted[deg_col].values, color="#C44E52", alpha=0.85, label="degraded")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(struct_counts_sorted["structure"].values, fontsize=5)
        ax.set_xlabel("# conditions", fontsize=9)
        ax.set_title(f"Poset-based cleaning: per-structure Δ{title} counts\n(sorted by net Dice)", fontsize=9)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "counts_per_structure.png", dpi=150)
    plt.close()

    # --- final Summary + markdown report ---
    n_total = len(df)
    make_report(df, out_dir, n_total, has_prec, THRESH, poset_threshold)
    print(f"\nPlots + report saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir",  default="data/datasets/TotalsegmentatorMRI_dataset_v200")
    p.add_argument("--exp_dir",   default="data/wraparound_experiments/wraparound")
    p.add_argument("--poset",     default="data/posets/empirical/totalseg_mri_empirical_poset.json")
    p.add_argument("--subject",   default="s0175",
                   help="Single subject (ignored if --subjects is given)")
    p.add_argument("--subjects",  nargs="+", default=None,
                   help="One or more subjects, e.g. s0175 s0236 s0219")
    p.add_argument("--out_dir",   default="data/wraparound_experiments/wraparound_cleaning_eval")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Min poset probability to count as hard constraint (default: 0.95)")
    p.add_argument("--method", choices=["cm3", "cm4"], default="cm3",
                   help="Cleaning method: cm3 (published) or cm4 (experimental center-conflict)")
    p.add_argument("--d_fracs", type=float, nargs="+", default=None,
                   help="Shift fractions to evaluate, e.g. 0.05 0.10 0.25 0.50 (default: all 10)")
    p.add_argument("--r_vals", type=float, nargs="+", default=None,
                   help="Ghost intensities to evaluate, e.g. 0.25 0.50 0.75 1.00 (default: all 4)")
    args = p.parse_args()

    d_fracs = args.d_fracs or DEFAULT_D_FRACS
    r_vals  = args.r_vals  or DEFAULT_R_VALS
    tags    = build_tags(d_fracs, r_vals)
    print(
        f"Evaluating {len(tags)} conditions: d={[f'{d:.2f}' for d in d_fracs]}  "
        f"r={r_vals}  method={args.method}"
    )

    rows = run_evaluation(args, tags=tags)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "results.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults CSV → {csv_path}")
        make_plots(rows, out_dir, poset_threshold=args.threshold)


if __name__ == "__main__":
    main()
