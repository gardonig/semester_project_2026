#!/usr/bin/env python3
"""Plot mean dice (before and after CM3 cleaning) by d_frac and r_val."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

RESULTS = Path(__file__).parents[2] / "data/experiments/wraparound_v3_eval/results.csv"
OUT = Path(__file__).parents[2] / "data/experiments/wraparound_v3_eval/dice_by_d_r.png"

df = pd.read_csv(RESULTS)
df = df[df["has_gt"] == True]

agg = df.groupby(["d_frac", "r_val"])[["dice_before", "dice_m3"]].mean().reset_index()

r_vals = sorted(agg["r_val"].unique())
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(r_vals)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

for ax, col, title in zip(
    axes,
    ["dice_before", "dice_m3"],
    ["Before cleaning", "After CM3 cleaning"],
):
    for r, color in zip(r_vals, colors):
        sub = agg[agg["r_val"] == r].sort_values("d_frac")
        ax.plot(sub["d_frac"], sub[col], marker="o", color=color, label=f"r = {r}")
    ax.set_xlabel("Shift fraction d", fontsize=12)
    ax.set_ylabel("Mean Dice", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title="Ghost intensity (r)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

fig.suptitle("Mean Dice by shift fraction (d) and ghost intensity (r)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT}")
