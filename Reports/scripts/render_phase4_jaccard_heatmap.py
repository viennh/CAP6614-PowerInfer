#!/usr/bin/env python3
"""
Build a Jaccard heatmap PNG from phase4 overlap_results.json (no model required).

Usage:
  python Reports/scripts/render_phase4_jaccard_heatmap.py \\
    Reports/results/phase4_neuron_variation_13b/overlap_results.json \\
    Reports/results/phase4_neuron_variation_13b/plot1_jaccard_heatmap.png \\
    "ReluLLaMA-13B"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("overlap_json", type=Path)
    p.add_argument("output_png", type=Path)
    p.add_argument("title_suffix", nargs="?", default="ReluLLaMA-13B")
    args = p.parse_args()

    data = json.loads(args.overlap_json.read_text())
    cats = data["categories"]
    J = np.array(data["jaccard_matrix"])
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(J, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(cats)))
    ax.set_yticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_yticklabels(cats)
    ax.set_title(f"{args.title_suffix}: Pairwise Jaccard (hot neuron sets)", fontsize=13)
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(
                j,
                i,
                f"{J[i, j]:.2f}",
                ha="center",
                va="center",
                color="w" if J[i, j] < 0.55 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, label="Jaccard")
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
