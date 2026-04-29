#!/usr/bin/env python3
"""
Rebuild Phase 2 figures plot4 (coverage bars) and plot5 (Gini by layer) from an
existing layer_summary.csv (no activation_*.pt required).

Usage:
  python Reports/scripts/render_phase2_plots_from_layer_summary.py \\
    Reports/results/phase2_profiling_13b/layer_summary.csv \\
    Reports/results/phase2_profiling_13b \\
    "ReluLLaMA-13B"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Phase2 plot4/plot5 from layer_summary.csv")
    p.add_argument("csv_path", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("label", nargs="?", default="ReluLLaMA-13B")
    args = p.parse_args()

    rows = []
    with args.csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        print("ERROR: empty CSV", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nl = len(rows)
    layers = np.arange(nl)
    pct_80 = [float(row["Hot@80% %"]) for row in rows]
    pct_90 = [float(row["Hot@90% %"]) for row in rows]
    pct_95 = [float(row["Hot@95% %"]) for row in rows]
    gini = [float(row["Gini"]) for row in rows]
    label = args.label

    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(layers - width, pct_80, width, label="80% coverage", color="#2196F3", alpha=0.85)
    ax.bar(layers, pct_90, width, label="90% coverage", color="#FF9800", alpha=0.85)
    ax.bar(layers + width, pct_95, width, label="95% coverage", color="#4CAF50", alpha=0.85)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("% of Neurons Required", fontsize=12)
    ax.set_title(f"{label}: Neurons Needed for Coverage Thresholds per Layer", fontsize=14)
    ax.set_xticks(layers[:: max(1, nl // 20)])
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p4 = args.output_dir / "plot4_coverage_bars.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"Saved: {p4}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(nl), gini, marker="o", color="#E91E63", linewidth=2, markersize=5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title(f"{label}: Activation Skewness (Gini) per Layer", fontsize=14)
    ax.set_xticks(range(0, nl, max(1, nl // 10)))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p5 = args.output_dir / "plot5_gini_by_layer.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"Saved: {p5}")


if __name__ == "__main__":
    main()
