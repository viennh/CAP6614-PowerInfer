#!/usr/bin/env python3
"""
Generate 3-way comparison plots for Phase 2 profiling across ReluLLaMA-7B, 13B, and 70B.

Usage:
  python Reports/scripts/render_phase2_3way_comparison.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv_data(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        print(f"ERROR: empty CSV {csv_path}", file=sys.stderr)
        sys.exit(1)
    nl = len(rows)
    layers = np.arange(nl)
    pct_80 = [float(row["Hot@80% %"]) for row in rows]
    pct_90 = [float(row["Hot@90% %"]) for row in rows]
    pct_95 = [float(row["Hot@95% %"]) for row in rows]
    gini = [float(row["Gini"]) for row in rows]
    return layers, pct_80, pct_90, pct_95, gini, nl


def main() -> None:
    base_dir = Path("Reports/results")
    models = [
        ("phase2_profiling", "ReluLLaMA-7B"),
        ("phase2_profiling_13b", "ReluLLaMA-13B"),
        ("phase2_profiling_70b", "ReluLLaMA-70B"),
    ]
    
    data = {}
    for folder, label in models:
        csv_path = base_dir / folder / "layer_summary.csv"
        if not csv_path.exists():
            print(f"ERROR: {csv_path} not found", file=sys.stderr)
            sys.exit(1)
        data[label] = load_csv_data(csv_path)
    
    output_dir = base_dir / "phase2_3way_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 4: Coverage bars comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    width = 0.2
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    thresholds = ['80%', '90%', '95%']
    for i, (label, (layers, pct_80, pct_90, pct_95, gini, nl)) in enumerate(data.items()):
        ax.bar(layers + (i-1)*width, pct_80, width, label=f"{label} 80%", color=colors[0], alpha=0.7)
        ax.bar(layers + (i-1)*width, pct_90, width, label=f"{label} 90%", color=colors[1], alpha=0.7, bottom=pct_80)
        ax.bar(layers + (i-1)*width, pct_95, width, label=f"{label} 95%", color=colors[2], alpha=0.7, bottom=[a+b for a,b in zip(pct_80, pct_90)])
    
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("% of Neurons Required", fontsize=12)
    ax.set_title("3-Way Comparison: Neurons Needed for Coverage Thresholds per Layer", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p4 = output_dir / "plot4_3way_coverage_bars.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"Saved: {p4}")
    
    # Plot 5: Gini by layer comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#E91E63', '#9C27B0', '#3F51B5']
    for i, (label, (layers, pct_80, pct_90, pct_95, gini, nl)) in enumerate(data.items()):
        ax.plot(range(nl), gini, marker="o", color=colors[i], linewidth=2, markersize=5, label=label)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title("3-Way Comparison: Activation Skewness (Gini) per Layer", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p5 = output_dir / "plot5_3way_gini_by_layer.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"Saved: {p5}")


if __name__ == "__main__":
    main()