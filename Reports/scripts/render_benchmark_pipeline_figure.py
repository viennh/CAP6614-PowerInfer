#!/usr/bin/env python3
"""Render Reports/figures/benchmark_pipeline.png (four-phase ReluLLaMA benchmark pipeline)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    out = repo / "Reports" / "figures" / "benchmark_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 10), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    phases = [
        (
            "Phase 1 — Prerequisites & artifacts",
            [
                "ReluLLaMA .powerinfer.gguf + dense .gguf",
                "activation_*.pt (profiling tensors) · CUDA builds",
            ],
        ),
        (
            "Phase 2 — Activation distribution profiling",
            [
                "phase2_profile_neurons.py [--preset 13b]",
                "→ Reports/results/phase2_profiling(_13b)/",
            ],
        ),
        (
            "Phase 3 — Inference throughput",
            [
                "phase3_benchmark.py (PowerInfer, llama.cpp)",
                "phase3_vllm_benchmark.py (GPU) · plots & CSV/JSON",
            ],
        ),
        (
            "Phase 4 — Hot-neuron overlap across prompts",
            [
                "phase4_neuron_variation.py [--preset 13b]",
                "→ overlap_results.json · Jaccard & density figures",
            ],
        ),
    ]

    n = len(phases)
    margin_top, margin_bot = 0.085, 0.04
    gap = 0.022
    usable = 1.0 - margin_top - margin_bot - (n - 1) * gap
    box_h = usable / n
    x0, w = 0.06, 0.88
    y_top = 1.0 - margin_top - 0.042

    for i, (title, lines) in enumerate(phases):
        y_bot = y_top - box_h
        ax.add_patch(
            FancyBboxPatch(
                (x0, y_bot),
                w,
                box_h,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                linewidth=1.15,
                edgecolor="#2c3e50",
                facecolor="#ecf0f1",
                mutation_aspect=0.35,
            )
        )
        ax.text(
            x0 + w / 2,
            y_bot + box_h - 0.009,
            title,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="#1a252f",
        )
        ax.text(
            x0 + w / 2,
            y_bot + 0.02,
            "\n".join(lines),
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#34495e",
            linespacing=1.35,
        )
        if i < n - 1:
            y_next_top = y_bot - gap
            ax.add_patch(
                FancyArrowPatch(
                    (0.5, y_bot),
                    (0.5, y_next_top),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1.15,
                    color="#7f8c8d",
                    clip_on=False,
                )
            )
        y_top = y_bot - gap

    ax.text(
        0.5,
        0.985,
        "ReluLLaMA benchmark pipeline (four phases)",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#1a252f",
    )

    fig.savefig(out, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
