#!/usr/bin/env python3
"""
Phase 2: Profile Hot/Cold Neuron Distribution for ReluLLaMA (7B / 13B / 70B).

Loads pre-computed per-layer activation counts from the **PowerInfer GGUF** bundles
(not from SparseLLM base weights — those repos have no activation/*.pt), e.g.:
  ReluLLaMA-7B/activation/activation_{0..31}.pt   (32 layers)  ← from PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF
  ReluLLaMA-13B/activation/activation_{0..39}.pt   (40 layers)  ← from PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF
  ReluLLaMA-70B/activation/activation_{0..79}.pt   (80 layers) ← from PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF
and produces:
  1. Summary table (printed + CSV)
  2. Plot 1 — Ranked activation frequency (log-scale) for selected layers
  3. Plot 2 — Cumulative coverage curves (all layers overlaid)
  4. Plot 3 — Heatmap of normalized activation across all layers
  5. Plot 4 — Bar chart: % of neurons needed for 80%/90%/95% coverage per layer
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

THRESHOLDS = [0.80, 0.90, 0.95]


@dataclass(frozen=True)
class Phase2Config:
    num_layers: int
    activation_dir: Path
    output_dir: Path
    label: str


PRESETS: dict[str, Phase2Config] = {
    "7b": Phase2Config(
        num_layers=32,
        activation_dir=Path("ReluLLaMA-7B/activation"),
        output_dir=Path("Reports/results/phase2_profiling"),
        label="ReluLLaMA-7B",
    ),
    "13b": Phase2Config(
        num_layers=40,
        activation_dir=Path("ReluLLaMA-13B/activation"),
        output_dir=Path("Reports/results/phase2_profiling_13b"),
        label="ReluLLaMA-13B",
    ),
    "70b": Phase2Config(
        num_layers=80,
        activation_dir=Path("ReluLLaMA-70B/activation"),
        output_dir=Path("Reports/results/phase2_profiling_70b"),
        label="ReluLLaMA-70B",
    ),
}

_ACTIVATION_MISSING_HINT = """
The files activation_0.pt … are shipped with the PowerInfer Hugging Face **GGUF** repos
(e.g. PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF → ./ReluLLaMA-13B/activation/), not with
SparseLLM/ReluLLaMA-13B (base weights only).

Download example:
  huggingface-cli download --resume-download \\
    --local-dir ReluLLaMA-13B \\
    --local-dir-use-symlinks False \\
    PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF

Then run with default paths, e.g.:  python Reports/scripts/phase2_profile_neurons.py --preset 13b
"""


def load_activations(cfg: Phase2Config):
    """Load per-layer activation tensors and return as a list of numpy arrays."""
    activations = []
    for i in range(cfg.num_layers):
        path = cfg.activation_dir / f"activation_{i}.pt"
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            print(_ACTIVATION_MISSING_HINT, file=sys.stderr)
            sys.exit(1)
        t = torch.load(path, map_location="cpu", weights_only=False)
        activations.append(t.numpy().astype(np.float64))
    return activations


def compute_layer_stats(act):
    """Compute summary statistics for a single layer's activation counts."""
    total = act.sum()
    sorted_act = np.sort(act)[::-1]
    cumulative = np.cumsum(sorted_act) / total
    n = len(act)

    coverage = {}
    for thresh in THRESHOLDS:
        k = int((cumulative < thresh).sum()) + 1
        coverage[thresh] = k

    nonzero = int((act > 0).sum())
    gini = _gini_coefficient(sorted_act)

    return {
        "total": int(total),
        "n_neurons": n,
        "nonzero": nonzero,
        "max": int(act.max()),
        "min": int(act.min()),
        "mean": float(act.mean()),
        "std": float(act.std()),
        "gini": gini,
        "coverage": coverage,
        "sorted": sorted_act,
        "cumulative": cumulative,
    }


def _gini_coefficient(sorted_desc):
    """Gini coefficient from descending-sorted values (1 = maximally unequal)."""
    n = len(sorted_desc)
    vals = sorted_desc[::-1]  # ascending for Gini formula
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * vals) / np.sum(vals) - (n + 1)) / n)


def print_summary_table(all_stats, cfg: Phase2Config):
    """Print and save a summary CSV."""
    header = [
        "Layer", "Neurons", "Nonzero", "Max", "Mean", "Std", "Gini",
        "Hot@80%", "Hot@80% %", "Hot@90%", "Hot@90% %", "Hot@95%", "Hot@95% %",
    ]
    print(f"\n{'='*120}")
    print(f"{cfg.label + ' Activation Profile Summary':^120}")
    print(f"{'='*120}")
    fmt = "{:<6} {:>7} {:>7} {:>8} {:>10.1f} {:>10.1f} {:>6.3f} {:>7} {:>7.1f}% {:>7} {:>7.1f}% {:>7} {:>7.1f}%"
    hdr = "{:<6} {:>7} {:>7} {:>8} {:>10} {:>10} {:>6} {:>7} {:>8} {:>7} {:>8} {:>7} {:>8}"
    print(hdr.format(*header))
    print("-" * 120)

    csv_path = cfg.output_dir / "layer_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, s in enumerate(all_stats):
            n = s["n_neurons"]
            row = [
                i, n, s["nonzero"], s["max"], s["mean"], s["std"], s["gini"],
                s["coverage"][0.80], 100 * s["coverage"][0.80] / n,
                s["coverage"][0.90], 100 * s["coverage"][0.90] / n,
                s["coverage"][0.95], 100 * s["coverage"][0.95] / n,
            ]
            print(fmt.format(*row))
            writer.writerow(row)

    print(f"\nSaved: {csv_path}")


def plot1_ranked_frequency(all_stats, cfg: Phase2Config):
    """Plot ranked neuron activation frequency (log scale) for selected layers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    nl = cfg.num_layers
    layers_to_show = [0, nl // 4, nl // 2, (3 * nl) // 4, nl - 1]

    for i in layers_to_show:
        s = all_stats[i]
        ax.semilogy(
            np.arange(1, s["n_neurons"] + 1),
            s["sorted"],
            label=f"Layer {i}",
            alpha=0.8,
            linewidth=1.2,
        )

    ax.set_xlabel("Neuron Rank (sorted by activation count)", fontsize=12)
    ax.set_ylabel("Activation Count (log scale)", fontsize=12)
    ax.set_title(f"{cfg.label}: Neuron Activation Frequency (Power-Law)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = cfg.output_dir / "plot1_ranked_frequency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot2_cumulative_coverage(all_stats, cfg: Phase2Config):
    """Plot cumulative activation coverage vs. % of neurons for all layers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    nl = cfg.num_layers
    cmap = plt.cm.viridis(np.linspace(0, 1, nl))

    for i, s in enumerate(all_stats):
        n = s["n_neurons"]
        x = 100 * np.arange(1, n + 1) / n
        ax.plot(x, 100 * s["cumulative"], color=cmap[i], alpha=0.6, linewidth=0.8)

    ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% coverage")
    ax.axhline(90, color="orange", linestyle="--", alpha=0.7, label="90% coverage")
    ax.axhline(95, color="green", linestyle="--", alpha=0.7, label="95% coverage")

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, nl - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Layer Index")

    ax.set_xlabel("% of Neurons (ranked by activation count)", fontsize=12)
    ax.set_ylabel("Cumulative Activation Coverage (%)", fontsize=12)
    ax.set_title(f"{cfg.label}: Cumulative Coverage — How Few Neurons Dominate", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = cfg.output_dir / "plot2_cumulative_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot3_heatmap(activations, cfg: Phase2Config):
    """Heatmap of normalized activation intensity across all layers and neurons."""
    n_neurons = len(activations[0])
    n_bins = 200  # bin neurons into groups for visualization
    nl = cfg.num_layers

    heatmap = np.zeros((nl, n_bins))
    for i, act in enumerate(activations):
        sorted_act = np.sort(act)[::-1].astype(np.float64)
        sorted_act /= sorted_act.max()  # normalize to [0, 1]
        bin_size = n_neurons // n_bins
        for b in range(n_bins):
            heatmap[i, b] = sorted_act[b * bin_size : (b + 1) * bin_size].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heatmap, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xlabel("Neuron Rank Bin (left=hottest, right=coldest)", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(f"{cfg.label}: Normalized Activation Heatmap Across Layers", fontsize=14)
    ax.set_yticks(range(0, nl, max(1, nl // 8)))
    n_ticks = 5
    ax.set_xticks(np.linspace(0, n_bins - 1, n_ticks).astype(int))
    ax.set_xticklabels([f"{int(x)}%" for x in np.linspace(0, 100, n_ticks)])
    fig.colorbar(im, ax=ax, label="Normalized Activation", shrink=0.8)
    fig.tight_layout()

    path = cfg.output_dir / "plot3_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot4_coverage_bars(all_stats, cfg: Phase2Config):
    """Bar chart: % of neurons needed for 80%/90%/95% coverage per layer."""
    nl = cfg.num_layers
    layers = np.arange(nl)
    width = 0.25

    pct_80 = [100 * s["coverage"][0.80] / s["n_neurons"] for s in all_stats]
    pct_90 = [100 * s["coverage"][0.90] / s["n_neurons"] for s in all_stats]
    pct_95 = [100 * s["coverage"][0.95] / s["n_neurons"] for s in all_stats]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(layers - width, pct_80, width, label="80% coverage", color="#2196F3", alpha=0.85)
    ax.bar(layers, pct_90, width, label="90% coverage", color="#FF9800", alpha=0.85)
    ax.bar(layers + width, pct_95, width, label="95% coverage", color="#4CAF50", alpha=0.85)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("% of Neurons Required", fontsize=12)
    ax.set_title(f"{cfg.label}: Neurons Needed for Coverage Thresholds per Layer", fontsize=14)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = cfg.output_dir / "plot4_coverage_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot5_gini_by_layer(all_stats, cfg: Phase2Config):
    """Line chart of Gini coefficient per layer (higher = more skewed)."""
    nl = cfg.num_layers
    ginis = [s["gini"] for s in all_stats]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(nl), ginis, marker="o", color="#E91E63", linewidth=2, markersize=6)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title(f"{cfg.label}: Activation Skewness (Gini) per Layer", fontsize=14)
    ax.set_xticks(range(0, nl, max(1, nl // 16)))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = cfg.output_dir / "plot5_gini_by_layer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def parse_args() -> Phase2Config:
    p = argparse.ArgumentParser(
        description="Phase 2: profile hot/cold neuron distribution from activation_*.pt files.",
    )
    p.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="7b",
        help=(
            "7b: 32 layers → ReluLLaMA-7B/activation; "
            "13b: 40 → ReluLLaMA-13B/activation; "
            "70b: 80 → ReluLLaMA-70B/activation"
        ),
    )
    p.add_argument(
        "--activation-dir",
        type=Path,
        help=(
            "Directory containing activation_{0..N-1}.pt (from PowerInfer *-PowerInfer-GGUF download; "
            "SparseLLM base model dirs do not include these files)"
        ),
    )
    p.add_argument("--num-layers", type=int, help="Override: number of layers (files 0..n-1)")
    p.add_argument("--output-dir", type=Path, help="Override: where to write CSV and PNGs")
    p.add_argument("--label", help="Override: model name for plot titles and summary header")
    args = p.parse_args()
    base = PRESETS[args.preset]
    return Phase2Config(
        num_layers=args.num_layers if args.num_layers is not None else base.num_layers,
        activation_dir=args.activation_dir if args.activation_dir is not None else base.activation_dir,
        output_dir=args.output_dir if args.output_dir is not None else base.output_dir,
        label=args.label if args.label is not None else base.label,
    )


def main():
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading activation profiles...")
    activations = load_activations(cfg)
    print(f"Loaded {len(activations)} layers, each with {len(activations[0])} neurons\n")

    print("Computing per-layer statistics...")
    all_stats = [compute_layer_stats(act) for act in activations]

    print_summary_table(all_stats, cfg)

    print("\nGenerating plots...")
    plot1_ranked_frequency(all_stats, cfg)
    plot2_cumulative_coverage(all_stats, cfg)
    plot3_heatmap(activations, cfg)
    plot4_coverage_bars(all_stats, cfg)
    plot5_gini_by_layer(all_stats, cfg)

    print(f"\nAll outputs saved to {cfg.output_dir}/")


if __name__ == "__main__":
    main()
