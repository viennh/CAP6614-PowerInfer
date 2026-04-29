#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 4: Analyze Hot Neuron Variation Across Input Types

Loads SparseLLM ReluLLaMA (7B / 13B / 70B) via transformers, hooks into the FFN
intermediate layer (after ReLU), and records which neurons fire (> 0)
for each prompt category. Then computes overlap metrics and generates
comparison plots.

Layer count and FFN width are read from `model.config` (not hardcoded).

Usage (from the PowerInfer repo root):
    python Reports/scripts/phase4_neuron_variation.py

    python Reports/scripts/phase4_neuron_variation.py --preset 13b --max-new-tokens 32
    python Reports/scripts/phase4_neuron_variation.py --preset 70b --device auto   # needs: pip install accelerate

    # Custom model path
    python Reports/scripts/phase4_neuron_variation.py --model-path ./SparseLLM-ReluLLaMA-7B
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass(frozen=True)
class ModelGeometry:
    """FFN shape for ReluLLaMA / LLaMA (from HF config)."""

    num_layers: int
    n_neurons: int  # intermediate_size (per FFN column)


PRESET_PATHS = {
    "7b": Path("SparseLLM-ReluLLaMA-7B"),
    "13b": Path("SparseLLM-ReluLLaMA-13B"),
    "70b": Path("SparseLLM-ReluLLaMA-70B"),
}


def output_dir_for_preset(preset: str) -> Path:
    base = Path("Reports/results")
    if preset == "7b":
        return base / "phase4_neuron_variation"
    return base / f"phase4_neuron_variation_{preset}"


def geometry_from_model(model) -> ModelGeometry:
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if n_layers is None:
        raise ValueError("Could not read num_hidden_layers from model config")
    n_int = getattr(cfg, "intermediate_size", None)
    if n_int is None:
        raise ValueError("Could not read intermediate_size from model config")
    return ModelGeometry(num_layers=int(n_layers), n_neurons=int(n_int))

PROMPTS = {
    "creative": (
        "Once upon a time in a magical kingdom, there lived a young princess "
        "who dreamed of exploring the world beyond the castle walls."
    ),
    "code": (
        "Write a Python function that takes a list of integers and returns "
        "the longest increasing subsequence using dynamic programming."
    ),
    "factual": (
        "Explain the process of photosynthesis in plants, including the "
        "light-dependent and light-independent reactions."
    ),
    "reasoning": (
        "If all roses are flowers and some flowers fade quickly, can we "
        "conclude that some roses fade quickly? Explain your reasoning step by step."
    ),
    "conversational": (
        "Hey, how are you doing today? I was thinking about learning a new "
        "programming language. Any suggestions?"
    ),
}


# ── Activation capture ──────────────────────────────────────────────────────

class FFNActivationCapture:
    """Hook into LLaMA FFN intermediate outputs (after ReLU) to record activations."""

    def __init__(self, model, geom: ModelGeometry):
        self.model = model
        self.geom = geom
        self.hooks = []
        self.activations = defaultdict(list)  # layer_idx -> list of [n_neurons] bool tensors
        self._active = False

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            if not self._active:
                return
            # output shape: [batch, seq_len, intermediate_size]
            # A neuron is "activated" if its ReLU output > 0
            activated = (output > 0).detach().cpu()
            # Store per-token activation masks (flatten batch dim)
            for b in range(activated.shape[0]):
                for t in range(activated.shape[1]):
                    self.activations[layer_idx].append(activated[b, t])
        return hook_fn

    def register_hooks(self):
        for i, layer in enumerate(self.model.model.layers):
            # In LlamaForCausalLM with ReLU, the "gate" projection followed
            # by activation is in layer.mlp.up_proj (or the act_fn applied
            # to gate_proj). We hook the activation function output.
            # For ReluLLaMA: mlp uses gate_proj -> ReLU, then element-wise
            # multiply with up_proj. We want the post-ReLU gate output.
            hook = layer.mlp.act_fn.register_forward_hook(self._make_hook(i))
            self.hooks.append(hook)

    def start(self):
        self.activations.clear()
        self._active = True

    def stop(self):
        self._active = False

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_neuron_activation_counts(self):
        """Return per-layer activation counts: shape [num_layers, n_neurons]."""
        nl, nn = self.geom.num_layers, self.geom.n_neurons
        counts = np.zeros((nl, nn), dtype=np.int64)
        for layer_idx, masks in self.activations.items():
            for mask in masks:
                counts[layer_idx] += mask.numpy().astype(np.int64)
        return counts

    def get_hot_neuron_sets(self, threshold_frac=0.1):
        """Return per-layer sets of 'hot' neurons (activated in >= threshold_frac of tokens)."""
        counts = self.get_neuron_activation_counts()
        hot_sets = {}
        for layer_idx in range(self.geom.num_layers):
            n_tokens = len(self.activations.get(layer_idx, []))
            if n_tokens == 0:
                hot_sets[layer_idx] = set()
                continue
            threshold = threshold_frac * n_tokens
            hot_sets[layer_idx] = set(np.where(counts[layer_idx] >= threshold)[0])
        return hot_sets


# ── Overlap metrics ─────────────────────────────────────────────────────────

def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def overlap_coefficient(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ── Main pipeline ───────────────────────────────────────────────────────────

def run_prompts(
    model,
    tokenizer,
    capture,
    max_new_tokens,
    geom: ModelGeometry,
    *,
    hot_threshold_frac: float,
):
    """Run all prompts and return per-category hot neuron sets and raw counts."""
    dev = next(model.parameters()).device
    category_hot_sets = {}
    category_counts = {}
    category_n_tokens = {}

    for name, prompt in PROMPTS.items():
        print(f"\n  [{name}] Running prompt...")
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        capture.start()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        capture.stop()

        n_tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        n_total_tokens = outputs.shape[1]
        print(f"    Generated {n_tokens_generated} tokens ({n_total_tokens} total)")

        counts = capture.get_neuron_activation_counts()
        hot_sets = capture.get_hot_neuron_sets(threshold_frac=hot_threshold_frac)

        total_activated = sum(len(s) for s in hot_sets.values())
        print(
            f"    Avg hot neurons/layer: {total_activated / geom.num_layers:.0f} / {geom.n_neurons}"
        )

        category_hot_sets[name] = hot_sets
        category_counts[name] = counts
        category_n_tokens[name] = n_total_tokens

    return category_hot_sets, category_counts, category_n_tokens


def compute_overlap_matrices(category_hot_sets, geom: ModelGeometry):
    """Compute pairwise Jaccard and overlap matrices, per-layer and aggregated."""
    categories = list(category_hot_sets.keys())
    n = len(categories)
    nl = geom.num_layers

    # Aggregate: (layer_idx, neuron_id) pairs across the stack
    agg_sets = {}
    for cat in categories:
        agg_sets[cat] = set()
        for layer_idx in range(nl):
            for neuron in category_hot_sets[cat].get(layer_idx, set()):
                agg_sets[cat].add((layer_idx, neuron))

    jaccard_matrix = np.zeros((n, n))
    overlap_matrix = np.zeros((n, n))
    for i, ci in enumerate(categories):
        for j, cj in enumerate(categories):
            jaccard_matrix[i, j] = jaccard(agg_sets[ci], agg_sets[cj])
            overlap_matrix[i, j] = overlap_coefficient(agg_sets[ci], agg_sets[cj])

    # Per-layer Jaccard matrices
    per_layer_jaccard = np.zeros((nl, n, n))
    for layer in range(nl):
        for i, ci in enumerate(categories):
            for j, cj in enumerate(categories):
                si = category_hot_sets[ci].get(layer, set())
                sj = category_hot_sets[cj].get(layer, set())
                per_layer_jaccard[layer, i, j] = jaccard(si, sj)

    return categories, jaccard_matrix, overlap_matrix, per_layer_jaccard, agg_sets


def find_universal_and_specific(category_hot_sets, geom: ModelGeometry):
    """Identify universally hot and category-specific neurons per layer."""
    categories = list(category_hot_sets.keys())
    universal = {}
    specific = {cat: {} for cat in categories}
    nl = geom.num_layers

    for layer in range(nl):
        sets = [category_hot_sets[cat].get(layer, set()) for cat in categories]
        if sets:
            universal[layer] = set.intersection(*sets) if all(sets) else set()
        else:
            universal[layer] = set()

        for i, cat in enumerate(categories):
            others = [s for j, s in enumerate(sets) if j != i]
            others_union = set.union(*others) if others else set()
            specific[cat][layer] = sets[i] - others_union

    return universal, specific


# ── Output ──────────────────────────────────────────────────────────────────

def _sample_layer_indices(num_layers: int) -> list[int]:
    if num_layers <= 5:
        return list(range(num_layers))
    return sorted({0, num_layers // 4, num_layers // 2, (3 * num_layers) // 4, num_layers - 1})


def save_summary(
    categories,
    category_hot_sets,
    universal,
    specific,
    jaccard_matrix,
    overlap_matrix,
    category_n_tokens,
    geom: ModelGeometry,
    out_dir: Path,
    preset_label: str,
):
    """Print and save summary statistics."""
    nl, nn = geom.num_layers, geom.n_neurons
    print(f"\n{'='*100}")
    title = f"Phase 4: Hot Neuron Variation Summary ({preset_label.upper()})"
    print(f"{title:^100}")
    print(f"{'='*100}")

    print(f"\n--- Hot Neuron Counts per Category (averaged across layers) ---")
    header = f"{'Category':<16} {'Tokens':>7} {'Avg Hot':>8} {'% of Total':>10}"
    print(header)
    print("-" * 50)
    rows = []
    for cat in categories:
        n_hot = np.mean([len(category_hot_sets[cat].get(l, set())) for l in range(nl)])
        pct = 100 * n_hot / nn
        print(f"{cat:<16} {category_n_tokens[cat]:>7} {n_hot:>8.0f} {pct:>9.1f}%")
        rows.append({"category": cat, "tokens": category_n_tokens[cat],
                      "avg_hot": n_hot, "pct": pct})

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hot_neuron_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "tokens", "avg_hot", "pct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n--- Universal Neurons (hot in ALL categories) ---")
    for layer in _sample_layer_indices(nl):
        n_uni = len(universal[layer])
        print(f"  Layer {layer:>2}: {n_uni:>5} universal neurons ({100*n_uni/nn:.1f}%)")

    print(f"\n--- Category-Specific Neurons (hot in ONLY one category) ---")
    for cat in categories:
        avg_specific = np.mean([len(specific[cat].get(l, set())) for l in range(nl)])
        print(f"  {cat:<16}: {avg_specific:>6.0f} avg specific neurons/layer")

    print(f"\n--- Pairwise Jaccard Similarity (aggregated) ---")
    print(f"{'':>16}", end="")
    for cat in categories:
        print(f"{cat:>14}", end="")
    print()
    for i, ci in enumerate(categories):
        print(f"{ci:>16}", end="")
        for j in range(len(categories)):
            print(f"{jaccard_matrix[i,j]:>14.3f}", end="")
        print()

    # Save full results as JSON
    results = {
        "categories": categories,
        "jaccard_matrix": jaccard_matrix.tolist(),
        "overlap_matrix": overlap_matrix.tolist(),
        "universal_per_layer": {str(l): len(s) for l, s in universal.items()},
        "specific_per_layer": {
            cat: {str(l): len(s) for l, s in layers.items()}
            for cat, layers in specific.items()
        },
    }
    with open(out_dir / "overlap_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir}/overlap_results.json")


def plot1_jaccard_heatmap(categories, jaccard_matrix, out_dir: Path):
    """Heatmap of pairwise Jaccard similarity."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(jaccard_matrix, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(categories, fontsize=11)
    for i in range(len(categories)):
        for j in range(len(categories)):
            ax.text(j, i, f"{jaccard_matrix[i,j]:.3f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if jaccard_matrix[i,j] > 0.6 else "black")
    ax.set_title("Pairwise Jaccard Similarity of Hot Neuron Sets", fontsize=14)
    fig.colorbar(im, ax=ax, label="Jaccard Index", shrink=0.8)
    fig.tight_layout()
    path = out_dir / "plot1_jaccard_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot2_jaccard_by_layer(categories, per_layer_jaccard, geom: ModelGeometry, out_dir: Path):
    """Line plot of average pairwise Jaccard per layer."""
    nl = geom.num_layers
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, ci in enumerate(categories):
        for j, cj in enumerate(categories):
            if j <= i:
                continue
            jaccards = per_layer_jaccard[:, i, j]
            ax.plot(range(nl), jaccards, marker=".", markersize=4,
                    linewidth=1.2, alpha=0.7, label=f"{ci} vs {cj}")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Jaccard Similarity", fontsize=12)
    ax.set_title("Hot Neuron Overlap by Layer (pairwise)", fontsize=14)
    ax.set_xticks(range(0, nl, max(1, nl // 16)))
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = out_dir / "plot2_jaccard_by_layer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot3_universal_vs_specific(
    categories, universal, specific, geom: ModelGeometry, out_dir: Path
):
    """Stacked bar chart: universal, shared, and category-specific neurons per layer."""
    nl = geom.num_layers
    fig, ax = plt.subplots(figsize=(14, 6))

    uni_counts = [len(universal.get(l, set())) for l in range(nl)]
    spec_counts = {cat: [len(specific[cat].get(l, set())) for l in range(nl)]
                   for cat in categories}

    x = np.arange(nl)
    ax.bar(x, uni_counts, label="Universal (all categories)", color="#4CAF50", alpha=0.85)

    bottom = np.array(uni_counts, dtype=float)
    colors = ["#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]
    for idx, cat in enumerate(categories):
        ax.bar(x, spec_counts[cat], bottom=bottom,
               label=f"Specific to {cat}", color=colors[idx % len(colors)], alpha=0.7)
        bottom += np.array(spec_counts[cat], dtype=float)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Number of Neurons", fontsize=12)
    ax.set_title("Universal vs. Category-Specific Hot Neurons per Layer", fontsize=14)
    ax.set_xticks(range(0, nl, max(1, nl // 16)))
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "plot3_universal_vs_specific.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot4_activation_density(categories, category_counts, geom: ModelGeometry, out_dir: Path):
    """Per-layer activation density (fraction of neurons activated) by category."""
    nl, nn = geom.num_layers, geom.n_neurons
    fig, ax = plt.subplots(figsize=(12, 6))

    for cat in categories:
        counts = category_counts[cat]
        density = [(counts[l] > 0).sum() / nn for l in range(nl)]
        ax.plot(range(nl), density, marker="o", markersize=5,
                linewidth=2, label=cat)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Activation Density (fraction of neurons > 0)", fontsize=12)
    ax.set_title("Per-Layer Activation Density by Prompt Category", fontsize=14)
    ax.set_xticks(range(0, nl, max(1, nl // 16)))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = out_dir / "plot4_activation_density.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def _load_model(model_path: Path, device_arg: str | None):
    """Load CausalLM with appropriate device placement."""
    mp = str(model_path)
    kw: dict = {"torch_dtype": torch.float16, "trust_remote_code": True}

    if device_arg == "auto":
        try:
            import accelerate  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                "ERROR: --device auto needs `pip install accelerate` (HF uses it for device_map / offload).\n"
                "  For a single full GPU, omit --device or use --device cuda and ensure the model fits in VRAM.\n"
            ) from e
        kw["device_map"] = "auto"
        kw["low_cpu_mem_usage"] = True
        print("Loading with device_map='auto' (multi-GPU / CPU offload via accelerate).")
        return AutoModelForCausalLM.from_pretrained(mp, **kw)

    if device_arg is None:
        if torch.backends.mps.is_available():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    else:
        dev = device_arg

    if dev == "mps":
        m = AutoModelForCausalLM.from_pretrained(mp, **kw)
        return m.to("mps")
    if dev == "cpu":
        return AutoModelForCausalLM.from_pretrained(mp, **kw).to("cpu")

    # cuda or cuda:0: load to CPU first, then .to(device). Using device_map here
    # requires `accelerate` in recent transformers even for a single string device.
    m = AutoModelForCausalLM.from_pretrained(mp, **kw)
    return m.to(dev)


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Neuron variation across input types")
    parser.add_argument(
        "--preset",
        choices=["7b", "13b", "70b"],
        default="7b",
        help="SparseLLM ReluLLaMA checkpoint size (default: 7b). Sets default --model-path.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to SparseLLM ReluLLaMA folder (overrides --preset default).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Tokens to generate per prompt (default: 64)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Fraction of tokens a neuron must fire in to be 'hot' (default: 0.1)")
    parser.add_argument(
        "--device",
        default=None,
        help="Placement: cpu, mps, cuda, cuda:0, or auto (HF accelerate; recommended for 70B). "
        "If omitted: mps > cuda > cpu. For 70B that does not fit on one GPU, use --device auto.",
    )
    args = parser.parse_args()

    model_path = args.model_path if args.model_path is not None else PRESET_PATHS[args.preset]
    if not model_path.exists():
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = output_dir_for_preset(args.preset)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preset: {args.preset}  ->  {model_path}")
    print(f"Output directory: {out_dir}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Hot neuron threshold: {args.threshold} (fired in >= {100*args.threshold:.0f}% of tokens)")

    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
    model = _load_model(model_path, args.device)
    model.eval()
    geom = geometry_from_model(model)
    print(
        f"Model geometry: {geom.num_layers} layers, {geom.n_neurons} FFN neurons/layer "
        f"({sum(p.numel() for p in model.parameters())/1e9:.1f}B params)"
    )

    capture = FFNActivationCapture(model, geom)
    capture.register_hooks()

    print("\nRunning prompts...")
    category_hot_sets, category_counts, category_n_tokens = run_prompts(
        model,
        tokenizer,
        capture,
        args.max_new_tokens,
        geom,
        hot_threshold_frac=args.threshold,
    )
    capture.remove_hooks()

    print("\nComputing overlap metrics...")
    categories, jaccard_mat, overlap_mat, per_layer_jaccard, agg_sets = compute_overlap_matrices(
        category_hot_sets, geom
    )

    universal, specific = find_universal_and_specific(category_hot_sets, geom)

    save_summary(
        categories,
        category_hot_sets,
        universal,
        specific,
        jaccard_mat,
        overlap_mat,
        category_n_tokens,
        geom,
        out_dir,
        args.preset,
    )

    print("\nGenerating plots...")
    plot1_jaccard_heatmap(categories, jaccard_mat, out_dir)
    plot2_jaccard_by_layer(categories, per_layer_jaccard, geom, out_dir)
    plot3_universal_vs_specific(categories, universal, specific, geom, out_dir)
    plot4_activation_density(categories, category_counts, geom, out_dir)

    print(f"\nAll Phase 4 outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
