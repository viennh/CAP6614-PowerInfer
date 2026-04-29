#!/usr/bin/env python3
"""
Generate 3-way comparison plots for Phase 3 benchmarks for 13b model.

Usage:
  python Reports/scripts/plot_3way_13b.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_three_way_comparison(csv_path: Path, output_dir: Path):
    """Generate comparison plots including vLLM alongside PowerInfer and llama.cpp."""
    if not csv_path.exists():
        print(f"No combined CSV found at {csv_path}, skipping 3-way plot.")
        return

    all_data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("gen_tok_per_sec"):
                try:
                    speed = float(row["gen_tok_per_sec"])
                    key = (row["prompt"], int(row["n_tokens"]))
                    all_data[key][row["engine"]].append(speed)
                except (ValueError, KeyError):
                    continue

    if not all_data:
        print("No valid data for 3-way plot.")
        return

    prompts_in_data = sorted(set(k[0] for k in all_data.keys()))
    n_tokens_in_data = sorted(set(k[1] for k in all_data.keys()))
    engines = ["powerinfer", "llamacpp", "vllm"]
    engine_labels = {"powerinfer": "PowerInfer", "llamacpp": "llama.cpp", "vllm": "vLLM"}
    engine_colors = {"powerinfer": "#2196F3", "llamacpp": "#FF9800", "vllm": "#4CAF50"}

    for n_tok in n_tokens_in_data:
        if n_tok not in [32, 64, 128]:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(prompts_in_data))
        n_engines = len(engines)
        width = 0.25

        for idx, engine in enumerate(engines):
            vals = []
            for p in prompts_in_data:
                speeds = all_data.get((p, n_tok), {}).get(engine, [])
                vals.append(np.mean(speeds) if speeds else 0)

            bars = ax.bar(
                x + (idx - n_engines / 2 + 0.5) * width,
                vals, width,
                label=engine_labels.get(engine, engine),
                color=engine_colors.get(engine, "#999"),
                alpha=0.85,
            )
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Prompt Category", fontsize=12)
        ax.set_ylabel("Generation Speed (tokens/sec)", fontsize=12)
        ax.set_title(f"PowerInfer vs. llama.cpp vs. vLLM — Generation Speed (13B, n={n_tok})", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(prompts_in_data, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        path = output_dir / f"plot_3way_comparison_13b_n{n_tok}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def _powerinfer_llamacpp_rows_from_json(json_path: Path) -> list[dict]:
    """Flatten benchmark_results_13b.json to the same dict shape as benchmark_results_13b.csv."""
    with open(json_path) as f:
        data = json.load(f)
    rows = []
    for e in data:
        t = e["timings"]
        rows.append(
            {
                "engine": e["engine"],
                "prompt": e["prompt"],
                "n_tokens": str(e["n_tokens"]),
                "run": str(e["run"]),
                "load_ms": str(t["load_ms"]),
                "prompt_eval_ms": str(t["prompt_eval_ms"]),
                "prompt_tokens": str(t["prompt_tokens"]),
                "prompt_tok_per_sec": str(t["prompt_tok_per_sec"]),
                "eval_ms": str(t["eval_ms"]),
                "eval_tokens": str(t["eval_tokens"]),
                "ms_per_token": str(t["ms_per_token"]),
                "gen_tok_per_sec": str(t["gen_tok_per_sec"]),
                "total_ms": str(t["total_ms"]),
            }
        )
    return rows


def write_combined_csv(output_dir: Path) -> Path:
    """Build combined CSV: PowerInfer/llama.cpp from JSON (if present) else CSV, then vLLM."""
    json_path = output_dir / "benchmark_results_13b.json"
    csv_13b = output_dir / "benchmark_results_13b.csv"
    vllm_csv = output_dir / "vllm_benchmark_results_13b.csv"
    out = output_dir / "benchmark_results_13b_combined.csv"

    rows_13b: list[dict] = []
    if json_path.is_file():
        rows_13b = _powerinfer_llamacpp_rows_from_json(json_path)
        if csv_13b.is_file():
            with open(csv_13b, newline="") as f:
                from_csv = list(csv.DictReader(f))
            if from_csv != rows_13b:
                print(
                    "Warning: benchmark_results_13b.json and .csv differ; "
                    "using JSON for combined file."
                )
    elif csv_13b.is_file():
        with open(csv_13b, newline="") as f:
            rows_13b = list(csv.DictReader(f))

    if not vllm_csv.is_file() or not rows_13b:
        return out

    with open(vllm_csv, newline="") as f:
        vrows = list(csv.DictReader(f))
    rows = rows_13b + vrows
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} data rows)")
    return out


def main():
    output_dir = Path("Reports/results/phase3_benchmarks")
    csv_path = write_combined_csv(output_dir)
    plot_three_way_comparison(csv_path, output_dir)


if __name__ == "__main__":
    main()