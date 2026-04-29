#!/usr/bin/env python3
from __future__ import annotations

"""
Phase 3: Inference Speed Benchmarking — PowerInfer vs. llama.cpp

Runs both engines across a set of prompts and output lengths,
parses their timing output, and generates comparison tables and plots.

Usage (from the PowerInfer repo root):
    # Quick sanity check — 1 prompt, 32 tokens (< 10 min on Apple CPU)
    python Reports/scripts/phase3_benchmark.py --mode local --quick

    # 13B / 70B (paths: ReluLLaMA-{13,70}B/*.gguf and ReluLLaMA-{13,70}B-dense/*.gguf)
    python Reports/scripts/phase3_benchmark.py --mode newton --preset 13b --quick
    python Reports/scripts/phase3_benchmark.py --mode newton --preset 70b --ngl 99

    # Full local benchmark — all prompts, multiple token counts (~2-3 hours on Apple CPU for 7B)
    python Reports/scripts/phase3_benchmark.py --mode local

    # Newton GPU cluster (full benchmark, fast)
    python Reports/scripts/phase3_benchmark.py --mode newton

Note: The FP16 dense model (~13 GB for 7B) runs at ~0.5 tok/s on Apple M4 Max CPU.
      PowerInfer's sparse model is significantly faster (~10 tok/s).
      For quick local tests, use --quick. Full benchmarks are best run on Newton.
      13B/70B on CPU are extremely slow; prefer --mode newton. On ~16GB GPUs, lower
      --ngl for PowerInfer main and llama.cpp on Newton, or use a quantized GGUF via --llamacpp-model.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("Reports/results/phase3_benchmarks")

# ReluLLaMA PowerInfer GGUF + dense F16 from convert-dense.py (repo-relative paths).
MODEL_PRESETS: dict[str, dict[str, Path]] = {
    "7b": {
        "powerinfer": Path("ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf"),
        "llamacpp": Path("ReluLLaMA-7B-dense/llama-7b-relu.gguf"),
    },
    "13b": {
        "powerinfer": Path("ReluLLaMA-13B/llama-13b-relu.powerinfer.gguf"),
        "llamacpp": Path("ReluLLaMA-13B-dense/llama-13b-relu.gguf"),
    },
    "70b": {
        "powerinfer": Path("ReluLLaMA-70B/llama-70b-relu.powerinfer.gguf"),
        "llamacpp": Path("ReluLLaMA-70B-dense/llama-70b-relu.gguf"),
    },
}

# Default per-run timeouts (seconds) when --timeout is not set
TIMEOUT_LOCAL_QUICK = {"7b": 600, "13b": 3600, "70b": 14_400}
TIMEOUT_LOCAL_FULL = {"7b": 600, "13b": 7200, "70b": 28_800}
# Newton: 13B/70B need headroom for GGUF mmap + first-token latency on busy or 16GB-class GPUs.
TIMEOUT_NEWTON = {"7b": 300, "13b": 3600, "70b": 10_800}

PROMPTS = {
    "creative": "Once upon a time in a magical kingdom, there lived a young princess who dreamed of exploring the world beyond the castle walls.",
    "code": "Write a Python function that takes a list of integers and returns the longest increasing subsequence using dynamic programming.",
    "factual": "Explain the process of photosynthesis in plants, including the light-dependent and light-independent reactions.",
    "reasoning": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning step by step.",
    "conversational": "Hey, how are you doing today? I was thinking about learning a new programming language. Any suggestions?",
}

# ── Engine configs ──────────────────────────────────────────────────────────

def _get_local_config(preset: str, llamacpp_model_override: Path | None):
    paths = MODEL_PRESETS[preset]
    lc_model = llamacpp_model_override or paths["llamacpp"]
    return {
        "powerinfer": {
            "bin": "./build/bin/main",
            "model": str(paths["powerinfer"]),
            "extra_args": [],
        },
        "llamacpp": {
            "bin": "../llama-cpp-upstream/build/bin/llama-completion",
            "model": str(lc_model),
            "extra_args": ["-ngl", "0"],
        },
    }


def _get_newton_config(preset: str, ngl: int, llamacpp_model_override: Path | None):
    home = os.path.expanduser("~")
    paths = MODEL_PRESETS[preset]
    lc_model = llamacpp_model_override or paths["llamacpp"]
    return {
        "powerinfer": {
            "bin": "./build/bin/main",
            "model": str(paths["powerinfer"]),
            # Match llama.cpp GPU offload: PowerInfer `main` also accepts `-ngl` (see examples/main/README.md).
            "extra_args": ["-ngl", str(ngl)],
        },
        "llamacpp": {
            "bin": f"{home}/PowerInfer/LLAMA/llama.cpp/build/bin/llama-completion",
            "model": str(lc_model),
            "extra_args": ["-ngl", str(ngl)],
        },
    }


# ── Runner ──────────────────────────────────────────────────────────────────

def run_inference(bin_path, model_path, prompt, n_tokens, threads, extra_args, timeout):
    """Run an inference binary and return the raw stderr+stdout output."""
    cmd = [
        bin_path,
        "-m", model_path,
        "-n", str(n_tokens),
        "-t", str(threads),
        "--ignore-eos",
        "-p", prompt,
        *extra_args,
    ]
    print(f"  CMD: {' '.join(cmd[:6])} ... -n {n_tokens}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return None
    except FileNotFoundError:
        print(f"  ERROR: binary not found: {bin_path}")
        return None


# ── Parsers ─────────────────────────────────────────────────────────────────

def parse_powerinfer_timings(output):
    """Parse PowerInfer's llama_print_timings output."""
    if output is None:
        return None

    timings = {}

    m = re.search(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second", output)
    if m:
        timings["prompt_eval_ms"] = float(m.group(1))
        timings["prompt_tokens"] = int(m.group(2))
        timings["prompt_tok_per_sec"] = float(m.group(3))

    m = re.search(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*ms per token,\s*([\d.]+)\s*tokens per second", output)
    if m:
        timings["eval_ms"] = float(m.group(1))
        timings["eval_tokens"] = int(m.group(2))
        timings["ms_per_token"] = float(m.group(3))
        timings["gen_tok_per_sec"] = float(m.group(4))

    m = re.search(r"total time\s*=\s*([\d.]+)\s*ms", output)
    if m:
        timings["total_ms"] = float(m.group(1))

    m = re.search(r"load time\s*=\s*([\d.]+)\s*ms", output)
    if m:
        timings["load_ms"] = float(m.group(1))

    return timings if timings else None


def parse_llamacpp_timings(output):
    """Parse llama.cpp (llama-cli) timing output.

    Newer llama.cpp uses llama_perf_context_print with the same format as
    PowerInfer's llama_print_timings, so we reuse the same parser.
    """
    if output is None:
        return None
    timings = parse_powerinfer_timings(output)
    if timings:
        return timings

    # Fallback: newer llama.cpp may use slightly different labels
    t = {}
    m = re.search(r"prompt\s+eval\s+time\s*=\s*([\d.]+)\s*ms.*?([\d.]+)\s*tokens per second", output, re.DOTALL)
    if m:
        t["prompt_eval_ms"] = float(m.group(1))
        t["prompt_tok_per_sec"] = float(m.group(2))

    m = re.search(r"eval\s+time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+).*?([\d.]+)\s*tokens per second", output)
    if m:
        t["eval_ms"] = float(m.group(1))
        t["eval_tokens"] = int(m.group(2))
        t["gen_tok_per_sec"] = float(m.group(3))

    m = re.search(r"total\s+time\s*=\s*([\d.]+)\s*ms", output)
    if m:
        t["total_ms"] = float(m.group(1))

    return t if t else None


# ── Benchmark orchestration ─────────────────────────────────────────────────

def run_benchmark(engines, n_tokens_list, threads, timeout, num_runs, log_subdir: str):
    """Run all benchmarks and return structured results."""
    results = []
    log_root = OUTPUT_DIR / "logs" / log_subdir
    log_root.mkdir(parents=True, exist_ok=True)

    for prompt_name, prompt_text in PROMPTS.items():
        for n_tokens in n_tokens_list:
            for engine_name, engine_cfg in engines.items():
                for run_idx in range(num_runs):
                    print(f"\n[{engine_name}] prompt={prompt_name}, n={n_tokens}, run={run_idx+1}/{num_runs}")

                    output = run_inference(
                        bin_path=engine_cfg["bin"],
                        model_path=engine_cfg["model"],
                        prompt=prompt_text,
                        n_tokens=n_tokens,
                        threads=threads,
                        extra_args=engine_cfg["extra_args"],
                        timeout=timeout,
                    )

                    if engine_name == "powerinfer":
                        timings = parse_powerinfer_timings(output)
                    else:
                        timings = parse_llamacpp_timings(output)

                    if timings:
                        print(f"  -> gen: {timings.get('gen_tok_per_sec', '?')} tok/s, "
                              f"total: {timings.get('total_ms', '?')} ms")
                    else:
                        print("  -> FAILED to parse timings")

                    log_name = f"{engine_name}_{prompt_name}_n{n_tokens}_run{run_idx}.log"
                    if output:
                        (log_root / log_name).write_text(output)

                    results.append({
                        "engine": engine_name,
                        "prompt": prompt_name,
                        "n_tokens": n_tokens,
                        "run": run_idx,
                        "timings": timings,
                    })

    return results


# ── Output ──────────────────────────────────────────────────────────────────

def _results_suffix(preset: str) -> str:
    return "" if preset == "7b" else f"_{preset}"


def save_results_csv(results, preset: str):
    """Save all benchmark results to CSV."""
    path = OUTPUT_DIR / f"benchmark_results{_results_suffix(preset)}.csv"
    fields = [
        "engine", "prompt", "n_tokens", "run",
        "load_ms", "prompt_eval_ms", "prompt_tokens", "prompt_tok_per_sec",
        "eval_ms", "eval_tokens", "ms_per_token", "gen_tok_per_sec", "total_ms",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {"engine": r["engine"], "prompt": r["prompt"],
                   "n_tokens": r["n_tokens"], "run": r["run"]}
            if r["timings"]:
                row.update(r["timings"])
            writer.writerow(row)

    print(f"\nSaved: {path}")


def save_results_json(results, preset: str):
    """Save all benchmark results to JSON for later analysis."""
    path = OUTPUT_DIR / f"benchmark_results{_results_suffix(preset)}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {path}")


def print_summary(results, preset: str):
    """Print a summary comparison table."""
    label = f"Benchmark Summary — {preset.upper()}"
    print(f"\n{'='*90}")
    print(f"{label:^90}")
    print(f"{'='*90}")

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        if r["timings"] and "gen_tok_per_sec" in r["timings"]:
            key = (r["engine"], r["prompt"], r["n_tokens"])
            grouped[key].append(r["timings"]["gen_tok_per_sec"])

    fmt = "{:<12} {:<16} {:>8} {:>12} {:>12} {:>12}"
    print(fmt.format("Engine", "Prompt", "N_Tok", "Gen tok/s", "Std", "Speedup"))
    print("-" * 90)

    pi_speeds = {}
    lc_speeds = {}

    for (engine, prompt, n_tok), speeds in sorted(grouped.items()):
        avg = np.mean(speeds)
        std = np.std(speeds) if len(speeds) > 1 else 0
        if engine == "powerinfer":
            pi_speeds[(prompt, n_tok)] = avg
        else:
            lc_speeds[(prompt, n_tok)] = avg
        print(fmt.format(engine, prompt, n_tok, f"{avg:.2f}", f"{std:.2f}", ""))

    print("-" * 90)
    print("\nSpeedup (PowerInfer / llama.cpp):")
    for key in sorted(set(pi_speeds.keys()) & set(lc_speeds.keys())):
        prompt, n_tok = key
        speedup = pi_speeds[key] / lc_speeds[key] if lc_speeds[key] > 0 else float("inf")
        print(f"  {prompt:<16} n={n_tok:<4}  -> {speedup:.2f}x")


def plot_comparison(results, preset: str):
    """Generate comparison bar charts."""
    suf = _results_suffix(preset)
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["timings"] and "gen_tok_per_sec" in r["timings"]:
            key = (r["prompt"], r["n_tokens"])
            grouped[key][r["engine"]].append(r["timings"]["gen_tok_per_sec"])

    if not grouped:
        print("No valid results to plot.")
        return

    # Plot 1: Generation speed by prompt category
    prompts_in_data = sorted(set(k[0] for k in grouped.keys()))
    n_tokens_in_data = sorted(set(k[1] for k in grouped.keys()))

    for n_tok in n_tokens_in_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(prompts_in_data))
        width = 0.35

        pi_vals = []
        lc_vals = []
        for p in prompts_in_data:
            pi_speeds = grouped.get((p, n_tok), {}).get("powerinfer", [])
            lc_speeds = grouped.get((p, n_tok), {}).get("llamacpp", [])
            pi_vals.append(np.mean(pi_speeds) if pi_speeds else 0)
            lc_vals.append(np.mean(lc_speeds) if lc_speeds else 0)

        bars1 = ax.bar(x - width / 2, pi_vals, width, label="PowerInfer", color="#2196F3", alpha=0.85)
        bars2 = ax.bar(x + width / 2, lc_vals, width, label="llama.cpp", color="#FF9800", alpha=0.85)

        for bar, val in zip(bars1, pi_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        for bar, val in zip(bars2, lc_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Prompt Category", fontsize=12)
        ax.set_ylabel("Generation Speed (tokens/sec)", fontsize=12)
        ax.set_title(
            f"PowerInfer vs. llama.cpp ({preset.upper()}) — Generation Speed (n={n_tok} tokens)",
            fontsize=14,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(prompts_in_data, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        path = OUTPUT_DIR / f"plot_speed_comparison_n{n_tok}{suf}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")

    # Plot 2: Speedup chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for n_tok in n_tokens_in_data:
        speedups = []
        labels = []
        for p in prompts_in_data:
            pi_s = grouped.get((p, n_tok), {}).get("powerinfer", [])
            lc_s = grouped.get((p, n_tok), {}).get("llamacpp", [])
            if pi_s and lc_s:
                speedups.append(np.mean(pi_s) / np.mean(lc_s))
                labels.append(p)

        if speedups:
            ax.bar(np.arange(len(labels)) + 0.15 * n_tokens_in_data.index(n_tok),
                   speedups, 0.15, label=f"n={n_tok}", alpha=0.85)

    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, label="Parity (1x)")
    ax.set_xlabel("Prompt Category", fontsize=12)
    ax.set_ylabel("Speedup (PowerInfer / llama.cpp)", fontsize=12)
    ax.set_title(f"PowerInfer Speedup over llama.cpp ({preset.upper()}) by Prompt Type", fontsize=14)
    ax.set_xticks(np.arange(len(prompts_in_data)))
    ax.set_xticklabels(prompts_in_data, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    path = OUTPUT_DIR / f"plot_speedup{suf}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Benchmark PowerInfer vs. llama.cpp")
    parser.add_argument("--mode", choices=["local", "newton"], default="local",
                        help="'local' for Apple M4 Max CPU-only, 'newton' for GPU cluster")
    parser.add_argument(
        "--preset",
        choices=["7b", "13b", "70b"],
        default="7b",
        help="Model size: ReluLLaMA-{7,13,70}B PowerInfer GGUF + matching dense GGUF (default: 7b)",
    )
    parser.add_argument(
        "--ngl",
        type=int,
        default=99,
        help="Newton only: GPU layers (-ngl) for both PowerInfer main and llama.cpp. "
        "Use a lower value on ~16GB VRAM with F16 13B/70B dense.",
    )
    parser.add_argument(
        "--llamacpp-model",
        type=Path,
        default=None,
        help="Override path to llama.cpp GGUF (e.g. quantized file); default uses MODEL_PRESETS dense F16",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check: 1 prompt, 32 tokens only")
    parser.add_argument("--threads", type=int, default=8, help="Number of CPU threads")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per config (for averaging)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=(
            "Timeout per run in seconds (auto if not set). Newton defaults: 7b=300, 13b=3600, 70b=10800 — "
            "raise further if runs still hit TIMEOUT (large models / slow I/O)."
        ),
    )
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Subset of prompt names to run (e.g., creative factual)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preset = args.preset
    lc_override = args.llamacpp_model

    if args.mode == "local":
        engines = _get_local_config(preset, lc_override)
        if args.timeout is not None:
            timeout = args.timeout
        elif args.quick:
            timeout = TIMEOUT_LOCAL_QUICK[preset]
        else:
            timeout = TIMEOUT_LOCAL_FULL[preset]
        if args.quick:
            n_tokens_list = [32]
            print("Mode: LOCAL QUICK (CPU-only, 1 prompt x 32 tokens)")
            print(f"Preset: {preset} — timeout {timeout}s/run")
            if preset == "7b":
                print("Estimated time: ~8-10 min (PowerInfer ~20s + llama.cpp ~60-90s per run)\n")
            else:
                print("WARNING: 13B/70B on CPU are very slow; prefer --mode newton for GPU.\n")
        else:
            n_tokens_list = [32, 64, 128]
            print("Mode: LOCAL FULL (CPU-only)")
            print(f"Preset: {preset} — token counts {n_tokens_list}, timeout {timeout}s/run")
            print("WARNING: FP16 dense llama.cpp is slow on CPU.")
            if preset != "7b":
                print("WARNING: 13B/70B full local runs can take many hours. Use --quick or --mode newton.\n")
            else:
                print("         Full 7B benchmark may take 2-3 hours. Use --quick for a fast check.\n")
    else:
        engines = _get_newton_config(preset, args.ngl, lc_override)
        timeout = args.timeout if args.timeout is not None else TIMEOUT_NEWTON[preset]
        n_tokens_list = [32, 64, 128]
        print("Mode: NEWTON (GPU cluster)")
        print(f"Preset: {preset} — PowerInfer + llama.cpp -ngl {args.ngl} — timeout {timeout}s/run")
        print(f"Token counts: {n_tokens_list}")

    # Filter prompts if specified
    global PROMPTS
    all_prompt_keys = list(PROMPTS.keys())
    if args.quick:
        PROMPTS = {"creative": PROMPTS["creative"]}
    elif args.prompts:
        PROMPTS = {k: v for k, v in PROMPTS.items() if k in args.prompts}
        if not PROMPTS:
            print(f"ERROR: no matching prompts. Choose from: {all_prompt_keys}")
            sys.exit(1)

    print(f"Prompts: {list(PROMPTS.keys())}")
    print(f"Timeout: {timeout}s per run\n")

    # Verify binaries and models exist
    for name, cfg in engines.items():
        if not Path(cfg["bin"]).exists():
            print(f"ERROR: {name} binary not found at {cfg['bin']}")
            sys.exit(1)
        if not Path(cfg["model"]).exists():
            print(f"ERROR: {name} model not found at {cfg['model']}")
            sys.exit(1)
        print(f"  {name}: {cfg['bin']} -> {cfg['model']}")

    log_subdir = f"preset_{preset}"
    t_start = time.time()
    results = run_benchmark(engines, n_tokens_list, args.threads, timeout, args.runs, log_subdir)
    elapsed = time.time() - t_start
    print(f"\nTotal benchmark time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    save_results_csv(results, preset)
    save_results_json(results, preset)
    print_summary(results, preset)

    print("\nGenerating plots...")
    plot_comparison(results, preset)

    print(f"\nAll Phase 3 outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
