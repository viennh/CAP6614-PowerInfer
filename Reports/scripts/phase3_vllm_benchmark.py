#!/usr/bin/env python3
"""
Phase 3 (vLLM): Inference Speed Benchmarking with vLLM

Benchmarks vLLM on the same prompts and token counts as phase3_benchmark.py,
then merges results into the existing benchmark CSV and generates updated plots.

Usage (from the PowerInfer repo root):
    # On Newton GPU (requires NVIDIA GPU):
    python Reports/scripts/phase3_vllm_benchmark.py

    # Llama-2 13B / 70B (official HF checkpoints; same SiLU baseline as 7B):
    python Reports/scripts/phase3_vllm_benchmark.py --preset 13b
    python Reports/scripts/phase3_vllm_benchmark.py --preset 70b --tensor-parallel-size 2

    # Quick test with one prompt:
    python Reports/scripts/phase3_vllm_benchmark.py --quick

    # Custom model path (must be standard SiLU LLaMA — NOT SparseLLM/ReluLLaMA):
    python Reports/scripts/phase3_vllm_benchmark.py --model meta-llama/Llama-2-7b-hf

Note: vLLM requires an NVIDIA GPU. This script will NOT work on Apple Silicon.
      vLLM's Llama implementation only supports SiLU. SparseLLM/ReluLLaMA (hidden_act=relu)
      will fail with ValueError: Unsupported activation: relu. For vLLM baselines, use
      meta-llama/Llama-2-{7,13,70}b-hf via --preset or --model — never point --model at
      a ReluLLaMA folder.
      This is still a fair comparison: same architecture and parameter count,
      measuring vLLM's serving throughput vs PowerInfer's sparse inference.

      Do not call torch.cuda in this process before vLLM's LLM() — vLLM v1 may fork
      workers, and CUDA cannot be re-initialized in a forked child (H100 included).
      VRAM preflight uses nvidia-smi instead.

      UCF Newton + H100: use the Slurm script in Reports/scripts/ucf_newton/ (GPU env
      sanitization, VLLM_TARGET_DEVICE, optional --constraint=h100). This entrypoint
      sets multiprocessing start method to "spawn" so vLLM EngineCore workers do not
      inherit a fork-broken CUDA context (CUDA unknown error in get_device_capability).
      By default vLLM runs with enforce_eager=True (no CUDA graphs) to avoid Newton/v1
      startup failures (RuntimeError: cancelled after graph capture). Use --allow-cuda-graphs to opt in.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("Reports/results/phase3_benchmarks")

# Official Llama-2 chat/base HF IDs (SiLU) — same family as the default 7B baseline.
# ReluLLaMA (ReLU) is not supported by vLLM; use these for apples-to-architecture serving throughput.
PRESET_MODELS = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "70b": "meta-llama/Llama-2-70b-hf",
}

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


def ensure_vllm_compatible_llama(model_ref: str) -> None:
    """
    Preflight: vLLM's Llama backend only supports SiLU. ReluLLaMA (ReLU) aborts at load.

    Uses transformers to read config when available so we fail fast with a clear message.
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        return

    try:
        cfg = AutoConfig.from_pretrained(model_ref, trust_remote_code=True)
    except Exception as exc:
        print(
            f"Note: could not load config for preflight check ({exc!r}); "
            "proceeding — vLLM may still error if the model is incompatible.",
            file=sys.stderr,
        )
        return

    act = getattr(cfg, "hidden_act", None)
    if act is None:
        return
    act_s = str(act).lower()
    # vLLM llama.py: "Only silu is supported for now" — ReLU-family activations error out.
    if act_s == "relu" or act_s.startswith("relu_") or act_s in (
        "relu2",
        "relu_squared",
        "leaky_relu",
    ):
        sys.stderr.write(
            "\nERROR: This checkpoint uses hidden_act=%r. vLLM's Llama code only supports SiLU.\n"
            "SparseLLM/ReluLLaMA models cannot run in vLLM.\n\n"
            "Use the standard Llama-2 HF baseline (same size, SiLU), e.g. omit --model:\n"
            "  python Reports/scripts/phase3_vllm_benchmark.py --preset 13b\n"
            "or set explicitly:\n"
            "  python Reports/scripts/phase3_vllm_benchmark.py --preset 13b "
            "--model meta-llama/Llama-2-13b-hf\n\n"
            % (act,)
        )
        raise SystemExit(2)


def query_gpu0_total_memory_gib() -> float | None:
    """
    GPU 0 total VRAM from nvidia-smi (MiB → GiB). Does not touch torch.cuda — safe
    before vLLM forks engine workers.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        line = out.strip().splitlines()[0].strip()
        mib = float(line)
        return mib / 1024.0
    except Exception:
        return None


def query_visible_gpu_count() -> int | None:
    """
    Number of GPUs visible to this job (nvidia-smi -L). Does not use torch.cuda.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        return len(lines) if lines else None
    except Exception:
        return None


def require_tensor_parallel_matches_visible_gpus(tensor_parallel_size: int) -> None:
    if tensor_parallel_size <= 1:
        return
    n = query_visible_gpu_count()
    if n is None:
        return
    if tensor_parallel_size > n:
        sys.stderr.write(
            "\nERROR: --tensor-parallel-size %d exceeds visible GPUs (%d from nvidia-smi -L).\n"
            "Use --tensor-parallel-size 1 for a single-GPU Slurm job, or request matching GPUs, e.g.\n"
            "  #SBATCH --gres=gpu:%d\n"
            "  sbatch --export=ALL,TENSOR_PARALLEL=%d,... (same N)\n\n"
            % (tensor_parallel_size, n, tensor_parallel_size, tensor_parallel_size)
        )
        raise SystemExit(2)


def sanitize_gpu_env_for_vllm() -> None:
    """
    Empty CUDA_VISIBLE_DEVICES (common from bad shell exports) makes PyTorch see
    zero GPUs and vLLM can end up with an empty device_type → torch.device('').
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and not str(cvd).strip():
        del os.environ["CUDA_VISIBLE_DEVICES"]
    for key in ("VLLM_TARGET_DEVICE", "VLLM_DEVICE", "NVIDIA_VISIBLE_DEVICES"):
        val = os.environ.get(key)
        if val is not None and not str(val).strip():
            del os.environ[key]


def run_vllm_benchmark(
    model_name,
    n_tokens_list,
    prompts,
    num_runs,
    *,
    output_tag: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 512,
    max_num_seqs: int = 1,
    enforce_eager: bool = True,
):
    """Run vLLM benchmarks and return structured results."""
    sanitize_gpu_env_for_vllm()
    require_tensor_parallel_matches_visible_gpus(tensor_parallel_size)
    # Slurm: VLLM_TARGET_DEVICE + non-empty CVD (see sanitize). Do not instantiate
    # vllm.platforms cuda classes here — cls() can touch NVML/torch in the parent and
    # break forked EngineCore children (RuntimeError: CUDA unknown / get_device_capability).
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

    from vllm import LLM, SamplingParams

    print(f"Loading vLLM model: {model_name}")
    if tensor_parallel_size > 1:
        print(f"  tensor_parallel_size={tensor_parallel_size}")
    if enforce_eager:
        print("  enforce_eager=True (default: CUDA graphs off — avoids v1 compile_or_warm_up_model / RuntimeError: cancelled)")
    else:
        print("  enforce_eager=False (--allow-cuda-graphs: max throughput, may fail on some Newton/v1+TP setups)")
    t0 = time.time()
    # trust_remote_code only applies to HuggingFace Auto* loaders; vLLM ignores it here
    # and logs a warning. max_num_seqs=1 minimizes KV/cache reservation for this
    # single-sequence benchmark and helps avoid OOM on ~16GB GPUs with Llama-2-7B fp16.
    llm_kwargs = dict(
        model=model_name,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
    )
    llm_kwargs["enforce_eager"] = enforce_eager
    llm = LLM(**llm_kwargs)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    results = []

    for prompt_name, prompt_text in prompts.items():
        for n_tokens in n_tokens_list:
            for run_idx in range(num_runs):
                print(f"[vllm] prompt={prompt_name}, n={n_tokens}, run={run_idx+1}/{num_runs}")

                sampling_params = SamplingParams(
                    max_tokens=n_tokens,
                    temperature=0.0,
                )

                t_start = time.time()
                outputs = llm.generate([prompt_text], sampling_params)
                t_end = time.time()

                output = outputs[0]
                n_prompt_tokens = len(output.prompt_token_ids)
                n_generated = len(output.outputs[0].token_ids)
                total_ms = (t_end - t_start) * 1000

                # vLLM metrics (if available)
                metrics = getattr(output, "metrics", None)
                if metrics:
                    ttft_ms = getattr(metrics, "first_token_time", None)
                    if ttft_ms and metrics.arrival_time:
                        ttft_ms = (ttft_ms - metrics.arrival_time) * 1000
                    else:
                        ttft_ms = None
                    finish_time = getattr(metrics, "finished_time", None)
                    if finish_time and metrics.arrival_time:
                        e2e_ms = (finish_time - metrics.arrival_time) * 1000
                    else:
                        e2e_ms = total_ms
                else:
                    ttft_ms = None
                    e2e_ms = total_ms

                gen_tok_per_sec = (n_generated / (e2e_ms / 1000)) if e2e_ms > 0 else 0

                print(f"  -> {n_generated} tokens in {e2e_ms:.0f}ms "
                      f"({gen_tok_per_sec:.2f} tok/s)")

                timings = {
                    "load_ms": load_time * 1000,
                    "prompt_tokens": n_prompt_tokens,
                    "eval_tokens": n_generated,
                    "gen_tok_per_sec": gen_tok_per_sec,
                    "total_ms": e2e_ms,
                }
                if ttft_ms is not None:
                    timings["prompt_eval_ms"] = ttft_ms
                    timings["prompt_tok_per_sec"] = (
                        n_prompt_tokens / (ttft_ms / 1000) if ttft_ms > 0 else 0
                    )

                # Save raw output
                log_dir = OUTPUT_DIR / "logs" / f"vllm_{output_tag}"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_name = f"vllm_{prompt_name}_n{n_tokens}_run{run_idx}.log"
                (log_dir / log_name).write_text(
                    f"Prompt: {prompt_name}\n"
                    f"N tokens requested: {n_tokens}\n"
                    f"N tokens generated: {n_generated}\n"
                    f"Prompt tokens: {n_prompt_tokens}\n"
                    f"Total time: {e2e_ms:.2f} ms\n"
                    f"Gen tok/s: {gen_tok_per_sec:.2f}\n"
                    f"TTFT: {ttft_ms}\n\n"
                    f"--- Output ---\n{output.outputs[0].text}\n"
                )

                results.append({
                    "engine": "vllm",
                    "prompt": prompt_name,
                    "n_tokens": n_tokens,
                    "run": run_idx,
                    "timings": timings,
                })

    return results


def _results_suffix(output_tag: str) -> str:
    """Filename suffix for preset-specific outputs (7b keeps legacy unprefixed names)."""
    if output_tag == "7b":
        return ""
    return f"_{output_tag}"


def save_vllm_csv(results, output_tag: str):
    """Save vLLM results to a separate CSV and append to the combined CSV."""
    suf = _results_suffix(output_tag)
    fields = [
        "engine", "prompt", "n_tokens", "run",
        "load_ms", "prompt_eval_ms", "prompt_tokens", "prompt_tok_per_sec",
        "eval_ms", "eval_tokens", "ms_per_token", "gen_tok_per_sec", "total_ms",
    ]

    # Save vLLM-only CSV
    vllm_path = OUTPUT_DIR / f"vllm_benchmark_results{suf}.csv"
    with open(vllm_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {"engine": r["engine"], "prompt": r["prompt"],
                   "n_tokens": r["n_tokens"], "run": r["run"]}
            if r["timings"]:
                row.update(r["timings"])
            writer.writerow(row)
    print(f"Saved: {vllm_path}")

    # Merge into combined benchmark_results.csv only for 7B so 13B/70B runs do not wipe vLLM rows.
    combined_path = OUTPUT_DIR / "benchmark_results.csv"
    if output_tag != "7b":
        print(
            f"Note: Skipping merge into {combined_path} (preset/output_tag={output_tag!r}; "
            "only 7B vLLM runs update the combined CSV)."
        )
    elif combined_path.exists():
        # Read existing, filter out old vllm rows, append new
        existing = []
        with open(combined_path, "r") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames
            for row in reader:
                if row.get("engine") != "vllm":
                    existing.append(row)

        use_fields = existing_fields if existing_fields else fields
        with open(combined_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=use_fields)
            writer.writeheader()
            for row in existing:
                writer.writerow(row)
            for r in results:
                row = {"engine": r["engine"], "prompt": r["prompt"],
                       "n_tokens": r["n_tokens"], "run": r["run"]}
                if r["timings"]:
                    row.update(r["timings"])
                writer.writerow(row)
        print(f"Merged into: {combined_path}")

    # Save JSON
    json_path = OUTPUT_DIR / f"vllm_benchmark_results{suf}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {json_path}")


def print_summary(results, label: str = ""):
    """Print a summary table."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    title = f"vLLM Benchmark Summary ({label})" if label else "vLLM Benchmark Summary"
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")

    grouped = defaultdict(list)
    for r in results:
        if r["timings"] and "gen_tok_per_sec" in r["timings"]:
            key = (r["prompt"], r["n_tokens"])
            grouped[key].append(r["timings"]["gen_tok_per_sec"])

    fmt = "{:<16} {:>8} {:>12} {:>12}"
    print(fmt.format("Prompt", "N_Tok", "Gen tok/s", "Total ms"))
    print("-" * 70)
    for (prompt, n_tok), speeds in sorted(grouped.items()):
        avg_speed = np.mean(speeds)
        avg_ms = np.mean([
            r["timings"]["total_ms"] for r in results
            if r["prompt"] == prompt and r["n_tokens"] == n_tok
            and r["timings"] and "total_ms" in r["timings"]
        ])
        print(fmt.format(prompt, n_tok, f"{avg_speed:.2f}", f"{avg_ms:.0f}"))


def plot_three_way_comparison(results):
    """Generate comparison plots including vLLM alongside PowerInfer and llama.cpp."""
    import matplotlib.pyplot as plt
    from collections import defaultdict

    combined_path = OUTPUT_DIR / "benchmark_results.csv"
    if not combined_path.exists():
        print("No combined benchmark_results.csv found, skipping 3-way plot.")
        return

    all_data = defaultdict(lambda: defaultdict(list))
    with open(combined_path, "r") as f:
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
        ax.set_title(f"PowerInfer vs. llama.cpp vs. vLLM — Generation Speed (n={n_tok})", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(prompts_in_data, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        path = OUTPUT_DIR / f"plot_3way_comparison_n{n_tok}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: vLLM Benchmark")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_MODELS.keys()),
        default=None,
        help="Use standard Llama-2 HF model: 7b/13b/70b (writes vllm_benchmark_results_<preset>.* except 7b)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "HF id or local path for standard SiLU LLaMA (e.g. meta-llama/Llama-2-13b-hf). "
            "Do NOT use SparseLLM/ReluLLaMA — vLLM does not support ReLU (preflight exit)."
        ),
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 prompt, 32 tokens only")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per config for averaging")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size. 70B often needs 2–8 GPUs (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help=(
            "vLLM gpu_memory_utilization (default: 0.90 for 7B, 0.85 for 13B; "
            "lower further if you see CUDA OOM during model load)"
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="vLLM max_model_len (default: 512; lower if KV cache OOM)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="vLLM max_num_seqs (default: 1 for this single-request benchmark; values >1 use more VRAM)",
    )
    parser.add_argument(
        "--allow-cuda-graphs",
        action="store_true",
        help=(
            "Set enforce_eager=False so vLLM can use CUDA graphs (often faster). Default is eager mode "
            "for stable startup on Newton / vLLM v1 (avoids RuntimeError: cancelled after graph capture)."
        ),
    )
    args = parser.parse_args()

    sanitize_gpu_env_for_vllm()
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.preset:
        output_tag = args.preset
        model = args.model or PRESET_MODELS[args.preset]
    else:
        model = args.model or PRESET_MODELS["7b"]
        output_tag = "7b" if model == PRESET_MODELS["7b"] else "custom"

    gpu_mem = args.gpu_memory_utilization
    if gpu_mem is None:
        # Defaults tuned for 16GB-class GPUs: full fp16 7B is tight; 13B fp16 needs
        # multi-GPU TP — these values only help marginally for 13B.
        gpu_mem = 0.85 if output_tag == "13b" else 0.90

    ensure_vllm_compatible_llama(model)

    total_gib = query_gpu0_total_memory_gib()
    if (
        total_gib is not None
        and output_tag == "13b"
        and args.tensor_parallel_size == 1
        and total_gib < 22
    ):
        print(
            f"\nWARNING: GPU 0 reports ~{total_gib:.1f} GiB (nvidia-smi) — Llama-2-13B fp16 "
            "needs roughly 24+ GiB weights alone. Expect CUDA OOM unless you use "
            "--tensor-parallel-size > 1 or --preset 7b.\n",
            file=sys.stderr,
        )

    prompts = PROMPTS
    if args.quick:
        prompts = {"creative": PROMPTS["creative"]}
        n_tokens_list = [32]
        print("Mode: QUICK (1 prompt, 32 tokens)\n")
    else:
        n_tokens_list = [32, 64, 128]
        print("Mode: FULL (all prompts, n=32/64/128)\n")

    print(f"Model: {model}")
    print(f"Output tag: {output_tag}  (CSV/JSON/logs under this name)")
    print(
        f"gpu_memory_utilization={gpu_mem}, max_model_len={args.max_model_len}, "
        f"max_num_seqs={args.max_num_seqs}, tensor_parallel_size={args.tensor_parallel_size}"
        + (", allow_cuda_graphs=True" if args.allow_cuda_graphs else ", enforce_eager=True (default)")
    )
    print(f"Prompts: {list(prompts.keys())}")
    print(f"Token counts: {n_tokens_list}")
    print(f"Runs per config: {args.runs}\n")

    t_start = time.time()
    results = run_vllm_benchmark(
        model,
        n_tokens_list,
        prompts,
        args.runs,
        output_tag=output_tag,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=not args.allow_cuda_graphs,
    )
    elapsed = time.time() - t_start
    print(f"\nTotal benchmark time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    save_vllm_csv(results, output_tag)
    print_summary(results, label=output_tag)

    print("\nGenerating plots...")
    if output_tag == "7b":
        plot_three_way_comparison(results)
    else:
        print(
            "Skipping plot_3way_comparison_*.png: those charts read benchmark_results.csv, "
            "which is only merged for --preset 7b. Use 7B run or merge CSVs manually for 3-way plots."
        )

    print(f"\nAll vLLM outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    # vLLM v1 uses multiprocessing for EngineCore; default "fork" on Linux makes CUDA
    # unusable in children if the parent has touched the runtime. Spawn avoids that.
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
