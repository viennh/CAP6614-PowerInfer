# Experimental Setup for PowerInfer Course Project

## Current State Summary

- PowerInfer repo is **cloned and built** (`build/bin/main` exists)
- **ReluLLaMA-7B** model is downloaded (~14 GB GGUF + 32 layer activation profiles)
- A successful test run was completed (128-token generation from "Once upon a time")
- Hardware: **Apple M4 Max** (32 GPU cores, ARM/NEON, Metal-capable)

## Important Hardware Caveat

PowerInfer's core speedup comes from its **GPU-CPU hybrid split** (hot neurons on GPU, cold on CPU), which is designed for **NVIDIA CUDA GPUs**. On Apple Silicon, it runs CPU-only with NEON/BLAS — the hot/cold offloading to a discrete GPU doesn't apply. GPU experiments will be run on the **UCF Newton HPC cluster**. The profiling and analysis portions can be done locally.

### UCF Newton and **16 GB VRAM** (e.g. Tesla V100-PCIE-16GB)

Many shared GPU nodes expose **~16 GB** per GPU. Rough weight-only sizes for **FP16/BF16** dense weights are about **7B → ~14 GB**, **13B → ~26 GB**, **70B → ~140 GB** — so **full model + KV cache on one 16 GB GPU is only realistic for ~7B** (with headroom for typical short runs). Plan accordingly:

| Goal on 16 GB | What to do |
|----------------|------------|
| **PowerInfer** ReluLLaMA-7B | Usually OK for `-n 128`–style runs; watch `nvidia-smi` if you increase context or batch. |
| **PowerInfer** 13B / 70B | Often **will not** fit entirely on 16 GB; use a **larger-GPU** queue if your site offers 24–40 GB+ GPUs, or accept **CPU / mixed** behavior and document it. |
| **llama.cpp dense F16** + `-ngl 99` | Treat as **7B-only** on 16 GB. For **13B/70B**, **quantize** (e.g. `llama-quantize` → `Q4_K_M` / `Q5_K_M`) and then `-ngl 99`, **or** lower `-ngl` (partial GPU) / `-ngl 0` (CPU, needs large host RAM for 13B+). |
| **vLLM** `dtype=float16` | **7B** is the safe default on 16 GB. **13B** usually needs a bigger GPU, **multi-GPU**, or **smaller dtype / quantization** — check vLLM docs and your card’s free memory. |
| **Phase 2** (`phase2_profile_neurons.py`) | **CPU + disk only** — no VRAM requirement beyond a normal Python env. |

**llama.cpp: quantized baseline on 16 GB (example for 13B after `convert-dense.py`):**

```bash
# From llama.cpp build dir; pick a quantization type that fits (Q4_K_M is a common default)
./build/bin/llama-quantize \
  ./ReluLLaMA-13B-dense/llama-13b-relu.gguf \
  ./ReluLLaMA-13B-dense/llama-13b-relu-q4_k_m.gguf Q4_K_M

./build/bin/llama-completion \
  -m ./ReluLLaMA-13B-dense/llama-13b-relu-q4_k_m.gguf \
  -n 32 -t 8 -ngl 99 -p "Once upon a time"
```

If full-GPU **F16** comparison is required for 13B/70B, schedule jobs on nodes with **≥24 GB (ideally 40 GB+)** VRAM when the cluster provides them.

---

## Phase 1: Environment and Baseline Setup

### Step 1.1 — Verify local prerequisites

**Status: done.** The following are already in place:

| Component | Version / Path | Status |
|-----------|---------------|--------|
| Python | 3.10.11 (`.venv/bin/python`) | Installed |
| CMake | 4.2.3 (`/opt/homebrew/bin/cmake`) | Installed |
| PowerInfer binary | `build/bin/main` | Built |
| ReluLLaMA-7B model | `ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf` (~14 GB) | Downloaded |
| Activation profiles | `ReluLLaMA-7B/activation/activation_{0..31}.pt` (32 files) | Downloaded |
| Python packages | numpy 2.2.6, torch 2.10.0, transformers 5.3.0, sentencepiece 0.2.1 | Installed |

Init Environment on UCF:
```bash
module load anaconda/anaconda-2024.10
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Confirm everything works by running:
```bash
source .venv/bin/activate
./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf -n 16 -t 8 -p "Hello world"
```

### Step 1.2 — Rebuild PowerInfer with Apple Accelerate (recommended)

The current build has `LLAMA_METAL=OFF` and `LLAMA_BLAS=OFF`, meaning it uses plain CPU only. On Apple M4 Max you should rebuild with the **Accelerate** BLAS backend for faster matrix operations:

```bash
# Remove the old build
rm -rf build

# Rebuild with Apple Accelerate framework
cmake -S . -B build -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=Apple
cmake --build build --config Release
```

Verify the rebuild by checking the `system_info` line in the output:
```bash
./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf -n 16 -t 8 -p "Once upon a time"
# Look for: BLAS = 1 in system_info output
```

> **Note:** `LLAMA_METAL` (GPU compute via Metal) is listed as unsupported for sparse inference in the README. Accelerate BLAS is the best local option.

### Step 1.3 — Install missing Python packages

The virtual environment is missing `matplotlib` (needed for Phase 2 visualization):

```bash
source .venv/bin/activate
pip install matplotlib
```

### Step 1.4 — Create the results directory structure

Organize outputs for all phases:

```bash
mkdir -p Reports/results/phase2_profiling
mkdir -p Reports/results/phase3_benchmarks
mkdir -p Reports/results/phase4_neuron_variation
mkdir -p Reports/scripts
```

### Step 1.5 — Set up UCF Newton HPC for GPU experiments

PowerInfer's GPU-CPU hybrid offloading requires an NVIDIA GPU. Use the **UCF Newton cluster**.

#### 1.5.1 — Connect to Newton

```bash
ssh <NID>@newton.ist.ucf.edu
```

#### 1.5.2 — Request an interactive GPU session

```bash
srun --mem=32gb --gres=gpu:1 --time=8:00:00 --pty bash
```

> Use `--gres=gpu:1` (one GPU is sufficient for PowerInfer single-GPU inference). Add `--gres=gpu:2` only if you want to compare multi-GPU vLLM later. Use `--partition=gpu` if your cluster requires it.

#### 1.5.3 — Load required modules

Newton provides CUDA, CMake, GCC, and Python via the module system:

```bash
module load gcc/gcc-12.2.0
module load cuda/cuda-12.6.0
module load cmake/cmake-3.31.5
module load anaconda/anaconda-2024.10

# Verify
gcc --version        # should show 12.2.0
nvcc --version       # should show CUDA 12.6
cmake --version      # should show 3.31.5
nvidia-smi           # should show the allocated GPU
```

Available CUDA versions on Newton: `11.3, 11.6.0, 11.8.0, 12.1.0, 12.2.2, 12.3.0, 12.4.0, 12.6.0, 13.1.0`. Use **12.6.0** for best compatibility.

#### 1.5.4 — Clone and build PowerInfer on Newton

```bash
# Clone into your home or scratch directory
cd ~/projects   # or wherever you store work on Newton
git clone https://github.com/SJTU-IPADS/PowerInfer.git
cd PowerInfer

# Create a Python virtual environment using Anaconda's Python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install matplotlib huggingface-hub

# Build PowerInfer with CUDA
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
```

#### 1.5.5 — Download the ReluLLaMA-7B model on Newton

```bash
huggingface-cli download --resume-download \
    --local-dir ReluLLaMA-7B \
    --local-dir-use-symlinks False \
    PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF


huggingface-cli download --resume-download \
    --local-dir ReluLLaMA-13B \
    --local-dir-use-symlinks False \
    PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF

huggingface-cli download --resume-download \
    --local-dir ReluLLaMA-70B \
    --local-dir-use-symlinks False \
    PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF

```

> **Storage tip:** If your home directory quota is tight, download to a scratch directory (`/scratch/<NID>/`) and symlink.

#### 1.5.6 — Verify PowerInfer runs with GPU offloading

**7B (recommended first test on 16 GB VRAM):**

```bash
./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf \
    -n 32 -t 8 -p "Once upon a time"
```

**13B / 70B:** On **16 GB** GPUs these may **OOM** or spill unpredictably — prefer a **larger-GPU** partition if available; otherwise run only after confirming memory (see *UCF Newton and 16 GB VRAM* above).

```bash
./build/bin/main -m ./ReluLLaMA-13B/llama-13b-relu.powerinfer.gguf \
    -n 32 -t 8 -p "Once upon a time"

./build/bin/main -m ./ReluLLaMA-70B/llama-70b-relu.powerinfer.gguf \
    -n 32 -t 8 -p "Once upon a time"
```

Check the output for:
- `BLAS = 1` in the `system_info` line
- GPU offloading messages (e.g., "offloading X layers to GPU")
- Token generation speed in the timing summary

#### 1.5.7 — (Optional) Create a SLURM batch script for longer runs

Save as `run_powerinfer.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=powerinfer
#SBATCH --output=powerinfer_%j.log
#SBATCH --error=powerinfer_%j.err
#SBATCH --mem=64gb
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

module load gcc/gcc-12.2.0
module load cuda/cuda-12.6.0
module load cmake/cmake-3.31.5
module load anaconda/anaconda-2024.10

cd ~/projects/PowerInfer
source .venv/bin/activate

./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf \
    -n 128 -t 8 -p "Once upon a time"
```

Submit with:
```bash
sbatch run_powerinfer.slurm
```

### Step 1.6 — Set up llama.cpp baseline (on Newton)

llama.cpp is the primary speed comparison target. Install it **separately** from PowerInfer.

> **Prerequisite:** You should already be in a GPU session with modules loaded (Step 1.5.2–1.5.3).

```bash
# 1. Clone upstream llama.cpp alongside PowerInfer
cd ~/PowerInfer
mkdir LLAMA
cd LLAMA
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 2. Build with CUDA (modules already loaded from Step 1.5.3)
#    Note: upstream llama.cpp deprecated LLAMA_CUBLAS — use GGML_CUDA instead
cmake -S . -B build -DGGML_CUDA=ON

# UCF
cmake -S . -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_HOST_COMPILER=/apps/gcc/gcc-12.2.0/bin/g++

cmake --build build --config Release

# 3. Download the original SparseLLM model for conversion
cd ~/PowerInfer
source .venv/bin/activate
huggingface-cli download --resume-download \
    --local-dir SparseLLM-ReluLLaMA-7B \
    --local-dir-use-symlinks False \
    SparseLLM/ReluLLaMA-7B

huggingface-cli download --resume-download \
    --local-dir SparseLLM-ReluLLaMA-13B \
    --local-dir-use-symlinks False \
    SparseLLM/ReluLLaMA-13B


huggingface-cli download --resume-download \
    --local-dir SparseLLM-ReluLLaMA-70B \
    --local-dir-use-symlinks False \
    SparseLLM/ReluLLaMA-70B

# 4. Convert to dense GGUF format (compatible with llama.cpp)
mkdir -p ReluLLaMA-7B-dense
python convert-dense.py \
    --outfile ./ReluLLaMA-7B-dense/llama-7b-relu.gguf \
    --outtype f16 \
    ./SparseLLM-ReluLLaMA-7B

mkdir -p ReluLLaMA-13B-dense
python convert-dense.py \
    --outfile ./ReluLLaMA-13B-dense/llama-13b-relu.gguf \
    --outtype f16 \
    ./SparseLLM-ReluLLaMA-13B

mkdir -p ReluLLaMA-70B-dense
python convert-dense.py \
    --outfile ./ReluLLaMA-70B-dense/llama-70b-relu.gguf \
    --outtype f16 \
    ./SparseLLM-ReluLLaMA-70B

# 5. Verify: run inference with llama.cpp (offload layers to GPU)
#    Use llama-completion (exits after generation, unlike llama-cli which enters interactive mode)
#    On 16 GB VRAM: F16 + -ngl 99 is realistic for 7B only; for 13B/70B quantize first (see "16 GB VRAM" above).
~/PowerInfer/LLAMA/llama.cpp/build/bin/llama-completion \
    -m ./ReluLLaMA-7B-dense/llama-7b-relu.gguf \
    -n 32 -t 8 -ngl 99 -p "Once upon a time"

~/PowerInfer/LLAMA/llama.cpp/build/bin/llama-completion \
    -m ./ReluLLaMA-13B-dense/llama-13b-relu.gguf \
    -n 32 -t 8 -ngl 99 -p "Once upon a time"

~/PowerInfer/LLAMA/llama.cpp/build/bin/llama-completion \
    -m ./ReluLLaMA-70B-dense/llama-70b-relu.gguf \
    -n 32 -t 8 -ngl 99 -p "Once upon a time"
```

> **Important:** The dense GGUF from `convert-dense.py` uses altered activation functions (ReLU) and may not behave identically with upstream llama.cpp. This is expected — the comparison measures inference engine speed on the same model weights.

### Step 1.7 — Set up vLLM baseline (on Newton)

vLLM is a high-throughput serving engine and the second comparison target.

> **Prerequisite:** GPU session with modules loaded (Step 1.5.2–1.5.3), PowerInfer `.venv` activated.

```bash
# 1. Install vLLM into the existing virtual environment
cd ~/PowerInfer
source .venv/bin/activate
pip install vllm

# 2. Verify it loads the model and generates output
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='SparseLLM/ReluLLaMA-7B', dtype='float16')
output = llm.generate(['Once upon a time'], SamplingParams(max_tokens=32))
print(output[0].outputs[0].text)
"
```

> **Note:** vLLM does not exploit activation sparsity — it runs dense inference. This makes it a useful upper-bound reference for standard GPU serving throughput.
>
> **If vLLM install fails** due to CUDA version mismatch, try: `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu126` to match the `cuda/cuda-12.6.0` module.

### Step 1.8 — Prepare a local llama.cpp baseline (optional, for Apple M4 Max)

For preliminary CPU-only comparisons on your Mac:

```bash
# 1. Clone upstream llama.cpp alongside PowerInfer
cd ~/MyWorkspace/UCF/UCF-Spring-Courses/CurrentTopicsML/CourseProject
git clone https://github.com/ggerganov/llama.cpp llama-cpp-upstream
cd llama-cpp-upstream

# 2. Build with Accelerate (no GPU)
cmake -S . -B build -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=Apple
cmake --build build --config Release

# 3. Convert the model you already have to dense GGUF
cd ../PowerInfer
python convert-dense.py \
    --outfile ./ReluLLaMA-7B-dense/llama-7b-relu.gguf \
    --outtype f16 \
    ./SparseLLM-ReluLLaMA-7B  # download first if not available locally

# 4. Run and compare timing with PowerInfer locally
# PowerInfer (CPU-only, Accelerate)
./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf \
    -n 128 -t 8 -p "Once upon a time" 2>&1 | tee Reports/results/phase3_benchmarks/powerinfer_cpu_local.log

# llama.cpp (CPU-only, Accelerate)
#   Use llama-completion (exits after generation, unlike llama-cli which enters interactive mode)
../llama-cpp-upstream/build/bin/llama-completion \
    -m ./ReluLLaMA-7B-dense/llama-7b-relu.gguf \
    -n 128 -t 8 -p "Once upon a time" 2>&1 | tee Reports/results/phase3_benchmarks/llamacpp_cpu_local.log
```

### Step 1.9 — Verification checklist

Before moving to Phase 2, confirm all boxes are checked:

- [ ] PowerInfer builds and runs locally with `BLAS = 1` in system_info
- [ ] `ReluLLaMA-7B` model + 32 activation profiles present
- [ ] `matplotlib` installed in `.venv`
- [ ] Results directories created under `Reports/`
- [ ] (Newton) Modules load: `gcc/gcc-12.2.0`, `cuda/cuda-12.6.0`, `cmake/cmake-3.31.5`, `anaconda/anaconda-2024.10`
- [ ] (Newton) PowerInfer builds with CUDA and runs with GPU offloading
- [ ] (Newton) llama.cpp builds with CUDA and runs (**on 16 GB:** `-ngl 99` with **7B** F16 dense, or **13B+** using **quantized** GGUF / lower `-ngl` / larger-GPU node)
- [ ] (Newton) vLLM loads and generates output (**7B** on 16 GB; **13B+** may need more VRAM or different settings)
- [ ] Benchmark prompt set drafted (see Phase 3, Step 3.1)

---

## Phase 2: Profile Hot/Cold Neuron Distribution

The activation statistics are already in `ReluLLaMA-7B/activation/activation_{0..31}.pt` (one per transformer layer).

### Step 2.1 — Write a profiling analysis script

Load and analyze the pre-computed activation statistics:

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

for layer_idx in range(32):
    act = torch.load(f"ReluLLaMA-7B/activation/activation_{layer_idx}.pt")
    sorted_act, _ = torch.sort(act, descending=True)

    total = sorted_act.sum()
    cumulative = torch.cumsum(sorted_act, dim=0) / total

    hot_80 = (cumulative < 0.80).sum().item()
    hot_90 = (cumulative < 0.90).sum().item()

    print(f"Layer {layer_idx}: {hot_80}/{len(act)} neurons cover 80% of activations "
          f"({100*hot_80/len(act):.1f}%)")
```

### Step 2.2 — Visualize the power-law distribution

Generate plots that show:
1. **Neuron activation frequency** ranked by frequency (log scale) — demonstrates the power-law
2. **Cumulative activation coverage** vs. percentage of neurons — shows the 80/20 split
3. **Heatmap across layers** — shows which layers have more/less skewed distributions

```bash
python Reports/scripts/phase2_profile_neurons.py

# 13B / 70B: activation_*.pt live under the PowerInfer GGUF download (e.g. ReluLLaMA-13B/activation/),
# not under SparseLLM/ReluLLaMA-* (those repos are base weights only). Download e.g.:
#   huggingface-cli download ... PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF --local-dir ReluLLaMA-13B
python Reports/scripts/phase2_profile_neurons.py --preset 13b

python Reports/scripts/phase2_profile_neurons.py --preset 70b
```
---

## Phase 3: Inference Speed Benchmarking

### Step 3.1 — Define benchmark prompts

Create a diverse prompt set representing different input types:

| Category | Example Prompt | Purpose |
|----------|---------------|---------|
| Creative writing | "Once upon a time in a magical kingdom..." | Narrative generation |
| Code generation | "Write a Python function that sorts a list..." | Technical/structured |
| Factual QA | "Explain the process of photosynthesis..." | Knowledge retrieval |
| Reasoning | "If all roses are flowers and some flowers fade..." | Logical chains |
| Conversational | "Hey, how are you doing today?" | Short casual |

### Step 3.2 — Run benchmarks systematically

For each framework, measure across the prompt set with consistent parameters:

```bash
# On Newton — make sure modules are loaded and .venv is activated first:
#   module load gcc/gcc-12.2.0 cuda/cuda-12.6.0 cmake/cmake-3.31.5 anaconda/anaconda-2024.10
#   source .venv/bin/activate

# PowerInfer (uses GPU; on 16 GB prefer 7B for headroom)
./build/bin/main -m ./ReluLLaMA-7B/llama-7b-relu.powerinfer.gguf \
    -n 128 -t 8 -p "$PROMPT" 2>&1 | tee Reports/results/phase3_benchmarks/powerinfer_run.log

# 13B: may require >16 GB VRAM — use only if your GPU fits, or use a larger-GPU queue
./build/bin/main -m ./ReluLLaMA-13B/llama-13b-relu.powerinfer.gguf \
    -n 128 -t 8 -p "$PROMPT" 2>&1 | tee Reports/results/phase3_benchmarks/powerinfer_13b_run.log

# llama.cpp — on 16 GB use F16 dense + -ngl 99 for 7B; for 13B use a quantized .gguf (see Step 1.6)
#   Use llama-completion (not llama-cli) — it exits after generation instead of entering interactive mode
~/PowerInfer/LLAMA/llama.cpp/build/bin/llama-completion \
    -m ./ReluLLaMA-7B-dense/llama-7b-relu.gguf \
    -n 128 -t 8 -ngl 99 -p "$PROMPT" 2>&1 | tee Reports/results/phase3_benchmarks/llamacpp_run.log

# vLLM (Python)
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='SparseLLM/ReluLLaMA-7B', dtype='float16')
out = llm.generate(['$PROMPT'], SamplingParams(max_tokens=128))
print(out[0].outputs[0].text)
" 2>&1 | tee Reports/results/phase3_benchmarks/vllm_run.log
```

### Step 3.3 — Metrics to collect

For each run, extract:
- **Prompt eval time** (prefill, ms)
- **Token generation rate** (tokens/sec during generation)
- **End-to-end latency** (total time for prompt + N generated tokens)
- **Peak GPU memory usage** (`nvidia-smi` or the framework's own reporting)
- **Peak CPU memory usage**

PowerInfer and llama.cpp both print timing stats at the end of a run. Parse them from the logs.

Note: Fix issue on UCF
```bash
python -m venv ~/vllm-env --clear
source ~/vllm-env/bin/activate
pip install --upgrade pip
pip install vllm matplotlib transformers
# vLLM baseline must be standard SiLU LLaMA — not SparseLLM/ReluLLaMA (ReLU is unsupported)
python Reports/scripts/phase3_vllm_benchmark.py --quick
```

```bash
python Reports/scripts/phase3_benchmark.py --mode local

# PowerInfer vs llama.cpp — same ReluLLaMA checkpoints as convert-dense (13B / 70B)
# Outputs: benchmark_results_13b.csv, plot_speed_comparison_n128_13b.png, logs/preset_13b/…
python Reports/scripts/phase3_benchmark.py --mode newton --preset 13b --quick
python Reports/scripts/phase3_benchmark.py --mode newton --preset 70b --quick
# On ~16 GB VRAM with F16 dense, lower llama.cpp GPU layers, e.g.:
# python Reports/scripts/phase3_benchmark.py --mode newton --preset 13b --ngl 24 --quick
# If you see TIMEOUT after 600s, pull latest script (Newton 13b default is 3600s/run) or set e.g. --timeout 7200
```
On Newton (NVIDIA GPU required):
```bash
# Load modules and activate venv
module load gcc/gcc-12.2.0 cuda/cuda-12.6.0 cmake/cmake-3.31.5 anaconda/anaconda-2024.10
source .venv/bin/activate
# Quick test (1 prompt, 32 tokens)
python Reports/scripts/phase3_vllm_benchmark.py --quick
# Full benchmark (all 5 prompts x 3 token lengths)
python Reports/scripts/phase3_vllm_benchmark.py

# Llama-2 13B / 70B (official HF `meta-llama/Llama-2-*-hf`; vLLM cannot load ReluLLaMA)
# On 16 GB VRAM, 13B may need: --gpu-memory-utilization 0.85  (script defaults 0.88 for --preset 13b)
python Reports/scripts/phase3_vllm_benchmark.py --preset 13b --quick
python Reports/scripts/phase3_vllm_benchmark.py --preset 70b --tensor-parallel-size 2 --quick

python Reports/scripts/phase3_vllm_benchmark.py --model huggyllama/llama-7b
# Local SiLU LLaMA only (do not pass ./SparseLLM-ReluLLaMA-* — script preflight will exit)
# python Reports/scripts/phase3_vllm_benchmark.py --model /path/to/meta-llama-Llama-2-7b-hf
python Reports/scripts/phase3_vllm_benchmark.py --preset 13b


sbatch --constraint=h100  --gres=gpu:2 --export=ALL,PRESET=13b,NGL=24 Reports/scripts/ucf_newton/phase3_benchmark.slurm

./Reports/scripts/ucf_newton/submit_phase3_vllm_benchmark.sh --export=ALL,PRESET=13b,QUICK=1

./Reports/scripts/ucf_newton/submit_phase3_vllm_benchmark.sh /lustre/fs1/home/ha813608/PowerInfer \
  --export=ALL,PRESET=13b,GPU_MEM=0.82


sbatch --constraint=h100 --export=ALL,PRESET=13b,TENSOR_PARALLEL=2 Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm
sbatch --constraint=h100 --gres=gpu:2 --export=ALL,PRESET=13b Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm

sbatch --constraint=h100 --gres=gpu:2 --export=ALL,PRESET=13b Reports/scripts/ucf_newton/phase4_neuron_variation.slurm
```

---

## Phase 4: Analyze Hot Neuron Variation Across Input Types

### Step 4.1 — Instrument PowerInfer to log per-input activation

Modify PowerInfer's source to record which neurons fire for each prompt. The key file is `llama.cpp` (in the repo root) where the FFN forward pass happens. Look for the sparse neuron evaluation path and add logging that dumps the activated neuron indices per layer per token.

Alternatively, use the existing activation profiler approach:
- Run inference on each prompt category separately
- Dump the set of activated neurons for each category
- Compare intersection/difference across categories

### Step 4.2 — Compute overlap metrics

For each pair of input categories, compute:
- **Jaccard similarity** of hot neuron sets
- **Overlap coefficient** (intersection / min set size)
- Number of **universally hot** neurons (hot across all categories)
- Number of **category-specific** neurons (hot only for one category)

### Step 4.3 — Visualize

- Heatmap of pairwise Jaccard similarity across prompt categories
- Venn/UpSet diagram of neuron sets
- Per-layer breakdown: does the overlap pattern change in early vs. late layers?


```bash
# 7B (default paths: SparseLLM-ReluLLaMA-7B → Reports/results/phase4_neuron_variation/)
python Reports/scripts/phase4_neuron_variation.py --max-new-tokens 32

python Reports/scripts/phase4_neuron_variation.py --preset 13b
# 70B: use Hugging Face accelerate if one GPU is not enough
python Reports/scripts/phase4_neuron_variation.py --preset 70b --device auto --max-new-tokens 32


python Reports/scripts/render_phase2_plots_from_layer_summary.py \
    Reports/results/phase2_profiling_13b/layer_summary.csv \
    Reports/results/phase2_profiling_13b \
    "ReluLLaMA-13B"

 python Reports/scripts/render_phase4_jaccard_heatmap.py \
    Reports/results/phase4_neuron_variation_13b/overlap_results.json \
    Reports/results/phase4_neuron_variation_13b/plot1_jaccard_heatmap.png \
    "ReluLLaMA-13B"
```
---

