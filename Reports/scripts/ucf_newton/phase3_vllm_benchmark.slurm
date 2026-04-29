#!/bin/bash
#SBATCH --job-name=phase3-vllm
#SBATCH --output=phase3_vllm_benchmark_%j.out
#SBATCH --error=phase3_vllm_benchmark_%j.err
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
## UCF Newton — Hopper (H100) jobs:
##   Request H100 so you get a known stack (driver + SM_90); older GPUs can hide different bugs.
##SBATCH --constraint=h100
## Newton generic GPU pool (uncomment if --constraint=h100 is not valid on your account):
##SBATCH --partition=gpu
## Multi-GPU vLLM: TENSOR_PARALLEL must equal GPU count. Uncomment extra GPU line AND match N:
##SBATCH --gres=gpu:2
##   sbatch --gres=gpu:2 --export=ALL,PRESET=70b,TENSOR_PARALLEL=2 ...

# Phase 3 (vLLM only): Reports/scripts/phase3_vllm_benchmark.py
# Uses standard SiLU Llama-2 HF models (meta-llama/Llama-2-*-hf), NOT SparseLLM/ReluLLaMA.
#
# Repo root: $POWERINFER_ROOT → $SLURM_SUBMIT_DIR (if script exists) → relative to this file → $HOME/PowerInfer
# Submit from repo (cwd becomes SLURM_SUBMIT_DIR for resolve_powerinfer_root):
#   cd /path/to/PowerInfer && sbatch Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm
#
# Or use the wrapper (cd's to repo root, then sbatch):
#   bash Reports/scripts/ucf_newton/submit_phase3_vllm_benchmark.sh
#   bash Reports/scripts/ucf_newton/submit_phase3_vllm_benchmark.sh --export=ALL,PRESET=7b,QUICK=1
#
# Examples (default script uses #SBATCH --gres=gpu:1 → keep TENSOR_PARALLEL=1 or omit):
#   sbatch --export=ALL,PRESET=7b,QUICK=1 Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm
#   sbatch --export=ALL,PRESET=13b Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm
# Two GPUs + tensor parallel 2 (benchmark defaults to enforce_eager=True — no CUDA graphs):
#   sbatch --gres=gpu:2 --export=ALL,PRESET=70b,TENSOR_PARALLEL=2 Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm
# Opt into CUDA graphs for max throughput (may hit RuntimeError: cancelled on some nodes):
#   sbatch --export=ALL,PRESET=7b,ALLOW_CUDA_GRAPHS=1 Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm
#
# Gated Llama-2 weights: log in once with `huggingface-cli login` or set HF_TOKEN in the environment.

set -euo pipefail

# Reduce CUDA fragmentation OOMs during vLLM load (PyTorch 2.4+ prefers PYTORCH_ALLOC_CONF).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

[[ -f "${HOME}/.env" ]] && source "${HOME}/.env" || true

# Empty *VISIBLE_DEVICES breaks PyTorch / vLLM platform detection (Hopper included).
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]] && [[ -z "${CUDA_VISIBLE_DEVICES// }" ]]; then
  unset CUDA_VISIBLE_DEVICES
fi
if [[ -n "${NVIDIA_VISIBLE_DEVICES+x}" ]] && [[ -z "${NVIDIA_VISIBLE_DEVICES// }" ]]; then
  unset NVIDIA_VISIBLE_DEVICES
fi
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"

module load gcc/gcc-12.2.0
module load cuda/cuda-12.6.0
module load cmake/cmake-3.31.5
module load anaconda/anaconda-2024.10

# Prefer a venv that has vLLM + transformers + matplotlib (see ExperimentalSetup.md)
if [[ -f "${VLLM_VENV:-${HOME}/vllm-env}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VLLM_VENV:-${HOME}/vllm-env}/bin/activate"
elif [[ -f "${POWERINFER_VENV:-${HOME}/PowerInfer/.venv}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${POWERINFER_VENV:-${HOME}/PowerInfer/.venv}/bin/activate"
else
  echo "WARNING: No vLLM venv found. Set VLLM_VENV or create ${HOME}/vllm-env with vllm, matplotlib, transformers." >&2
fi

resolve_powerinfer_root() {
  local c
  if [[ -n "${POWERINFER_ROOT:-}" && -f "${POWERINFER_ROOT}/Reports/scripts/phase3_vllm_benchmark.py" ]]; then
    (cd "${POWERINFER_ROOT}" && pwd)
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/Reports/scripts/phase3_vllm_benchmark.py" ]]; then
    (cd "${SLURM_SUBMIT_DIR}" && pwd)
    return 0
  fi
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  c="$(cd "$script_dir/../../.." && pwd)"
  if [[ -f "${c}/Reports/scripts/phase3_vllm_benchmark.py" ]]; then
    echo "$c"
    return 0
  fi
  if [[ -f "${HOME}/PowerInfer/Reports/scripts/phase3_vllm_benchmark.py" ]]; then
    (cd "${HOME}/PowerInfer" && pwd)
    return 0
  fi
  return 1
}

ROOT="$(resolve_powerinfer_root)" || ROOT=""

echo "=== Phase 3 vLLM benchmark ($(date -Is)) ==="
echo "Host: $(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "Python: $(command -v python3) ($({ python3 --version; } 2>&1))"

if [[ -z "$ROOT" ]]; then
  echo "ERROR: Cannot find PowerInfer repo (Reports/scripts/phase3_vllm_benchmark.py missing)." >&2
  echo "  cd /path/to/PowerInfer && sbatch Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm" >&2
  echo "  Or: sbatch --export=ALL,POWERINFER_ROOT=/path/to/PowerInfer ..." >&2
  exit 1
fi

cd "$ROOT"
echo "Repo: $ROOT"
echo "SLURM: JOB_ID=${SLURM_JOB_ID:-} JOB_GPUS=${SLURM_JOB_GPUS:-} STEP_GPUS=${SLURM_STEP_GPUS:-} LOCALID=${SLURM_LOCALID:-}"
echo "GPU env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "CUDA: $(command -v nvcc 2>/dev/null && nvcc --version | head -1 || echo 'nvcc not in PATH')"
nvidia-smi || true

if [[ ! -f Reports/scripts/phase3_vllm_benchmark.py ]]; then
  echo "ERROR: Reports/scripts/phase3_vllm_benchmark.py not found under $ROOT" >&2
  exit 1
fi

python3 -c "import vllm" 2>/dev/null || {
  echo "ERROR: vllm is not importable in this Python. Install: pip install vllm matplotlib" >&2
  exit 1
}

: "${PRESET:=7b}"
: "${RUNS:=1}"
: "${TENSOR_PARALLEL:=1}"

# Safer defaults for 13B on 16 GB GPUs (set before MAX_MODEL_LEN default)
if [[ "$PRESET" == "13b" ]]; then
  : "${MAX_MODEL_LEN:=256}"
  : "${GPU_MEM:=0.70}"
else
  : "${MAX_MODEL_LEN:=512}"
fi

ARGS=( --preset "$PRESET" --runs "$RUNS" --tensor-parallel-size "$TENSOR_PARALLEL" --max-model-len "$MAX_MODEL_LEN" )
if [[ -n "${QUICK:-}" ]]; then
  ARGS+=( --quick )
fi
if [[ -n "${GPU_MEM:-}" ]]; then
  ARGS+=( --gpu-memory-utilization "$GPU_MEM" )
fi
if [[ -n "${MODEL:-}" ]]; then
  ARGS+=( --model "$MODEL" )
fi
# Benchmark defaults to enforce_eager=True in Python. Opt in to CUDA graphs with:
if [[ "${ALLOW_CUDA_GRAPHS:-0}" == "1" ]]; then
  ARGS+=( --allow-cuda-graphs )
fi

echo "Running: python3 Reports/scripts/phase3_vllm_benchmark.py ${ARGS[*]}"
python3 Reports/scripts/phase3_vllm_benchmark.py "${ARGS[@]}"

echo "=== Finished ($(date -Is)) ==="
