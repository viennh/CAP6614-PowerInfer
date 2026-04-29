#!/bin/bash
#SBATCH --job-name=phase4-neuron
#SBATCH --output=phase4_neuron_variation_%j.out
#SBATCH --error=phase4_neuron_variation_%j.err
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
## Newton: uncomment if required:
##SBATCH --partition=gpu
## 70B with HF accelerate across GPUs / heavy offload — request more memory time or multiple GPUs as needed:
##SBATCH --mem=256gb
##SBATCH --gres=gpu:2

# Phase 4: Reports/scripts/phase4_neuron_variation.py
# Loads SparseLLM ReluLLaMA (HF) and records ReLU FFN activations — needs torch, transformers, matplotlib.
#
# Repo root: $POWERINFER_ROOT → $SLURM_SUBMIT_DIR → path from this script → $HOME/PowerInfer
#   cd /path/to/PowerInfer && sbatch Reports/scripts/ucf_newton/phase4_neuron_variation.slurm
#
# Examples:
#   sbatch --export=ALL,PRESET=7b Reports/scripts/ucf_newton/phase4_neuron_variation.slurm
#   sbatch --export=ALL,PRESET=13b,MAX_NEW_TOKENS=32 Reports/scripts/ucf_newton/phase4_neuron_variation.slurm
#   sbatch --export=ALL,PRESET=70b,DEVICE=auto Reports/scripts/ucf_newton/phase4_neuron_variation.slurm
#
# Ensure ./SparseLLM-ReluLLaMA-{7B,13B,70B} exists under the repo (or set MODEL_PATH). HF_TOKEN if using remote hub.

set -euo pipefail

[[ -f "${HOME}/.env" ]] && source "${HOME}/.env" || true

module load gcc/gcc-12.2.0
module load cuda/cuda-12.6.0
module load cmake/cmake-3.31.5
module load anaconda/anaconda-2024.10

if [[ -f "${POWERINFER_VENV:-${HOME}/PowerInfer/.venv}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${POWERINFER_VENV:-${HOME}/PowerInfer/.venv}/bin/activate"
elif [[ -f "${VLLM_VENV:-${HOME}/vllm-env}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VLLM_VENV:-${HOME}/vllm-env}/bin/activate"
else
  echo "WARNING: No venv at POWERINFER_VENV / ~/PowerInfer/.venv / ~/vllm-env — need torch, transformers, matplotlib." >&2
fi

resolve_powerinfer_root() {
  if [[ -n "${POWERINFER_ROOT:-}" && -f "${POWERINFER_ROOT}/Reports/scripts/phase4_neuron_variation.py" ]]; then
    (cd "${POWERINFER_ROOT}" && pwd)
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/Reports/scripts/phase4_neuron_variation.py" ]]; then
    (cd "${SLURM_SUBMIT_DIR}" && pwd)
    return 0
  fi
  local script_dir c
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  c="$(cd "$script_dir/../../.." && pwd)"
  if [[ -f "${c}/Reports/scripts/phase4_neuron_variation.py" ]]; then
    echo "$c"
    return 0
  fi
  if [[ -f "${HOME}/PowerInfer/Reports/scripts/phase4_neuron_variation.py" ]]; then
    (cd "${HOME}/PowerInfer" && pwd)
    return 0
  fi
  return 1
}

ROOT="$(resolve_powerinfer_root)" || ROOT=""

echo "=== Phase 4 neuron variation ($(date -Is)) ==="
echo "Host: $(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "Python: $(command -v python3) ($({ python3 --version; } 2>&1))"

if [[ -z "$ROOT" ]]; then
  echo "ERROR: Cannot find PowerInfer repo (Reports/scripts/phase4_neuron_variation.py missing)." >&2
  echo "  cd /path/to/PowerInfer && sbatch Reports/scripts/ucf_newton/phase4_neuron_variation.slurm" >&2
  echo "  Or: sbatch --export=ALL,POWERINFER_ROOT=/path/to/PowerInfer ..." >&2
  exit 1
fi

cd "$ROOT"
echo "Repo: $ROOT"
echo "CUDA: $(command -v nvcc 2>/dev/null && nvcc --version | head -1 || echo 'nvcc not in PATH')"
nvidia-smi || true

if [[ ! -f Reports/scripts/phase4_neuron_variation.py ]]; then
  echo "ERROR: Reports/scripts/phase4_neuron_variation.py not found under $ROOT" >&2
  exit 1
fi

python3 -c "import torch, transformers, matplotlib" 2>/dev/null || {
  echo "ERROR: need torch, transformers, matplotlib in the active Python (pip install -r requirements.txt; pip install matplotlib)." >&2
  exit 1
}

: "${PRESET:=7b}"
: "${MAX_NEW_TOKENS:=64}"
: "${THRESHOLD:=0.1}"
# Default to GPU on CUDA nodes; use DEVICE=auto for 70B multi-GPU/offload, or cpu for CPU-only jobs.
: "${DEVICE:=cuda}"

ARGS=( --preset "$PRESET" --max-new-tokens "$MAX_NEW_TOKENS" --threshold "$THRESHOLD" )
if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=( --device "$DEVICE" )
fi
if [[ -n "${MODEL_PATH:-}" ]]; then
  ARGS+=( --model-path "$MODEL_PATH" )
fi

echo "Running: python3 Reports/scripts/phase4_neuron_variation.py ${ARGS[*]}"
python3 Reports/scripts/phase4_neuron_variation.py "${ARGS[@]}"

echo "=== Finished ($(date -Is)) ==="
