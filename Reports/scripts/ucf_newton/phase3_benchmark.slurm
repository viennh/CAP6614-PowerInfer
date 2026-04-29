#!/bin/bash
#SBATCH --job-name=phase3-bench
#SBATCH --output=phase3_benchmark_%j.out
#SBATCH --error=phase3_benchmark_%j.err
#SBATCH --mem=128gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
## Newton: uncomment if your site requires an explicit GPU partition:
##SBATCH --partition=gpu

# Phase 3: PowerInfer vs llama.cpp (see Reports/scripts/phase3_benchmark.py)
#
# Repo root is detected in this order: $POWERINFER_ROOT → $SLURM_SUBMIT_DIR (if
# build/bin/main exists) → path relative to this script → $HOME/PowerInfer
# Slurm often copies the batch file to a spool dir — submitting with:
#   cd /path/to/PowerInfer && sbatch Reports/scripts/ucf_newton/phase3_benchmark.slurm
# sets SLURM_SUBMIT_DIR correctly. Or: sbatch --export=ALL,POWERINFER_ROOT=/path/to/PowerInfer ...
#
# Override defaults (examples):
#   sbatch --gres=gpu:nvidia_h100_pcie:1 --export=ALL,PRESET=13b,NGL=40,TIMEOUT=7200 Reports/scripts/ucf_newton/phase3_benchmark.slurm
#   sbatch --export=ALL,PRESET=13b,NGL=40 Reports/scripts/ucf_newton/phase3_benchmark.slurm
#
# Requires: PowerInfer built with CUDA; LLAMA/llama.cpp built with CUDA at ~/PowerInfer/LLAMA/llama.cpp
#           (see Reports/ExperimentalSetup.md §1.6). Model GGUF paths must exist under the repo root.

set -euo pipefail

source ~/.env
source ~/vllm-env/bin/activate
module load gcc/gcc-12.2.0
module load cuda/cuda-12.6.0
module load cmake/cmake-3.31.5
module load anaconda/anaconda-2024.10

resolve_powerinfer_root() {
  if [[ -n "${POWERINFER_ROOT:-}" && -x "${POWERINFER_ROOT}/build/bin/main" ]]; then
    (cd "${POWERINFER_ROOT}" && pwd)
    return 0
  fi
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -x "${SLURM_SUBMIT_DIR}/build/bin/main" ]]; then
    (cd "${SLURM_SUBMIT_DIR}" && pwd)
    return 0
  fi
  local script_dir candidate
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  candidate="$(cd "$script_dir/../../.." && pwd)"
  if [[ -x "${candidate}/build/bin/main" ]]; then
    echo "$candidate"
    return 0
  fi
  if [[ -x "${HOME}/PowerInfer/build/bin/main" ]]; then
    (cd "${HOME}/PowerInfer" && pwd)
    return 0
  fi
  return 1
}

ROOT="$(resolve_powerinfer_root)" || ROOT=""

echo "=== Phase 3 benchmark ($(date -Is)) ==="
echo "Host: $(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "BASH_SOURCE=${BASH_SOURCE[0]}"

if [[ -z "$ROOT" ]]; then
  echo "ERROR: Cannot find PowerInfer repo containing build/bin/main." >&2
  echo "  Fix: cd /path/to/PowerInfer && sbatch Reports/scripts/ucf_newton/phase3_benchmark.slurm" >&2
  echo "  Or:  sbatch --export=ALL,POWERINFER_ROOT=/path/to/PowerInfer ..." >&2
  echo "  Then build: cmake -S . -B build -DLLAMA_CUBLAS=ON && cmake --build build --config Release" >&2
  exit 1
fi

cd "$ROOT"
echo "Repo: $ROOT"
echo "CUDA: $(command -v nvcc 2>/dev/null && nvcc --version | head -1 || echo 'nvcc not in PATH')"
nvidia-smi || true

if [[ ! -x ./build/bin/main ]]; then
  echo "ERROR: $ROOT/build/bin/main not executable. Build PowerInfer with CUDA first." >&2
  exit 1
fi

if [[ ! -f Reports/scripts/phase3_benchmark.py ]]; then
  echo "ERROR: Reports/scripts/phase3_benchmark.py not found under $ROOT" >&2
  exit 1
fi

VENV="${POWERINFER_VENV:-$ROOT/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
else
  echo "WARNING: No venv at $VENV — using system/python from PATH" >&2
fi

# Defaults (override with sbatch --export=PRESET=13b,NGL=40,QUICK=1,TIMEOUT=7200)
: "${PRESET:=7b}"
# NGL applies to BOTH PowerInfer `main` and llama.cpp (see phase3_benchmark.py). If unset, pick a
# safer default than 99 for 13B/70B F16 on typical 16–40 GB single-GPU nodes (99 is fine for 7B).
if [ -z "${NGL+x}" ]; then
  case "${PRESET}" in
    13b) NGL=40 ;;
    70b) NGL=24 ;;
    *)   NGL=99 ;;
  esac
fi
: "${THREADS:=8}"

ARGS=( --mode newton --preset "$PRESET" --ngl "$NGL" --threads "$THREADS" )
if [[ -n "${QUICK:-}" ]]; then
  ARGS+=( --quick )
fi
if [[ -n "${TIMEOUT:-}" ]]; then
  ARGS+=( --timeout "$TIMEOUT" )
fi

echo "Running: python Reports/scripts/phase3_benchmark.py ${ARGS[*]}"
python Reports/scripts/phase3_benchmark.py "${ARGS[@]}"

echo "=== Finished ($(date -Is)) ==="
