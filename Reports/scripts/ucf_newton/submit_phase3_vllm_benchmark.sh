#!/usr/bin/env bash
# Submit Phase 3 vLLM benchmark to Slurm (Newton / UCF).
#
# Usage (from anywhere):
#   bash Reports/scripts/ucf_newton/submit_phase3_vllm_benchmark.sh
#   bash Reports/scripts/ucf_newton/submit_phase3_vllm_benchmark.sh --export=ALL,PRESET=7b,QUICK=1
#   bash Reports/scripts/ucf_newton/submit_phase3_vllm_benchmark.sh --export=ALL,PRESET=13b,GPU_MEM=0.82
#
# Or chmod +x and run from repo root:
#   ./Reports/scripts/ucf_newton/submit_phase3_vllm_benchmark.sh
#
# Optional first argument: absolute path to PowerInfer repo (defaults to inferring from this script).
# Remaining arguments are passed to sbatch (e.g. --partition=gpu, --job-name=..., --export=...).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ "${1:-}" == -* ]] || [[ -z "${1:-}" ]]; then
  REPO_ROOT="$DEFAULT_REPO"
  SBATCH_ARGS=("$@")
else
  REPO_ROOT="$(cd "$1" && pwd)"
  shift
  SBATCH_ARGS=("$@")
fi

if [[ ! -f "${REPO_ROOT}/Reports/scripts/phase3_vllm_benchmark.py" ]]; then
  echo "ERROR: Not a PowerInfer repo (missing Reports/scripts/phase3_vllm_benchmark.py): ${REPO_ROOT}" >&2
  exit 1
fi

cd "$REPO_ROOT"
echo "Submitting from SLURM_SUBMIT_DIR=$(pwd)"
exec sbatch "${SBATCH_ARGS[@]}" Reports/scripts/ucf_newton/phase3_vllm_benchmark.slurm
