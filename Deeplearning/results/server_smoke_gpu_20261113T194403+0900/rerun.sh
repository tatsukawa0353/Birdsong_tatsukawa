#!/usr/bin/env bash
set -euo pipefail
smoke_source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
smoke_repo_dir="$(cd -- "$smoke_source_dir/../../.." && pwd)"
smoke_python="$smoke_repo_dir/Deeplearning/results/server_smoke_20261113T190418+0900/.venv/bin/python"
smoke_new_output="$(mktemp -d "$smoke_repo_dir/Deeplearning/results/server_smoke_gpu_rerun_$(date +%Y%m%dT%H%M%S)_XXXXXX")"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$smoke_new_output/mpl-cache"
export PYTHONPYCACHEPREFIX="$smoke_new_output/pycache"
cd -- "$smoke_repo_dir"
printf 'Output: %s\n' "$smoke_new_output"
"$smoke_python" -u -m Deeplearning.server_smoke --device cuda:0 --output "$smoke_new_output" > "$smoke_new_output/execution.log" 2>&1
cat "$smoke_new_output/summary.json"
