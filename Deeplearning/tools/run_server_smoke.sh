#!/usr/bin/env bash
set -euo pipefail
smoke_tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
smoke_repo_dir="$(cd -- "$smoke_tools_dir/../.." && pwd)"
smoke_python="$smoke_repo_dir/Deeplearning/.venv-rl/bin/python"
smoke_requested_device="${1:-cpu}"
if [[ "$smoke_requested_device" == cpu ]]; then
    smoke_device=cpu
elif [[ "$smoke_requested_device" =~ ^cuda:([0-9]+)$ ]]; then
    export CUDA_VISIBLE_DEVICES="${BASH_REMATCH[1]}"
    smoke_device=cuda:0
else
    printf 'Use cpu or cuda:N (physical GPU index).\n' >&2
    exit 2
fi
if [[ ! -x "$smoke_python" ]]; then
    printf 'Create the RL environment using Deeplearning/envs/rl/create.sh first.\n' >&2
    exit 1
fi
smoke_output="$(mktemp -d "$smoke_repo_dir/Deeplearning/results/server_smoke_$(date +%Y%m%dT%H%M%S)_XXXXXX")"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd -- "$smoke_repo_dir"
printf 'Output: %s\n' "$smoke_output"
"$smoke_python" -u -m Deeplearning.server_smoke --device "$smoke_device" --output "$smoke_output" > "$smoke_output/execution.log" 2>&1
cat "$smoke_output/summary.json"
