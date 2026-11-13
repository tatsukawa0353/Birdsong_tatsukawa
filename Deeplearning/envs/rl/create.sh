#!/usr/bin/env bash
set -euo pipefail
rl_spec_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rl_project_dir="$(cd -- "$rl_spec_dir/../.." && pwd)"
rl_env_dir="$rl_project_dir/.venv-rl"
if [[ -e "$rl_env_dir" || -L "$rl_env_dir" ]]; then
    printf 'Refusing to overwrite existing environment: %s\n' "$rl_env_dir" >&2
    exit 1
fi
python3 -m venv --system-site-packages "$rl_env_dir"
PYTHONDONTWRITEBYTECODE=1 "$rl_env_dir/bin/python" - "$rl_spec_dir/system-constraints.txt" <<'PY'
from pathlib import Path
from importlib.metadata import version
import sys
for line in Path(sys.argv[1]).read_text().splitlines():
    name, expected = line.split('==')
    actual = version(name)
    if actual != expected:
        raise RuntimeError(f'System package mismatch: {name}: {actual} != {expected}')
PY
PYTHONDONTWRITEBYTECODE=1 "$rl_env_dir/bin/python" -m pip install --no-cache-dir --no-compile --no-deps \
    --constraint "$rl_spec_dir/system-constraints.txt" --requirement "$rl_spec_dir/requirements-local.txt"
