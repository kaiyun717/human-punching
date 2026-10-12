#!/usr/bin/env bash
# All outputs stay inside human-punching. Sibling mjlab is a read-only git source.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
revision=8ee51fbcf806a7419189f706d9e394cbeb7790fa
source_dir="$project_root/.cache/deps/mjlab-8ee51fbc"
export UV_CACHE_DIR="$project_root/.cache/uv"
export TMPDIR="$project_root/.cache/physics-install/tmp"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$source_dir"
# The accepted git object already exists in the sibling repository.
# git archive reads it without changing that repository's index, branch or files.
git -C ../mjlab archive "$revision" | tar -x -C "$source_dir"
uv venv --python /usr/bin/python3 --allow-existing .venv-physics
uv pip install --python .venv-physics/bin/python --index-strategy unsafe-best-match \
  -r requirements-physics.lock
uv pip check --python .venv-physics/bin/python
