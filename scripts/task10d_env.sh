#!/usr/bin/env bash
# Reuse the compatible project environment and caches; no dependency changes.
TASK10D_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$TASK10D_ROOT/scripts/task10c_env.sh"
export PYTHONPATH="$TASK10D_ROOT/src:$TASK10D_ROOT/../robust_sav/src${PYTHONPATH:+:$PYTHONPATH}"
