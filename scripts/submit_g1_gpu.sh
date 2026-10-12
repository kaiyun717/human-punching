#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
mkdir -p outputs/slurm
exec sbatch --parsable scripts/target_g1_volta.sbatch "$@"
