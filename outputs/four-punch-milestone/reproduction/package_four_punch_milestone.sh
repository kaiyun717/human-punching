#!/usr/bin/env bash
# Reproduce the selected four-family release. Pass --check-only for a read-only
# preflight; without it, all strict checks must pass before atomic publication.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-physics/bin/python -B scripts/package_punch_policies.py \
  --checkpoint jab=outputs/punch-5ms-5931736/checkpoint_00100.pt \
  --evaluation jab=outputs/punch-5ms-5931806 \
  --baseline jab=outputs/punch-5ms-5931807 \
  --checkpoint cross=outputs/punch-5ms-5931870/checkpoint_00100.pt \
  --evaluation cross=outputs/punch-5ms-5931915 \
  --baseline cross=outputs/punch-5ms-5931916 \
  --checkpoint left_hook=outputs/punch-5ms-5931834/checkpoint_00300.pt \
  --evaluation left_hook=outputs/punch-5ms-5931934 \
  --baseline left_hook=outputs/punch-5ms-5931935 \
  --checkpoint right_hook=outputs/punch-5ms-5931884/checkpoint_00100.pt \
  --evaluation right_hook=outputs/punch-5ms-5931918 \
  --baseline right_hook=outputs/punch-5ms-5931919 \
  --media-dir jab=outputs/jab-render-5931851 \
  --media-dir jab=outputs/jab-render-5931873 \
  --media-dir jab=outputs/punch-5ms-5931806/independent/nominal-plots \
  --media-dir jab=outputs/punch-5ms-5931807/independent/baseline-nominal-plots \
  --media-dir cross=outputs/jab-render-5931928 \
  --media-dir cross=outputs/jab-render-5931947 \
  --media-dir cross=outputs/punch-5ms-5931915/independent/nominal-plots \
  --media-dir cross=outputs/punch-5ms-5931916/independent/baseline-nominal-plots \
  --media-dir left_hook=outputs/jab-render-5931955 \
  --media-dir left_hook=outputs/jab-render-5931949 \
  --media-dir left_hook=outputs/punch-5ms-5931934/independent/nominal-plots \
  --media-dir left_hook=outputs/punch-5ms-5931935/independent/baseline-nominal-plots \
  --media-dir right_hook=outputs/jab-render-5931931 \
  --media-dir right_hook=outputs/jab-render-5931948 \
  --media-dir right_hook=outputs/punch-5ms-5931918/independent/nominal-plots \
  --media-dir right_hook=outputs/punch-5ms-5931919/independent/baseline-nominal-plots \
  --support-dir outputs/five-ms-final-validation \
  --support-dir outputs/five-ms-release-independent-audit \
  --require-precision --allow-conditioned-timing "$@"
