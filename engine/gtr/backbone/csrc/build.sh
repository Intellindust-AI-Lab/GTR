#!/usr/bin/env bash
# GTR: Gated Token Recurrence for Efficient Dense Prediction
# Copyright (c) 2026 The GTR Authors. All Rights Reserved.

# Build the standalone gla_torch_ext extension in-place.
# Produces gla_torch_ext.*.so next to the sources so `import gla_torch_ext` works
# from this folder. Re-run after every kernel edit.
set -euo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
cd "$SRC_DIR"
rm -rf build *.so
"$PY" setup.py build_ext --inplace
echo "[build] done:"
ls -lh "$SRC_DIR"/*.so
