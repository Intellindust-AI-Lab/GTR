#!/usr/bin/env bash
# Compile the GatedLinearAttention TensorRT plugin against the GTR csrc kernels.
#
#   bash build.sh                 # RTX 4090 (sm_89)
#   SM="86;89" bash build.sh      # multi-arch fatbin
#
# Env:
#   TRT_ROOT  — TensorRT install prefix holding include/ and lib/ (default: tensorrt_plugin/tensorrt)
#   CUDA_HOME — CUDA toolkit (default /usr/local/cuda-12.8; nvcc major must match the TRT CUDA build)
#   GLA_SRC   — directory with gla_chunk.cuh (default: engine/gtr/backbone/csrc, used as-is)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TRT_ROOT="${TRT_ROOT:-${HERE}/../tensorrt}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
GLA_SRC="${GLA_SRC:-${HERE}/../../engine/gtr/backbone/csrc}"
SM="${SM:-89}"

GENCODE=""
IFS=';' read -ra ARCHS <<< "${SM}"
for a in "${ARCHS[@]}"; do
  GENCODE+=" -gencode=arch=compute_${a},code=sm_${a}"
done

test -f "${TRT_ROOT}/include/NvInfer.h" || { echo "NvInfer.h not found under ${TRT_ROOT}/include" >&2; exit 1; }
test -f "${GLA_SRC}/gla_chunk.cuh" || { echo "gla_chunk.cuh not found under ${GLA_SRC}" >&2; exit 1; }

echo "[build] SM=${SM} TRT_ROOT=${TRT_ROOT} GLA_SRC=${GLA_SRC}"
"${CUDA_HOME}/bin/nvcc" \
  -O3 -std=c++17 --use_fast_math \
  -Xcompiler -fPIC -shared \
  ${GENCODE} \
  -I"${TRT_ROOT}/include" -I"${GLA_SRC}" -I"${CUDA_HOME}/include" \
  -L"${TRT_ROOT}/lib" -L"${CUDA_HOME}/lib64" \
  -Xlinker -rpath -Xlinker "${TRT_ROOT}/lib" \
  -lnvinfer -lcudart \
  "${HERE}/gla_plugin.cu" -o "${HERE}/libgla_plugin.so"
echo "[build] produced ${HERE}/libgla_plugin.so"
ls -lh "${HERE}/libgla_plugin.so"
