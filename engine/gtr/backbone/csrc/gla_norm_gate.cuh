// GTR: Gated Token Recurrence for Efficient Dense Prediction
// Copyright (c) 2026 The GTR Authors. All Rights Reserved.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gla_norm_gate {

constexpr int kV = 64;

template <int kRowsPerBlock>
__launch_bounds__(kRowsPerBlock * 32)
__global__ void rmsnorm_gated_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ g,
    const __half* __restrict__ w,
    __half*       __restrict__ y,
    int N,
    float eps,
    float inv_D) {
    const int row = blockIdx.x * kRowsPerBlock + threadIdx.y;
    if (row >= N) return;

    const int lane = threadIdx.x;

    const __half2* x_row = reinterpret_cast<const __half2*>(x + row * kV);
    const __half2* g_row = reinterpret_cast<const __half2*>(g + row * kV);
    const __half2* w_row = reinterpret_cast<const __half2*>(w);

    __half2 x2 = x_row[lane];
    __half2 g2 = g_row[lane];
    __half2 w2 = w_row[lane];

    float xa = __low2float(x2);
    float xb = __high2float(x2);
    float ga = __low2float(g2);
    float gb = __high2float(g2);
    float wa = __low2float(w2);
    float wb = __high2float(w2);

    float ss = xa * xa + xb * xb;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, off);
    }

    const float rstd = rsqrtf(ss * inv_D + eps);

    const float silu_a = ga / (1.0f + __expf(-ga));
    const float silu_b = gb / (1.0f + __expf(-gb));

    const float ya = (xa * rstd) * wa * silu_a;
    const float yb = (xb * rstd) * wb * silu_b;

    __half2 out;
    out.x = __float2half_rn(ya);
    out.y = __float2half_rn(yb);
    __half2* y_row = reinterpret_cast<__half2*>(y + row * kV);
    y_row[lane] = out;
}

inline void launch_rmsnorm_gated(
    const __half* x, const __half* g, const __half* w, __half* y,
    int N, float eps, cudaStream_t stream) {
    constexpr int kRowsPerBlock = 8;
    const float inv_D = 1.0f / static_cast<float>(kV);
    const int n_blocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    dim3 grid(n_blocks);
    dim3 block(32, kRowsPerBlock);
    rmsnorm_gated_kernel<kRowsPerBlock><<<grid, block, 0, stream>>>(
        x, g, w, y, N, eps, inv_D);
}

}
