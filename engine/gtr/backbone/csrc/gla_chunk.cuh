// GTR: Gated Token Recurrence for Efficient Dense Prediction
// Copyright (c) 2026 The GTR Authors. All Rights Reserved.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace gla_chunk {

constexpr int kK  = 32;
constexpr int kV  = 64;
constexpr int kBT = 64;

__device__ __forceinline__ uint32_t cvta_to_shared_u32(const void* ptr) {
    uint32_t r;
    asm volatile("{ .reg .u64 q; cvta.to.shared.u64 q, %1; cvt.u32.u64 %0, q; }"
                 : "=r"(r) : "l"(ptr));
    return r;
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const __nv_bfloat16* row0col0, int ld_elem, int lane) {
#if __CUDA_ARCH__ >= 800
    const int row_off = lane & 15;
    const int col_off = (lane >> 4) * 8;
    const uint32_t s = cvta_to_shared_u32(row0col0 + row_off * ld_elem + col_off);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
#else
    (void)r0; (void)r1; (void)r2; (void)r3;
    (void)row0col0; (void)ld_elem; (void)lane;
    __trap();
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const __nv_bfloat16* row0col0, int ld_elem, int lane) {
#if __CUDA_ARCH__ >= 800
    const int row_off = lane & 15;
    const int col_off = (lane >> 4) * 8;
    const uint32_t s = cvta_to_shared_u32(row0col0 + row_off * ld_elem + col_off);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
#else
    (void)r0; (void)r1; (void)r2; (void)r3;
    (void)row0col0; (void)ld_elem; (void)lane;
    __trap();
#endif
}

__device__ __forceinline__ void cp_async16(
    void* smem_dst, const void* gmem_src, bool valid) {
#if __CUDA_ARCH__ >= 800
    const uint32_t s = cvta_to_shared_u32(smem_dst);
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.u32 p, %2, 0;\n"
        " cp.async.ca.shared.global [%0], [%1], 16, p;\n"
        "}\n" :: "r"(s), "l"(gmem_src), "r"(valid ? 0u : 1u));
#else
    (void)smem_dst; (void)gmem_src; (void)valid;
    __trap();
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n");
#endif
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
    (void)c0; (void)c1; (void)c2; (void)c3;
    __trap();
#endif
}

__device__ __forceinline__ void mma_m16n8k16_f16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
    (void)c0; (void)c1; (void)c2; (void)c3;
    __trap();
#endif
}

__device__ __forceinline__ unsigned ld_acquire_gpu_u32(const unsigned* p) {
#if __CUDA_ARCH__ >= 700
    unsigned v;
    asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
#else
    __trap(); return 0u;
#endif
}

__device__ __forceinline__ void st_release_gpu_u32(unsigned* p, unsigned v) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.gpu.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
#else
    (void)p; (void)v;
    __trap();
#endif
}

__device__ __forceinline__ void atom_add_release_gpu_u32(unsigned* p, unsigned v) {
#if __CUDA_ARCH__ >= 700
    unsigned old;
    asm volatile("atom.global.add.release.gpu.u32 %0, [%1], %2;"
                 : "=r"(old) : "l"(p), "r"(v) : "memory");
    (void)old;
#else
    (void)p; (void)v;
    __trap();
#endif
}

__device__ __forceinline__ float logsigmoid_f(float x) {
    return x >= 0.f ? -log1pf(__expf(-x))
                    : x - log1pf(__expf(x));
}

__launch_bounds__(128, 4)
__global__ void chunk_compute_S_kernel(
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ gk,
    __half*       __restrict__ G,
    float*        __restrict__ S,
    float*        __restrict__ eG_last,
    int T, int H, int NT,
    float inv_normalizer);

constexpr int kNvTiles = 4;
constexpr int kVTile   = kV / kNvTiles;

template <bool kFuseGate = false, bool kInlineScan = false>
__launch_bounds__(128, 3)
__global__ void chunk_fwd_o_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ G,
    const float*  __restrict__ h_in,
    __half*       __restrict__ o,
    int T, int H, int NT, float scale,
    const __half* __restrict__ g_in = nullptr,
    const __half* __restrict__ rms_w = nullptr,
    float rms_eps = 1e-6f,
    const float* __restrict__ S_ws = nullptr,
    const float* __restrict__ eL_ws = nullptr) {
#if __CUDA_ARCH__ < 800
    if (threadIdx.x == 0 && threadIdx.y == 0) __trap();
    return;
#else
    const int i_t  = blockIdx.x;
    const int i_bh = blockIdx.y;
    const int b = i_bh / H;
    const int h = i_bh % H;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int tig     = lane & 3;

    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int t_base    = i_t * kBT;
    const int chunk_len = min(kBT, T - t_base);

    const int stride_t_k = H * kK;
    const int stride_t_v = H * kV;
    const int base_k = (b * T * H + h) * kK;
    const int base_v = (b * T * H + h) * kV;

    const int h_stride_chunk = H * kK * kV;
    const int h_stride_bh    = (NT + 1) * h_stride_chunk;
    const float* h_chunk = h_in + b * h_stride_bh + i_t * h_stride_chunk + h * kK * kV;

    constexpr int kPadQK = 4;
    constexpr int kPad = 8;
    __shared__ __nv_bfloat16 sQt[kBT][kK + kPadQK];
    __shared__ __nv_bfloat16 sKt[kBT][kK + kPadQK];
    __shared__ __half        sV [kBT][kV + kPad];
    __shared__ __nv_bfloat16 sH [kK][kV + kPad];
    __shared__ __half        sA [kBT][kBT + kPad];

#pragma unroll
    for (int i = tid; i < kBT * kV / 8; i += 128) {
        const int r  = i >> 3;
        const int c8 = (i & 7) * 8;
        const int t  = t_base + r;
        const int t_clamped = t < T ? t : T - 1;
        cp_async16(&sV[r][c8],
                   v + (size_t)base_v + (size_t)t_clamped * stride_t_v + c8,
                   t < T);
    }
    cp_async_commit();

#pragma unroll
    for (int i = tid; i < kBT * kK / 8; i += 128) {
        const int r  = i >> 2;
        const int c8 = (i & 3) * 8;
        const int t  = t_base + r;
        uint4 qv8 = make_uint4(0, 0, 0, 0), kv8 = make_uint4(0, 0, 0, 0);
        uint4 gv8 = make_uint4(0, 0, 0, 0);
        if (t < T) {
            const size_t off = (size_t)base_k + (size_t)t * stride_t_k + c8;
            qv8 = *reinterpret_cast<const uint4*>(q + off);
            kv8 = *reinterpret_cast<const uint4*>(k + off);
            gv8 = *reinterpret_cast<const uint4*>(G + off);
        }
        const __half2* qh2 = reinterpret_cast<const __half2*>(&qv8);
        const __half2* kh2 = reinterpret_cast<const __half2*>(&kv8);
        const __half2* gh2 = reinterpret_cast<const __half2*>(&gv8);
        __nv_bfloat162 qb[4], kb[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 qf = __half22float2(qh2[p]);
            const float2 kf = __half22float2(kh2[p]);
            const float2 gf = __half22float2(gh2[p]);
            const float eg0 = __expf(gf.x), eg1 = __expf(gf.y);
            qb[p] = __floats2bfloat162_rn(qf.x * eg0, qf.y * eg1);
            kb[p] = __floats2bfloat162_rn(kf.x / eg0, kf.y / eg1);
        }
        *reinterpret_cast<uint2*>(&sQt[r][c8])     = *reinterpret_cast<const uint2*>(&qb[0]);
        *reinterpret_cast<uint2*>(&sQt[r][c8 + 4]) = *reinterpret_cast<const uint2*>(&qb[2]);
        *reinterpret_cast<uint2*>(&sKt[r][c8])     = *reinterpret_cast<const uint2*>(&kb[0]);
        *reinterpret_cast<uint2*>(&sKt[r][c8 + 4]) = *reinterpret_cast<const uint2*>(&kb[2]);
    }
    if constexpr (kInlineScan) {
        const int r  = tid >> 2;
        const int c8 = (tid & 3) * 16;
        const size_t S_off  = ((size_t)(b * NT) * H + h) * kK * kV + r * kV + c8;
        const size_t eL_off = ((size_t)(b * NT) * H + h) * kK + r;
        const size_t S_step  = (size_t)H * kK * kV;
        const size_t eL_step = (size_t)H * kK;
        float hacc[16] = {};
#pragma unroll 2
        for (int cc = 0; cc < i_t; ++cc) {
            const float d = __ldg(eL_ws + eL_off + (size_t)cc * eL_step);
            const float4 s0 = *reinterpret_cast<const float4*>(S_ws + S_off + (size_t)cc * S_step);
            const float4 s1 = *reinterpret_cast<const float4*>(S_ws + S_off + (size_t)cc * S_step + 4);
            const float4 s2 = *reinterpret_cast<const float4*>(S_ws + S_off + (size_t)cc * S_step + 8);
            const float4 s3 = *reinterpret_cast<const float4*>(S_ws + S_off + (size_t)cc * S_step + 12);
            const float sv[16] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w,
                                  s2.x, s2.y, s2.z, s2.w, s3.x, s3.y, s3.z, s3.w};
#pragma unroll
            for (int j = 0; j < 16; ++j) hacc[j] = d * hacc[j] + sv[j];
        }
        __nv_bfloat162 hb[8];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            hb[p] = __floats2bfloat162_rn(hacc[2 * p], hacc[2 * p + 1]);
        }
        *reinterpret_cast<uint4*>(&sH[r][c8])     = *reinterpret_cast<const uint4*>(&hb[0]);
        *reinterpret_cast<uint4*>(&sH[r][c8 + 8]) = *reinterpret_cast<const uint4*>(&hb[4]);
    } else {
#pragma unroll
        for (int i = tid; i < kK * kV / 8; i += 128) {
            const int r  = i >> 3;
            const int c8 = (i & 7) * 8;
            const float4 h01 = *reinterpret_cast<const float4*>(h_chunk + r * kV + c8);
            const float4 h23 = *reinterpret_cast<const float4*>(h_chunk + r * kV + c8 + 4);
            __nv_bfloat162 hb[4];
            hb[0] = __floats2bfloat162_rn(h01.x, h01.y);
            hb[1] = __floats2bfloat162_rn(h01.z, h01.w);
            hb[2] = __floats2bfloat162_rn(h23.x, h23.y);
            hb[3] = __floats2bfloat162_rn(h23.z, h23.w);
            *reinterpret_cast<uint4*>(&sH[r][c8]) = *reinterpret_cast<const uint4*>(hb);
        }
    }
    __syncthreads();

    auto pack = [&](__nv_bfloat16 h0, __nv_bfloat16 h1) -> uint32_t {
        __nv_bfloat162 p = __halves2bfloat162(h0, h1);
        return *reinterpret_cast<const uint32_t*>(&p);
    };
    const int m_row_base = wm * 32;
    const int n_col_base = wn * 32;

    float a_acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int f = 0; f < 4; ++f) a_acc[mt][nt][f] = 0.f;

#pragma unroll
    for (int k_tile = 0; k_tile < 2; ++k_tile) {
        const int kc_base = k_tile * 16;

#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int m_base = m_row_base + mt * 16;
            if (n_col_base >= m_base + 16) continue;
            uint32_t A[4];
            const int r0 = m_base + gid;
            const int r1 = m_base + gid + 8;
            const int c0 = kc_base + 2 * tig;
            const int c8 = kc_base + 2 * tig + 8;
            A[0] = pack(sQt[r0][c0    ], sQt[r0][c0 + 1]);
            A[1] = pack(sQt[r1][c0    ], sQt[r1][c0 + 1]);
            A[2] = pack(sQt[r0][c8    ], sQt[r0][c8 + 1]);
            A[3] = pack(sQt[r1][c8    ], sQt[r1][c8 + 1]);

#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int n_base = n_col_base + nt * 8;
                if (n_base >= m_base + 16) continue;
                uint32_t B[2];
                const int col_n = n_base + gid;
                B[0] = pack(sKt[col_n][kc_base + 2 * tig],     sKt[col_n][kc_base + 2 * tig + 1]);
                B[1] = pack(sKt[col_n][kc_base + 2 * tig + 8], sKt[col_n][kc_base + 2 * tig + 9]);
                mma_m16n8k16_bf16(
                    a_acc[mt][nt][0], a_acc[mt][nt][1], a_acc[mt][nt][2], a_acc[mt][nt][3],
                    A[0], A[1], A[2], A[3],
                    B[0], B[1],
                    a_acc[mt][nt][0], a_acc[mt][nt][1], a_acc[mt][nt][2], a_acc[mt][nt][3]);
            }
        }
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int m_base = m_row_base + mt * 16;
            const int n_base = n_col_base + nt * 8;
            if (n_base >= m_base + 16) continue;
            const int r0 = m_base + gid;
            const int r1 = m_base + gid + 8;
            const int c0 = n_base + 2 * tig;
            const int c1 = c0 + 1;
            const float m0 = (c0 <= r0) ? a_acc[mt][nt][0] : 0.f;
            const float m1 = (c1 <= r0) ? a_acc[mt][nt][1] : 0.f;
            const float m2 = (c0 <= r1) ? a_acc[mt][nt][2] : 0.f;
            const float m3 = (c1 <= r1) ? a_acc[mt][nt][3] : 0.f;
            *reinterpret_cast<__half2*>(&sA[r0][c0]) = __floats2half2_rn(m0, m1);
            *reinterpret_cast<__half2*>(&sA[r1][c0]) = __floats2half2_rn(m2, m3);
        }
    }
    cp_async_wait_all();
    __syncthreads();

    float oi_acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int f = 0; f < 4; ++f) oi_acc[mt][nt][f] = 0.f;

    constexpr int sA_ld = kBT + kPad;
    constexpr int sV_ld = kV + kPad;
#pragma unroll
    for (int k_tile = 0; k_tile < 4; ++k_tile) {
        const int kc_base = k_tile * 16;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int m_base = m_row_base + mt * 16;
            if (kc_base > m_base) continue;
            uint32_t A[4];
            ldmatrix_x4(A[0], A[1], A[2], A[3],
                        reinterpret_cast<const __nv_bfloat16*>(&sA[m_base][kc_base]),
                        sA_ld, lane);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int n_pair = n_col_base + p * 16;
                uint32_t B[4];
                ldmatrix_x4_trans(B[0], B[1], B[2], B[3],
                                  reinterpret_cast<const __nv_bfloat16*>(&sV[kc_base][n_pair]),
                                  sV_ld, lane);
                const int nt_a = 2 * p + 0;
                const int nt_b = 2 * p + 1;
                mma_m16n8k16_f16(
                    oi_acc[mt][nt_a][0], oi_acc[mt][nt_a][1], oi_acc[mt][nt_a][2], oi_acc[mt][nt_a][3],
                    A[0], A[1], A[2], A[3], B[0], B[1],
                    oi_acc[mt][nt_a][0], oi_acc[mt][nt_a][1], oi_acc[mt][nt_a][2], oi_acc[mt][nt_a][3]);
                mma_m16n8k16_f16(
                    oi_acc[mt][nt_b][0], oi_acc[mt][nt_b][1], oi_acc[mt][nt_b][2], oi_acc[mt][nt_b][3],
                    A[0], A[1], A[2], A[3], B[2], B[3],
                    oi_acc[mt][nt_b][0], oi_acc[mt][nt_b][1], oi_acc[mt][nt_b][2], oi_acc[mt][nt_b][3]);
            }
        }
    }

    float oe_acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int f = 0; f < 4; ++f) oe_acc[mt][nt][f] = 0.f;

    constexpr int sH_ld = kV + kPad;
#pragma unroll
    for (int k_tile = 0; k_tile < 2; ++k_tile) {
        const int kc_base = k_tile * 16;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int m_base = m_row_base + mt * 16;
            uint32_t A[4];
            const int r0 = m_base + gid;
            const int r1 = m_base + gid + 8;
            const int c0 = kc_base + 2 * tig;
            const int c8 = kc_base + 2 * tig + 8;
            A[0] = pack(sQt[r0][c0    ], sQt[r0][c0 + 1]);
            A[1] = pack(sQt[r1][c0    ], sQt[r1][c0 + 1]);
            A[2] = pack(sQt[r0][c8    ], sQt[r0][c8 + 1]);
            A[3] = pack(sQt[r1][c8    ], sQt[r1][c8 + 1]);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int n_pair = n_col_base + p * 16;
                uint32_t B[4];
                ldmatrix_x4_trans(B[0], B[1], B[2], B[3],
                                  &sH[kc_base][n_pair], sH_ld, lane);
                const int nt_a = 2 * p + 0;
                const int nt_b = 2 * p + 1;
                mma_m16n8k16_bf16(
                    oe_acc[mt][nt_a][0], oe_acc[mt][nt_a][1], oe_acc[mt][nt_a][2], oe_acc[mt][nt_a][3],
                    A[0], A[1], A[2], A[3], B[0], B[1],
                    oe_acc[mt][nt_a][0], oe_acc[mt][nt_a][1], oe_acc[mt][nt_a][2], oe_acc[mt][nt_a][3]);
                mma_m16n8k16_bf16(
                    oe_acc[mt][nt_b][0], oe_acc[mt][nt_b][1], oe_acc[mt][nt_b][2], oe_acc[mt][nt_b][3],
                    A[0], A[1], A[2], A[3], B[2], B[3],
                    oe_acc[mt][nt_b][0], oe_acc[mt][nt_b][1], oe_acc[mt][nt_b][2], oe_acc[mt][nt_b][3]);
            }
        }
    }

    if constexpr (kFuseGate) {
        __half (*sO)[kV] = reinterpret_cast<__half(*)[kV]>(&sV[0][0]);
        __half (*sG)[kV] = reinterpret_cast<__half(*)[kV]>(&sA[0][0]);
        __syncthreads();

#pragma unroll
        for (int i = tid; i < kBT * kV / 8; i += 128) {
            const int r  = i >> 3;
            const int c8 = (i & 7) * 8;
            const int t  = t_base + r;
            const int t_clamped = t < T ? t : T - 1;
            cp_async16(&sG[r][c8],
                       g_in + (size_t)base_v + (size_t)t_clamped * stride_t_v + c8,
                       t < T);
        }
        cp_async_commit();

#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int m_base = m_row_base + mt * 16;
                const int n_base = n_col_base + nt * 8;
                const int r0 = m_base + gid;
                const int r1 = m_base + gid + 8;
                const int c0 = n_base + 2 * tig;
                const int c1 = c0 + 1;
                const float o00 = scale * (oi_acc[mt][nt][0] + oe_acc[mt][nt][0]);
                const float o01 = scale * (oi_acc[mt][nt][1] + oe_acc[mt][nt][1]);
                const float o10 = scale * (oi_acc[mt][nt][2] + oe_acc[mt][nt][2]);
                const float o11 = scale * (oi_acc[mt][nt][3] + oe_acc[mt][nt][3]);
                sO[r0][c0] = __float2half(o00);
                sO[r0][c1] = __float2half(o01);
                sO[r1][c0] = __float2half(o10);
                sO[r1][c1] = __float2half(o11);
            }
        }

        __half2 w_local = *reinterpret_cast<const __half2*>(&rms_w[2 * lane]);
        const float w0 = __low2float(w_local);
        const float w1 = __high2float(w_local);
        constexpr float kInvV = 1.0f / static_cast<float>(kV);

        cp_async_wait_all();
        __syncthreads();

        const int row_base = warp_id * 16;
#pragma unroll 1
        for (int rr = 0; rr < 16; ++rr) {
            const int r = row_base + rr;
            if (r >= chunk_len) break;

            __half2 o2 = *reinterpret_cast<const __half2*>(&sO[r][2 * lane]);
            __half2 g2 = *reinterpret_cast<const __half2*>(&sG[r][2 * lane]);
            const float o0 = __low2float(o2);
            const float o1 = __high2float(o2);
            const float g0 = __low2float(g2);
            const float g1 = __high2float(g2);

            float ss = o0 * o0 + o1 * o1;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ss += __shfl_xor_sync(0xffffffff, ss, off);
            }
            const float rstd = rsqrtf(ss * kInvV + rms_eps);

            const float silu_g0 = g0 / (1.0f + __expf(-g0));
            const float silu_g1 = g1 / (1.0f + __expf(-g1));
            const float y0 = (o0 * rstd) * w0 * silu_g0;
            const float y1 = (o1 * rstd) * w1 * silu_g1;

            const int t = t_base + r;
            __half2 y2 = __halves2half2(__float2half_rn(y0), __float2half_rn(y1));
            *reinterpret_cast<__half2*>(&o[base_v + t * stride_t_v + 2 * lane]) = y2;
        }
    } else {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int m_base = m_row_base + mt * 16;
                const int n_base = n_col_base + nt * 8;
                const int r0 = m_base + gid;
                const int r1 = m_base + gid + 8;
                const int c0 = n_base + 2 * tig;
                const float o00 = scale * (oi_acc[mt][nt][0] + oe_acc[mt][nt][0]);
                const float o01 = scale * (oi_acc[mt][nt][1] + oe_acc[mt][nt][1]);
                const float o10 = scale * (oi_acc[mt][nt][2] + oe_acc[mt][nt][2]);
                const float o11 = scale * (oi_acc[mt][nt][3] + oe_acc[mt][nt][3]);
                const int t0 = t_base + r0;
                const int t1 = t_base + r1;
                if (t0 < T) {
                    *reinterpret_cast<__half2*>(&o[base_v + t0 * stride_t_v + c0]) =
                        __floats2half2_rn(o00, o01);
                }
                if (t1 < T) {
                    *reinterpret_cast<__half2*>(&o[base_v + t1 * stride_t_v + c0]) =
                        __floats2half2_rn(o10, o11);
                }
            }
        }
    }
#endif
}

__launch_bounds__(128, 4)
__global__ void chunk_compute_S_kernel(
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ gk,
    __half*       __restrict__ G,
    float*        __restrict__ S,
    float*        __restrict__ eG_last,
    int T, int H, int NT,
    float inv_normalizer) {
#if __CUDA_ARCH__ < 800
    if (threadIdx.x == 0 && threadIdx.y == 0) __trap();
    return;
#else
    const int i_t  = blockIdx.x;
    const int i_bh = blockIdx.y;
    const int b = i_bh / H;
    const int h = i_bh % H;

    const int tid_k = threadIdx.x;
    const int tid_v = threadIdx.y;
    const int tid_flat = tid_v * blockDim.x + tid_k;
    const int lane = tid_k;
    const int gid  = lane >> 2;
    const int tig  = lane & 3;

    const int wm = tid_v >> 1;
    const int wn = tid_v & 1;

    const int t_base    = i_t * kBT;
    const int chunk_len = min(kBT, T - t_base);

    const int stride_t_k = H * kK;
    const int stride_t_v = H * kV;
    const int base_k = (b * T * H + h) * kK;
    const int base_v = (b * T * H + h) * kV;

    constexpr int kPad = 8;
    __shared__ __half sKt[kBT][kK + kPad];
    __shared__ __half sV [kBT][kV + kPad];

    __shared__ float   seG_warp_sum[4][kK];

#pragma unroll
    for (int i = tid_flat; i < kBT * kV / 8; i += 128) {
        const int r  = i >> 3;
        const int c8 = (i & 7) * 8;
        const int t  = t_base + r;
        const int t_clamped = t < T ? t : T - 1;
        cp_async16(&sV[r][c8],
                   v + (size_t)base_v + (size_t)t_clamped * stride_t_v + c8,
                   t < T);
    }
    cp_async_commit();

    constexpr int kRowsPerWarp = kBT / 4;
    const int r_start = tid_v * kRowsPerWarp;
    const bool fold_logsigmoid = (inv_normalizer > 0.f);
    float local_acc[kRowsPerWarp];

    {
        const int c = tid_k;
        float acc = 0.f;
#pragma unroll
        for (int rr = 0; rr < kRowsPerWarp; ++rr) {
            const int r = r_start + rr;
            const int t = t_base + r;
            float gv = 0.f;
            if (t < T) {
                gv = __half2float(gk[base_k + t * stride_t_k + c]);
                if (fold_logsigmoid) {
                    gv = logsigmoid_f(gv) * inv_normalizer;
                }
            }
            acc += gv;
            local_acc[rr] = acc;
        }
        seG_warp_sum[tid_v][c] = acc;
    }

    __syncthreads();

    {
        const int c = tid_k;
        const float s0 = seG_warp_sum[0][c];
        const float s1 = seG_warp_sum[1][c];
        const float s2 = seG_warp_sum[2][c];
        const float s3 = seG_warp_sum[3][c];
        const float warp_offset = (tid_v > 0 ? s0 : 0.f)
                                + (tid_v > 1 ? s1 : 0.f)
                                + (tid_v > 2 ? s2 : 0.f);
        const float total = s0 + s1 + s2 + s3;

#pragma unroll
        for (int rr = 0; rr < kRowsPerWarp; ++rr) {
            const int r = r_start + rr;
            const int t = t_base + r;
            if (t < T) {
                const float G_r = warp_offset + local_acc[rr];
                G[base_k + t * stride_t_k + c] = __float2half(G_r);
                const float kraw =
                    __half2float(k[base_k + t * stride_t_k + c]);
                sKt[r][c] = __float2half(kraw * __expf(total - G_r));
            } else {
                sKt[r][c] = __float2half(0.f);
            }
        }

        if (tid_v == 0) {
            eG_last[((b * NT + i_t) * H + h) * kK + c] = __expf(total);
        }
    }
    cp_async_wait_all();
    __syncthreads();

    float acc[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int f = 0; f < 4; ++f) acc[n][f] = 0.f;

    const int m_row_base = wm * 16;
    const int n_col_base = wn * 32;

    constexpr int sKt_ld = kK + kPad;
    constexpr int sV_ld  = kV + kPad;
#pragma unroll
    for (int k_tile = 0; k_tile < 4; ++k_tile) {
        const int kc_base = k_tile * 16;
        uint32_t Ar[4];
        ldmatrix_x4_trans(Ar[0], Ar[1], Ar[2], Ar[3],
                          reinterpret_cast<const __nv_bfloat16*>(&sKt[kc_base][m_row_base]),
                          sKt_ld, lane);
        const uint32_t A0 = Ar[0];
        const uint32_t A1 = Ar[2];
        const uint32_t A2 = Ar[1];
        const uint32_t A3 = Ar[3];
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int n_pair = n_col_base + p * 16;
            uint32_t B[4];
            ldmatrix_x4_trans(B[0], B[1], B[2], B[3],
                              reinterpret_cast<const __nv_bfloat16*>(&sV[kc_base][n_pair]),
                              sV_ld, lane);
            const int nt_a = 2 * p + 0;
            const int nt_b = 2 * p + 1;
            mma_m16n8k16_f16(
                acc[nt_a][0], acc[nt_a][1], acc[nt_a][2], acc[nt_a][3],
                A0, A1, A2, A3, B[0], B[1],
                acc[nt_a][0], acc[nt_a][1], acc[nt_a][2], acc[nt_a][3]);
            mma_m16n8k16_f16(
                acc[nt_b][0], acc[nt_b][1], acc[nt_b][2], acc[nt_b][3],
                A0, A1, A2, A3, B[2], B[3],
                acc[nt_b][0], acc[nt_b][1], acc[nt_b][2], acc[nt_b][3]);
        }
    }

    const int S_stride_bh = NT * H * kK * kV;
    const int S_stride_t  = H * kK * kV;
    const int S_stride_h  = kK * kV;
    float* S_bh = S + b * S_stride_bh + i_t * S_stride_t + h * S_stride_h;

#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const int n_col = n_col_base + n * 8;
        const int r0 = m_row_base + gid;
        const int r1 = m_row_base + gid + 8;
        const int c0 = n_col + 2 * tig;
        *reinterpret_cast<float2*>(&S_bh[r0 * kV + c0]) = make_float2(acc[n][0], acc[n][1]);
        *reinterpret_cast<float2*>(&S_bh[r1 * kV + c0]) = make_float2(acc[n][2], acc[n][3]);
    }
#endif
}

inline int next_pow2_ge(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

__global__ void chunk_scan_h_seq_kernel(
    const float* __restrict__ S,
    const float* __restrict__ eG_last,
    float*       __restrict__ h_out,
    int H, int NT) {
    const int i_bh = blockIdx.x;
    const int i_k  = blockIdx.y;
    const int b = 0;
    const int h = i_bh % H;
    const int v = threadIdx.x;

    const int hk = h * kK + i_k;
    const int stride_t   = H * kK * kV;
    const int eG_stride_t = H * kK;

    const float* S_ptr  = S       + (size_t)(b * NT)       * stride_t   + hk * kV + v;
    float*       h_ptr  = h_out   + (size_t)(b * (NT + 1)) * stride_t   + hk * kV + v;
    const float* eG_ptr = eG_last + (size_t)(b * NT)       * eG_stride_t + hk;

    h_ptr[0] = 0.0f;
    float h_state = 0.0f;
    if (NT <= 16) {
#pragma unroll 4
        for (int c = 0; c < NT; ++c) {
            const float s = S_ptr[(size_t)c * stride_t];
            const float D = eG_ptr[(size_t)c * eG_stride_t];
            h_state = D * h_state + s;
            h_ptr[(size_t)(c + 1) * stride_t] = h_state;
        }
        return;
    }
    constexpr int PF = 16;
    float s_buf[PF], d_buf[PF];
#pragma unroll
    for (int j = 0; j < PF; ++j) {
        if (j < NT) {
            s_buf[j] = S_ptr[(size_t)j * stride_t];
            d_buf[j] = eG_ptr[(size_t)j * eG_stride_t];
        }
    }
    for (int base = 0; base < NT; base += PF) {
        float s_nxt[PF], d_nxt[PF];
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const int c = base + PF + j;
            if (c < NT) {
                s_nxt[j] = S_ptr[(size_t)c * stride_t];
                d_nxt[j] = eG_ptr[(size_t)c * eG_stride_t];
            }
        }
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const int c = base + j;
            if (c < NT) {
                h_state = d_buf[j] * h_state + s_buf[j];
                h_ptr[(size_t)(c + 1) * stride_t] = h_state;
            }
        }
#pragma unroll
        for (int j = 0; j < PF; ++j) { s_buf[j] = s_nxt[j]; d_buf[j] = d_nxt[j]; }
    }
}

constexpr int kScanSub = 8;
constexpr int kScanSig = kScanSub * 4;

template <bool kFuseGate = false>
__launch_bounds__(128, 3)
__global__ void chunk_scan_fwd_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const __half* __restrict__ G,
    const float*  __restrict__ S_ws,
    const float*  __restrict__ eL_ws,
    float*        __restrict__ h_ws,
    unsigned*     __restrict__ h_done,
    __half*       __restrict__ o,
    int B, int T, int H, int NT, float scale,
    const __half* __restrict__ g_in,
    const __half* __restrict__ rms_w,
    float rms_eps) {
#if __CUDA_ARCH__ < 800
    if (threadIdx.x == 0) __trap();
    return;
#else
    const int BH  = B * H;
    const int tid = threadIdx.x;
    const int lane    = tid & 31;
    const int warp_id = tid >> 5;

    const int stride_t_k = H * kK;
    const int stride_t_v = H * kV;

    if ((int)blockIdx.x < BH * kScanSub) {
        const int bh    = blockIdx.x / kScanSub;
        const int slice = blockIdx.x % kScanSub;
        const int b = bh / H;
        const int h = bh % H;
        const int ki = slice * (kK / kScanSub) + warp_id;
        const size_t S_off0  = ((((size_t)b * NT * H) + h) * kK + ki) * kV + lane;
        const size_t h_off1  = ((((size_t)b * (NT + 1) + 1) * H + h) * kK + ki) * kV + lane;
        const size_t eL_off0 = ((size_t)b * NT * H + h) * kK + ki;
        const size_t hS_stride_c = (size_t)H * kK * kV;
        const size_t eL_stride_c = (size_t)H * kK;
        unsigned* hd = h_done + (size_t)bh * NT;

        constexpr int kScanBatch = 16;
        constexpr int kScanPub   = 16;
        float s_cur[kScanBatch][2], d_cur[kScanBatch];
        float s_nxt[kScanBatch][2], d_nxt[kScanBatch];
        float hacc0 = 0.f, hacc1 = 0.f;

        const int n_scan = NT - 1;
#pragma unroll
        for (int j = 0; j < kScanBatch; ++j) {
            if (j < n_scan) {
                const float* Sp = S_ws + S_off0 + (size_t)j * hS_stride_c;
                s_cur[j][0] = Sp[0];
                s_cur[j][1] = Sp[32];
                d_cur[j] = __ldg(eL_ws + eL_off0 + (size_t)j * eL_stride_c);
            }
        }
        for (int base = 0; base < n_scan; base += kScanBatch) {
#pragma unroll
            for (int j = 0; j < kScanBatch; ++j) {
                const int c = base + kScanBatch + j;
                if (c < n_scan) {
                    const float* Sp = S_ws + S_off0 + (size_t)c * hS_stride_c;
                    s_nxt[j][0] = Sp[0];
                    s_nxt[j][1] = Sp[32];
                    d_nxt[j] = __ldg(eL_ws + eL_off0 + (size_t)c * eL_stride_c);
                }
            }
            int pub_from = 0;
#pragma unroll
            for (int j = 0; j < kScanBatch; ++j) {
                const int c = base + j;
                if (c < n_scan) {
                    hacc0 = d_cur[j] * hacc0 + s_cur[j][0];
                    hacc1 = d_cur[j] * hacc1 + s_cur[j][1];
                    float* hp = h_ws + h_off1 + (size_t)c * hS_stride_c;
                    hp[0]  = hacc0;
                    hp[32] = hacc1;
                }
                if (((j + 1) & (kScanPub - 1)) == 0 || (c + 1 >= n_scan)) {
                    const int npub = j + 1 - pub_from;
                    __syncwarp();
                    const int cc = base + pub_from + lane;
                    if (lane < npub && cc < n_scan) {
                        atom_add_release_gpu_u32(&hd[cc + 1], 1u);
                    }
                    pub_from = j + 1;
                }
            }
#pragma unroll
            for (int j = 0; j < kScanBatch; ++j) {
                s_cur[j][0] = s_nxt[j][0];
                s_cur[j][1] = s_nxt[j][1];
                d_cur[j] = d_nxt[j];
            }
        }
        return;
    }

    const int idx  = (int)blockIdx.x - BH * kScanSub;
    const int i_t  = idx / BH;
    const int i_bh = idx % BH;
    const int b = i_bh / H;
    const int h = i_bh % H;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int t_base    = i_t * kBT;
    const int chunk_len = min(kBT, T - t_base);

    const int base_k = (b * T * H + h) * kK;
    const int base_v = (b * T * H + h) * kV;

    constexpr int kPadQK = 4;
    constexpr int kPad = 8;
    __shared__ __align__(16) __nv_bfloat16 sQt[kBT][kK + kPadQK];
    __shared__ __align__(16) __nv_bfloat16 sKt[kBT][kK + kPadQK];
    __shared__ __align__(16) __half        sV [kBT][kV + kPad];
    __shared__ __align__(16) __nv_bfloat16 sH [kK][kV + kPad];
    __shared__ __align__(16) __half        sA [kBT][kBT + kPad];

#pragma unroll
    for (int i = tid; i < kBT * kV / 8; i += 128) {
        const int r  = i >> 3;
        const int c8 = (i & 7) * 8;
        const int t  = t_base + r;
        const int t_clamped = t < T ? t : T - 1;
        cp_async16(&sV[r][c8],
                   v + (size_t)base_v + (size_t)t_clamped * stride_t_v + c8,
                   t < T);
    }
    cp_async_commit();

#pragma unroll
    for (int i = tid; i < kBT * kK / 8; i += 128) {
        const int r  = i >> 2;
        const int c8 = (i & 3) * 8;
        const int t  = t_base + r;
        uint4 qv8 = make_uint4(0, 0, 0, 0), kv8 = make_uint4(0, 0, 0, 0);
        uint4 gv8 = make_uint4(0, 0, 0, 0);
        if (t < T) {
            const size_t off = (size_t)base_k + (size_t)t * stride_t_k + c8;
            qv8 = *reinterpret_cast<const uint4*>(q + off);
            kv8 = *reinterpret_cast<const uint4*>(k + off);
            gv8 = *reinterpret_cast<const uint4*>(G + off);
        }
        const __half2* qh2 = reinterpret_cast<const __half2*>(&qv8);
        const __half2* kh2 = reinterpret_cast<const __half2*>(&kv8);
        const __half2* gh2 = reinterpret_cast<const __half2*>(&gv8);
        __nv_bfloat162 qb[4], kb[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const float2 qf = __half22float2(qh2[p]);
            const float2 kf = __half22float2(kh2[p]);
            const float2 gf = __half22float2(gh2[p]);
            const float eg0 = __expf(gf.x), eg1 = __expf(gf.y);
            qb[p] = __floats2bfloat162_rn(qf.x * eg0, qf.y * eg1);
            kb[p] = __floats2bfloat162_rn(kf.x / eg0, kf.y / eg1);
        }
        *reinterpret_cast<uint2*>(&sQt[r][c8])     = *reinterpret_cast<const uint2*>(&qb[0]);
        *reinterpret_cast<uint2*>(&sQt[r][c8 + 4]) = *reinterpret_cast<const uint2*>(&qb[2]);
        *reinterpret_cast<uint2*>(&sKt[r][c8])     = *reinterpret_cast<const uint2*>(&kb[0]);
        *reinterpret_cast<uint2*>(&sKt[r][c8 + 4]) = *reinterpret_cast<const uint2*>(&kb[2]);
    }
    __syncthreads();

    auto pack = [&](__nv_bfloat16 h0, __nv_bfloat16 h1) -> uint32_t {
        __nv_bfloat162 p = __halves2bfloat162(h0, h1);
        return *reinterpret_cast<const uint32_t*>(&p);
    };
    const int m_row_base = wm * 32;
    const int n_col_base = wn * 32;

    float a_acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int f = 0; f < 4; ++f) a_acc[mt][nt][f] = 0.f;

#pragma unroll
    for (int k_tile = 0; k_tile < 2; ++k_tile) {
        const int kc_base = k_tile * 16;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int m_base = m_row_base + mt * 16;
            if (n_col_base >= m_base + 16) continue;
            uint32_t A[4];
            const int r0 = m_base + gid;
            const int r1 = m_base + gid + 8;
            const int c0 = kc_base + 2 * tig;
            const int c8 = kc_base + 2 * tig + 8;
            A[0] = pack(sQt[r0][c0    ], sQt[r0][c0 + 1]);
            A[1] = pack(sQt[r1][c0    ], sQt[r1][c0 + 1]);
            A[2] = pack(sQt[r0][c8    ], sQt[r0][c8 + 1]);
            A[3] = pack(sQt[r1][c8    ], sQt[r1][c8 + 1]);
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int n_base = n_col_base + nt * 8;
                if (n_base >= m_base + 16) continue;
                uint32_t Bf[2];
                const int col_n = n_base + gid;
                Bf[0] = pack(sKt[col_n][kc_base + 2 * tig],     sKt[col_n][kc_base + 2 * tig + 1]);
                Bf[1] = pack(sKt[col_n][kc_base + 2 * tig + 8], sKt[col_n][kc_base + 2 * tig + 9]);
                mma_m16n8k16_bf16(
                    a_acc[mt][nt][0], a_acc[mt][nt][1], a_acc[mt][nt][2], a_acc[mt][nt][3],
                    A[0], A[1], A[2], A[3],
                    Bf[0], Bf[1],
                    a_acc[mt][nt][0], a_acc[mt][nt][1], a_acc[mt][nt][2], a_acc[mt][nt][3]);
            }
        }
    }

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int m_base = m_row_base + mt * 16;
            const int n_base = n_col_base + nt * 8;
            if (n_base >= m_base + 16) continue;
            const int r0 = m_base + gid;
            const int r1 = m_base + gid + 8;
            const int c0 = n_base + 2 * tig;
            const int c1 = c0 + 1;
            const float m0 = (c0 <= r0) ? a_acc[mt][nt][0] : 0.f;
            const float m1 = (c1 <= r0) ? a_acc[mt][nt][1] : 0.f;
            const float m2 = (c0 <= r1) ? a_acc[mt][nt][2] : 0.f;
            const float m3 = (c1 <= r1) ? a_acc[mt][nt][3] : 0.f;
            *reinterpret_cast<__half2*>(&sA[r0][c0]) = __floats2half2_rn(m0, m1);
            *reinterpret_cast<__half2*>(&sA[r1][c0]) = __floats2half2_rn(m2, m3);
        }
    }
    cp_async_wait_all();
    __syncthreads();

    float oi_acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int f = 0; f < 4; ++f) oi_acc[mt][nt][f] = 0.f;

    constexpr int sA_ld = kBT + kPad;
    constexpr int sV_ld = kV + kPad;
#pragma unroll
    for (int k_tile = 0; k_tile < 4; ++k_tile) {
        const int kc_base = k_tile * 16;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int m_base = m_row_base + mt * 16;
            if (kc_base > m_base) continue;
            uint32_t A[4];
            ldmatrix_x4(A[0], A[1], A[2], A[3],
                        reinterpret_cast<const __nv_bfloat16*>(&sA[m_base][kc_base]),
                        sA_ld, lane);
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int n_pair = n_col_base + p * 16;
                uint32_t Bf[4];
                ldmatrix_x4_trans(Bf[0], Bf[1], Bf[2], Bf[3],
                                  reinterpret_cast<const __nv_bfloat16*>(&sV[kc_base][n_pair]),
                                  sV_ld, lane);
                const int nt_a = 2 * p + 0;
                const int nt_b = 2 * p + 1;
                mma_m16n8k16_f16(
                    oi_acc[mt][nt_a][0], oi_acc[mt][nt_a][1], oi_acc[mt][nt_a][2], oi_acc[mt][nt_a][3],
                    A[0], A[1], A[2], A[3], Bf[0], Bf[1],
                    oi_acc[mt][nt_a][0], oi_acc[mt][nt_a][1], oi_acc[mt][nt_a][2], oi_acc[mt][nt_a][3]);
                mma_m16n8k16_f16(
                    oi_acc[mt][nt_b][0], oi_acc[mt][nt_b][1], oi_acc[mt][nt_b][2], oi_acc[mt][nt_b][3],
                    A[0], A[1], A[2], A[3], Bf[2], Bf[3],
                    oi_acc[mt][nt_b][0], oi_acc[mt][nt_b][1], oi_acc[mt][nt_b][2], oi_acc[mt][nt_b][3]);
            }
        }
    }

    float oe_acc[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int f = 0; f < 4; ++f) oe_acc[mt][nt][f] = 0.f;

    if (i_t > 0) {
        unsigned* hflag = h_done + (size_t)i_bh * NT + i_t;
        if (tid == 0) {
            unsigned ns = 32;
            while (ld_acquire_gpu_u32(hflag) < (unsigned)kScanSig) {
                __nanosleep(ns);
                if (ns < 256) ns <<= 1;
            }
            *hflag = 0u;
        }
        __syncthreads();
        const float* h_chunk = h_ws
            + (((size_t)b * (NT + 1) + i_t) * H + h) * (size_t)kK * kV;
#pragma unroll
        for (int i = tid; i < kK * kV / 8; i += 128) {
            const int r  = i >> 3;
            const int c8 = (i & 7) * 8;
            const float4 h01 = *reinterpret_cast<const float4*>(h_chunk + r * kV + c8);
            const float4 h23 = *reinterpret_cast<const float4*>(h_chunk + r * kV + c8 + 4);
            __nv_bfloat162 hb[4];
            hb[0] = __floats2bfloat162_rn(h01.x, h01.y);
            hb[1] = __floats2bfloat162_rn(h01.z, h01.w);
            hb[2] = __floats2bfloat162_rn(h23.x, h23.y);
            hb[3] = __floats2bfloat162_rn(h23.z, h23.w);
            *reinterpret_cast<uint4*>(&sH[r][c8]) = *reinterpret_cast<const uint4*>(hb);
        }
        __syncthreads();

        constexpr int sH_ld = kV + kPad;
#pragma unroll
        for (int k_tile = 0; k_tile < 2; ++k_tile) {
            const int kc_base = k_tile * 16;
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int m_base = m_row_base + mt * 16;
                uint32_t A[4];
                const int r0 = m_base + gid;
                const int r1 = m_base + gid + 8;
                const int c0 = kc_base + 2 * tig;
                const int c8 = kc_base + 2 * tig + 8;
                A[0] = pack(sQt[r0][c0    ], sQt[r0][c0 + 1]);
                A[1] = pack(sQt[r1][c0    ], sQt[r1][c0 + 1]);
                A[2] = pack(sQt[r0][c8    ], sQt[r0][c8 + 1]);
                A[3] = pack(sQt[r1][c8    ], sQt[r1][c8 + 1]);
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const int n_pair = n_col_base + p * 16;
                    uint32_t Bf[4];
                    ldmatrix_x4_trans(Bf[0], Bf[1], Bf[2], Bf[3],
                                      &sH[kc_base][n_pair], sH_ld, lane);
                    const int nt_a = 2 * p + 0;
                    const int nt_b = 2 * p + 1;
                    mma_m16n8k16_bf16(
                        oe_acc[mt][nt_a][0], oe_acc[mt][nt_a][1], oe_acc[mt][nt_a][2], oe_acc[mt][nt_a][3],
                        A[0], A[1], A[2], A[3], Bf[0], Bf[1],
                        oe_acc[mt][nt_a][0], oe_acc[mt][nt_a][1], oe_acc[mt][nt_a][2], oe_acc[mt][nt_a][3]);
                    mma_m16n8k16_bf16(
                        oe_acc[mt][nt_b][0], oe_acc[mt][nt_b][1], oe_acc[mt][nt_b][2], oe_acc[mt][nt_b][3],
                        A[0], A[1], A[2], A[3], Bf[2], Bf[3],
                        oe_acc[mt][nt_b][0], oe_acc[mt][nt_b][1], oe_acc[mt][nt_b][2], oe_acc[mt][nt_b][3]);
                }
            }
        }
    }

    if constexpr (kFuseGate) {
        __half (*sO)[kV] = reinterpret_cast<__half(*)[kV]>(&sV[0][0]);
        __half (*sG)[kV] = reinterpret_cast<__half(*)[kV]>(&sA[0][0]);
        __syncthreads();

#pragma unroll
        for (int i = tid; i < kBT * kV / 8; i += 128) {
            const int r  = i >> 3;
            const int c8 = (i & 7) * 8;
            const int t  = t_base + r;
            const int t_clamped = t < T ? t : T - 1;
            cp_async16(&sG[r][c8],
                       g_in + (size_t)base_v + (size_t)t_clamped * stride_t_v + c8,
                       t < T);
        }
        cp_async_commit();

#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int m_base = m_row_base + mt * 16;
                const int n_base = n_col_base + nt * 8;
                const int r0 = m_base + gid;
                const int r1 = m_base + gid + 8;
                const int c0 = n_base + 2 * tig;
                const int c1 = c0 + 1;
                const float o00 = scale * (oi_acc[mt][nt][0] + oe_acc[mt][nt][0]);
                const float o01 = scale * (oi_acc[mt][nt][1] + oe_acc[mt][nt][1]);
                const float o10 = scale * (oi_acc[mt][nt][2] + oe_acc[mt][nt][2]);
                const float o11 = scale * (oi_acc[mt][nt][3] + oe_acc[mt][nt][3]);
                sO[r0][c0] = __float2half(o00);
                sO[r0][c1] = __float2half(o01);
                sO[r1][c0] = __float2half(o10);
                sO[r1][c1] = __float2half(o11);
            }
        }

        __half2 w_local = *reinterpret_cast<const __half2*>(&rms_w[2 * lane]);
        const float w0 = __low2float(w_local);
        const float w1 = __high2float(w_local);
        constexpr float kInvV = 1.0f / static_cast<float>(kV);

        cp_async_wait_all();
        __syncthreads();

        const int row_base = warp_id * 16;
#pragma unroll 1
        for (int rr = 0; rr < 16; ++rr) {
            const int r = row_base + rr;
            if (r >= chunk_len) break;

            __half2 o2 = *reinterpret_cast<const __half2*>(&sO[r][2 * lane]);
            __half2 g2 = *reinterpret_cast<const __half2*>(&sG[r][2 * lane]);
            const float o0 = __low2float(o2);
            const float o1 = __high2float(o2);
            const float g0 = __low2float(g2);
            const float g1 = __high2float(g2);

            float ss = o0 * o0 + o1 * o1;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                ss += __shfl_xor_sync(0xffffffff, ss, off);
            }
            const float rstd = rsqrtf(ss * kInvV + rms_eps);

            const float silu_g0 = g0 / (1.0f + __expf(-g0));
            const float silu_g1 = g1 / (1.0f + __expf(-g1));
            const float y0 = (o0 * rstd) * w0 * silu_g0;
            const float y1 = (o1 * rstd) * w1 * silu_g1;

            const int t = t_base + r;
            __half2 y2 = __halves2half2(__float2half_rn(y0), __float2half_rn(y1));
            *reinterpret_cast<__half2*>(&o[base_v + t * stride_t_v + 2 * lane]) = y2;
        }
    } else {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int m_base = m_row_base + mt * 16;
                const int n_base = n_col_base + nt * 8;
                const int r0 = m_base + gid;
                const int r1 = m_base + gid + 8;
                const int c0 = n_base + 2 * tig;
                const float o00 = scale * (oi_acc[mt][nt][0] + oe_acc[mt][nt][0]);
                const float o01 = scale * (oi_acc[mt][nt][1] + oe_acc[mt][nt][1]);
                const float o10 = scale * (oi_acc[mt][nt][2] + oe_acc[mt][nt][2]);
                const float o11 = scale * (oi_acc[mt][nt][3] + oe_acc[mt][nt][3]);
                const int t0 = t_base + r0;
                const int t1 = t_base + r1;
                if (t0 < T) {
                    *reinterpret_cast<__half2*>(&o[base_v + t0 * stride_t_v + c0]) =
                        __floats2half2_rn(o00, o01);
                }
                if (t1 < T) {
                    *reinterpret_cast<__half2*>(&o[base_v + t1 * stride_t_v + c0]) =
                        __floats2half2_rn(o10, o11);
                }
            }
        }
    }
#endif
}

inline size_t chunk_workspace_bytes(int B, int T, int H) {
    const int NT = (T + kBT - 1) / kBT;
    const size_t G_bytes  = size_t(B) * T * H * kK * sizeof(float);
    const size_t h_bytes  = size_t(B) * (NT + 1) * H * kK * kV * sizeof(float);
    const size_t S_bytes  = size_t(B) * NT * H * kK * kV * sizeof(float);
    const size_t eL_bytes = size_t(B) * NT * H * kK * sizeof(float);
    const size_t fl_bytes = (size_t(B) * H * NT + 16) * sizeof(float);
    return G_bytes + h_bytes + S_bytes + eL_bytes + fl_bytes;
}

inline bool chunk_path_supported_on_current_device() {
    static int s_cached = -1;
    if (s_cached >= 0) return s_cached != 0;

    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        s_cached = 0;
        return false;
    }
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    s_cached = (major >= 8) ? 1 : 0;
    return s_cached != 0;
}

inline int scan_fwd_resident_blocks(bool fuse_gate) {
    static int s_cached[2] = {-1, -1};
    int& s = s_cached[fuse_gate ? 1 : 0];
    if (s < 0) {
        int dev = 0, sm_count = 0, per_sm = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { s = 0; return s; }
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm,
            fuse_gate ? chunk_scan_fwd_kernel<true> : chunk_scan_fwd_kernel<false>,
            128, 0);
        s = (err == cudaSuccess) ? per_sm * sm_count : 0;
    }
    return s;
}

inline bool fused_path_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("GLA_FUSED");
        v = (e != nullptr && e[0] == '0') ? 0 : 1;
    }
    return v != 0;
}

inline void launch_chunk_gla(
    const __half* q, const __half* k, const __half* v, const __half* gk,
    __half* o,
    float* workspace_eG,
    float* workspace_h,
    int B, int T, int H, float scale,
    cudaStream_t stream,
    float* workspace_S,
    float* workspace_eL,
    float gk_normalizer = 0.f,
    const __half* g_in = nullptr,
    const __half* rms_w = nullptr,
    float rms_eps = 1e-6f,
    float* workspace_fl = nullptr
) {
    const int NT = (T + kBT - 1) / kBT;
    const float inv_norm_arg = (gk_normalizer > 0.f) ? (1.0f / gk_normalizer) : 0.f;
    const bool fuse_gate = (g_in != nullptr && rms_w != nullptr);

    if (workspace_fl != nullptr && fused_path_enabled()) {
        const int BH = B * H;
        const int n_scan_blocks = BH * kScanSub;
        const int slots = scan_fwd_resident_blocks(fuse_gate);
        const bool wave_aligned = (slots > 0) && (BH * NT % slots == 0);
        if (NT >= 6 && !wave_aligned && slots >= n_scan_blocks + 1) {
            __half* workspace_G2 = reinterpret_cast<__half*>(workspace_eG);
            {
                dim3 grid(NT, BH, 1);
                dim3 block(kK, 4);
                chunk_compute_S_kernel<<<grid, block, 0, stream>>>(
                    k, v, gk, workspace_G2, workspace_S, workspace_eL,
                    T, H, NT, inv_norm_arg);
            }
            {
                unsigned* h_done = reinterpret_cast<unsigned*>(workspace_fl);
                dim3 grid(n_scan_blocks + NT * BH);
                dim3 block(128);
                if (fuse_gate) {
                    chunk_scan_fwd_kernel<true><<<grid, block, 0, stream>>>(
                        q, k, v, workspace_G2, workspace_S, workspace_eL,
                        workspace_h, h_done, o, B, T, H, NT, scale,
                        g_in, rms_w, rms_eps);
                } else {
                    chunk_scan_fwd_kernel<false><<<grid, block, 0, stream>>>(
                        q, k, v, workspace_G2, workspace_S, workspace_eL,
                        workspace_h, h_done, o, B, T, H, NT, scale,
                        nullptr, nullptr, rms_eps);
                }
            }
            return;
        }
    }

    constexpr int kInlineScanMaxNT = 16;
    const bool inline_scan = (NT <= kInlineScanMaxNT);

    __half* workspace_G = reinterpret_cast<__half*>(workspace_eG);
    {
        dim3 grid(NT, B * H, 1);
        dim3 block(kK, 4);
        chunk_compute_S_kernel<<<grid, block, 0, stream>>>(
            k, v, gk, workspace_G, workspace_S, workspace_eL,
            T, H, NT, inv_norm_arg);
    }

    if (!inline_scan) {
        dim3 grid(B * H, kK);
        dim3 block(kV);
        chunk_scan_h_seq_kernel<<<grid, block, 0, stream>>>(
            workspace_S, workspace_eL, workspace_h, H, NT);
    }

    {
        dim3 grid(NT, B * H);
        dim3 block(128);
        if (fuse_gate) {
            if (inline_scan) {
                chunk_fwd_o_kernel<true, true><<<grid, block, 0, stream>>>(
                    q, k, v, workspace_G, workspace_h, o, T, H, NT, scale,
                    g_in, rms_w, rms_eps, workspace_S, workspace_eL);
            } else {
                chunk_fwd_o_kernel<true, false><<<grid, block, 0, stream>>>(
                    q, k, v, workspace_G, workspace_h, o, T, H, NT, scale,
                    g_in, rms_w, rms_eps);
            }
        } else {
            if (inline_scan) {
                chunk_fwd_o_kernel<false, true><<<grid, block, 0, stream>>>(
                    q, k, v, workspace_G, workspace_h, o, T, H, NT, scale,
                    nullptr, nullptr, 1e-6f, workspace_S, workspace_eL);
            } else {
                chunk_fwd_o_kernel<false, false><<<grid, block, 0, stream>>>(
                    q, k, v, workspace_G, workspace_h, o, T, H, NT, scale);
            }
        }
    }
}

}
