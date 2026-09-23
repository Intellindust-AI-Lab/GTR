#!/usr/bin/env python3
"""
Vendored verbatim from HA4DETR — https://github.com/linfeng93/HA4DETR.git
(Apache License 2.0; see HA4DETR_LICENSE in this directory). GPU batched Hungarian
solver; imported by matcher.py to replace the CPU scipy LSAP loop. Importing this
module JIT-compiles the CUDA extension (torch.utils.cpp_extension.load_inline).

GPU Support
-----------
* Validated on CUDA 12.2 / driver ≥ 535 with a single NVIDIA RTX 4090
* Requires PyTorch ≥ 2.1, SciPy ≥ 1.10, a C++17 tool-chain and NVCC

Run
---
$ python hungarian_gpu_batch.py        # builds extension, runs demo & check
"""

import os
import time
import torch
from torch.utils.cpp_extension import load_inline
from scipy.optimize import linear_sum_assignment

# ────────────────────────── C++ forward declaration ──────────────────────────
CPP_STUB = r"""
#include <torch/extension.h>
void hungarian_launcher(torch::Tensor cost, torch::Tensor ncols, torch::Tensor assignment);
"""

# ───────────────────────────── CUDA implementation ───────────────────────────
CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define _MAX_ROWS 300
#define _MAX_COLS 300
#define _BLOCK_SIZE 512
#define _WARP 32

constexpr int MAX_ROWS   = _MAX_ROWS;       // workers (≥ tasks)
constexpr int MAX_COLS   = _MAX_COLS;       // tasks, INF padded
constexpr int WARP       = _WARP;
constexpr float INF      = 1e20f;

/* ------------------------------- Utils ----------------------------------- */
template<int NT>
__device__ __forceinline__
void warp_min_reduce(float &val, int &idx)
{
    #pragma unroll
    for (int offset = NT / 2; offset > 0; offset >>= 1)
    {
        float val_other = __shfl_down_sync(0xffffffff, val , offset);
        int   idx_other = __shfl_down_sync(0xffffffff, idx , offset);
        if (val_other < val) {
            val = val_other;
            idx = idx_other;
        }
    }
}

/* ------------------------------------------------------------------------- *
 * One block solves one 300x300 assignment problem.
 * 512 threads cooperate; loops over rows are parallelised across threads,
 * while thread 0 performs the augmenting-path book-keeping.
 * We parallelise: ⭐ init          ⭐ Δ min-search     ⭐ u/v/minv update
 *                 ⭐ output write  ⭐ batch-grid over-subscription
 * ------------------------------------------------------------------------- */
template<int BLOCK_SIZE>
__global__ void hungarian_kernel(const float * __restrict__ cost,
                                 const int   * __restrict__ ncols,
                                 int         * __restrict__ assignment,
                                 int B,                         // batch size
                                 int cost_stride,               // 300 x 300
                                 int asgn_stride)               // 300 x 2
{
    int globalBid = blockIdx.x;
    const int tid = threadIdx.x;

    while (globalBid < B) {
        /* Pointers ---------------------------------------------------------------- */
        const float *costB = cost + globalBid * cost_stride;
        const int    cols  = ncols[globalBid];
        int         *asgnB = assignment + globalBid * asgn_stride;

        /* Shared Hungarian state -------------------------------------------------- */
        __shared__ float u[MAX_COLS + 1];               // task potentials
        __shared__ float v[MAX_ROWS + 1];               // worker potentials
        __shared__ int   p[MAX_ROWS + 1];               // matching: p[worker]=task
        __shared__ int   way[MAX_ROWS + 1];
        __shared__ float minv[MAX_ROWS + 1];
        __shared__ bool  used[MAX_ROWS + 1];
        __shared__ int   j0;
        __shared__ float delta_s;                       // best Δ in this iteration
        __shared__ int   j1_s;                          // argmin of Δ
        __shared__ bool  path_found;

        /* Init potentials (rows == cols == 300) ----------------------------------- */
        for (int k = tid; k <= MAX_ROWS; k += BLOCK_SIZE) {
            v[k] = u[k] = 0.0f;     // v[k] = 0.0f; if (k <= MAX_COLS) u[k] = 0.0f;
            p[k] = 0;
        }
        __syncthreads();

        /* Hungarian main loop over each column ------------------------------------ */
        for (int i = 1; i <= cols; ++i) {
            if (tid == 0) {
                p[0] = i;
                j0 = 0;
                path_found = false;
            }
            __syncthreads();

            /* Reset per-task buffers ---------------------------------------------- */
            for (int j = tid; j <= MAX_ROWS; j += BLOCK_SIZE) {
                minv[j] = INF;
                used[j] = false;
            }
            __syncthreads();

            /* grow alternating tree until an unmatched worker is found ------------ */
            while (!path_found) {
                /* Mark current row as used ---------------------------------------- */
                if (tid == 0) used[j0] = true;
                __syncthreads();

                /* Δ-scan: each thread processes stride-rows ----------------------- */
                int i0  = p[j0];                // current task index
                float best_val = INF;
                int best_j = 0;
                for (int j = tid + 1; j <= MAX_ROWS; j += BLOCK_SIZE) {
                    if (used[j]) continue;
                    float cur = costB[(j - 1) * MAX_COLS + (i0 - 1)] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < best_val) {
                        best_val = minv[j];
                        best_j = j;
                    }
                }

                /* Block-wide reduction to find global min Δ ----------------------- */
                const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP;
                warp_min_reduce<WARP>(best_val, best_j);
                if (BLOCK_SIZE > WARP) {
                    __shared__ float warp_min_val[WARPS_PER_BLOCK];
                    __shared__ int   warp_min_j[WARPS_PER_BLOCK];

                    if ((tid & (WARP - 1)) == 0) {
                        warp_min_val[tid / WARP] = best_val;
                        warp_min_j[tid / WARP] = best_j;
                    }
                    __syncthreads();

                    if (tid < WARPS_PER_BLOCK) {
                        best_val = warp_min_val[tid];
                        best_j = warp_min_j[tid];
                    }
                    else best_val = INF;

                    /* Final reduction --------------------------------------------- */
                    if (tid == 0) {
                        for (int k = 1; k < WARPS_PER_BLOCK; ++k)
                            if (warp_min_val[k] < best_val) {
                                best_val = warp_min_val[k];
                                best_j = warp_min_j[k];
                            }
                        delta_s = best_val;
                        j1_s = best_j;
                    }
                }
                else if (tid == 0) {
                    delta_s = best_val;
                    j1_s = best_j;
                }
                __syncthreads();

                /* Parallel update of u / v / minv --------------------------------- */
                for (int j = tid; j <= MAX_ROWS; j += BLOCK_SIZE) {
                    if (used[j]) {
                        u[p[j]] += delta_s;
                        v[j] -= delta_s;
                    } else minv[j] -= delta_s;
                }
                __syncthreads();

                if (tid == 0) {
                    j0 = j1_s;
                    if (p[j0] == 0) path_found = true;
                }
                __syncthreads();
            }

            /* Augment along alternating path -------------------------------------- */
            if (tid == 0) {
                while (j0 != 0) {
                    int j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                }
            }
            __syncthreads();
        }

        /* Output (row, col) pairs in parellel ------------------------------------- */
        for (int row = tid + 1; row <= MAX_ROWS; row += BLOCK_SIZE) {
            int task = p[row];
            asgnB[(row - 1) * 2] = row - 1;             // row index
            asgnB[(row - 1) * 2 + 1] = task - 1;        // col index
        }

        /* Move to next problem for oversubscription ------------------------------- */
        globalBid += gridDim.x;   // grid-stride over batch
        __syncthreads();
    }
}

/* ------------------------------- Launcher --------------------------------- */
void hungarian_launcher(torch::Tensor cost,
                        torch::Tensor ncols,
                        torch::Tensor assignment)
{
    TORCH_CHECK(cost.is_cuda(),                      "cost must be on CUDA");
    TORCH_CHECK(ncols.is_cuda(),                     "ncols must be on CUDA");
    TORCH_CHECK(assignment.is_cuda(),                "assignment must be on CUDA");
    TORCH_CHECK(cost.dtype() == torch::kFloat32,     "cost must be float32");
    TORCH_CHECK(ncols.dtype() == torch::kInt32,      "ncols must be int32");
    TORCH_CHECK(assignment.dtype() == torch::kInt32, "assignment must be int32");

    const int B = cost.size(0);
    const int R = cost.size(1);
    const int C = cost.size(2);

    const int cost_stride = R * C;
    const int asgn_stride = C * 2;

    TORCH_CHECK(ncols.size(0) == B, "ncols len must equal batch");

    int smCount;
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);

    /* Oversubscribe: at least 4xSM block, no more than batch size (B) ------------- */
    int blocks  = min(B, smCount * 4);      // work when B > 512, RTX 4090: 128 SM
    int threads = _BLOCK_SIZE;

    hungarian_kernel<_BLOCK_SIZE><<<blocks, threads>>>(
        cost.data_ptr<float>(),
        ncols.data_ptr<int>(),
        assignment.data_ptr<int>(),
        B,
        cost_stride,
        asgn_stride);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
"""

# ───────────────────── Build / load inline extension ─────────────────────────
# os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")  # RTX 4090
ext_mod = load_inline(
    name="hungarian_gpu_batch_ext",
    cpp_sources=[CPP_STUB],
    cuda_sources=[CUDA_SRC],
    functions=["hungarian_launcher"],
    extra_cflags=["-std=c++17", "-O3"],
    extra_cuda_cflags=["-std=c++17", "-O3", "--use_fast_math"], 
    verbose=False,
)
print(ext_mod.__file__)

# ───────────────────────────── Python wrapper  ───────────────────────────────
def hungarian_gpu(cost: torch.Tensor, ncols: torch.Tensor) -> torch.Tensor:
    """
    input:
        cost:  (B, _MAX_ROWS, _MAX_COLS)   float32 CUDA tensor
        ncols: (B,)                        int32 CUDA tensor
    return: 
        out:   (B, _MAX_ROWS, 2)           int32 CUDA tensor, -1 padded
    """
    B, _, C = cost.shape
    out = torch.empty((B, C, 2), dtype=torch.int32, device=cost.device)
    ext_mod.hungarian_launcher(cost, ncols, out)
    return out      # already row-sorted

def timeit(fn, *args, sync=True, loops=1):
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(loops):
        res = fn(*args)
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / loops, res

# ─────────────────────────────────── Demo (randomly padding) ────────────────────────────────────
# Function: hungarian_gpu
# - Input: A 3D float batched cost tensor of shape [B, _MAX_ROWS, _MAX_COLS], 
#          where each [_MAX_ROWS, Ni] cost tensor (Ni ∈ [1, _MAX_COLS]) is padded
#          with inf to width _MAX_COLS before stacking.
#
#          A 1D int32 tensor of shape [B] indicating the real task numbers of cost tensors
#          (i.e., {Ni}, i ∈ [0, B - 1])
#
# - Output: A 3D integer tensor of shape [B, _MAX_ROWS, 2], where each (Ni, 2) 
#           Hungarian assignment result (row i → column j) is padded with -1
#           to (_MAX_ROWS, 2) before stacking.
#
# - Performance Expectations: Maximize GPU parallelism to efficiently handle variable-sized
#                             cost matrices, achieving better speedup over 
#                             scipy.optimize.linear_sum_assignment executed sequentially on the CPU.

def demo():
    assert torch.cuda.is_available()
    
    # Test settings
    device = "cuda"
    batch_size = 16
    n_rows, n_cols = 300, 300
    loops = 11
    
    gpu_t_total, cpu_t_total, speedup_total = 0, 0, 0
    for i in range(loops):

        # Generate the input cost tensor
        # (During DETR's training, the Hungarian algorithm inputs are stored on GPU; our demo follows this setup.)
        torch.manual_seed(i)
        Ns = torch.randint(1, 301, (batch_size,), device=device, dtype=torch.int32)
        cost = torch.full((batch_size, n_rows, n_cols), float('inf'), device=device, dtype=torch.float32)
        for b in range(cost.size(0)):
            Ni = Ns[b].item()
            cost[b, :, :Ni] = torch.rand((n_rows, Ni), device=device, dtype=torch.float32)
        cost = cost.contiguous()

        # GPU implementation
        gpu_t, gpu_out = timeit(hungarian_gpu, cost, Ns)

        # CPU reference
        cpu_out = torch.full_like(gpu_out, int('-1'), device="cpu")
        cpu_out[:, :, 0] = torch.arange(n_cols).to(torch.int32)
        t0 = time.perf_counter()
        for b in range(cost.size(0)):
            Ni = Ns[b].item()
            r, c = linear_sum_assignment(cost[b][:, :Ni].cpu())
            cpu_out[b, r, 1] = torch.tensor(c).to(torch.int32)
        cpu_t = time.perf_counter() - t0

        # Checking: Check if the GPU and CPU results match.
        # (Due to non-uniqueness in the Hungarian algorithm, we consider results consistent if the final costs differ by less than 1e-4.)
        torch.testing.assert_close(gpu_out.cpu(), cpu_out, rtol=0, atol=0)
        try:
            torch.testing.assert_close(gpu_out.cpu(), cpu_out, rtol=0, atol=0)
        except AssertionError:
            indices_gpu = [(_gpu_out[:, 0][_valid[:, 1]], _gpu_out[:, 1][_valid[:, 1]]) for _gpu_out, _valid in zip(gpu_out, gpu_out > -1)]
            indices_cpu = [(_cpu_out[:, 0][_valid[:, 1]], _cpu_out[:, 1][_valid[:, 1]]) for _cpu_out, _valid in zip(cpu_out, cpu_out > -1)]
            err = 0
            for i in range(batch_size):
                c_gpu = cost[i][indices_gpu[i][0].cpu(), indices_gpu[i][1].cpu()].sum().item()
                c_cpu = cost[i][indices_cpu[i][0], indices_cpu[i][1]].sum().item()
                err += abs(c_gpu - c_cpu)
            err /= batch_size
            if err < 1e-4:
                print(f"WARNING: Alignments from CPU and GPU mismatch, but their real costs are equal [abs(ERROR) = {err}].")
            else:
                raise AssertionError("Something wrong in GPU implemetation.")

        print(f"Mean valid ncols in cost matrix: {Ns.sum().item() / batch_size:5.2f}")
        print(f"GPU runtime      : {gpu_t*1e3:7.2f} ms")
        print(f"SciPy CPU runtime: {cpu_t*1e3:7.2f} ms")
        print(f"Speed-up         : {cpu_t / gpu_t:7.2f} x\n")

        # Skip Loop 0 due to GPU implementation warm-up
        gpu_t_total += gpu_t if i > 0 else 0
        cpu_t_total += cpu_t if i > 0 else 0
        speedup_total += cpu_t / gpu_t if i > 0 else 0
        time.sleep(1)

    if loops > 1:
        print(f"Average of the last {loops - 1} Loops")
        print(f"GPU runtime      : {(gpu_t_total / (loops - 1))*1e3:7.2f} ms")
        print(f"SciPy CPU runtime: {(cpu_t_total / (loops - 1))*1e3:7.2f} ms")
        print(f"Speed-up         : {(speedup_total / (loops - 1)):7.2f} x\n")


if __name__ == "__main__":
    demo()