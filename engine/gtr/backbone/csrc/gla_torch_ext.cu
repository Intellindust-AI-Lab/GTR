// GTR: Gated Token Recurrence for Efficient Dense Prediction
// Copyright (c) 2026 The GTR Authors. All Rights Reserved.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include "gla_chunk.cuh"
#include "gla_norm_gate.cuh"

namespace {

inline void check_input(const torch::Tensor& t, const char* name, int expect_last) {
    TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, " must be fp16");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 4, name, " must be [B,T,H,D]");
    TORCH_CHECK(t.size(3) == expect_last, name, " last dim must be ", expect_last,
                ", got ", t.size(3));
}

int64_t workspace_floats(int64_t B, int64_t T, int64_t H) {
    const int64_t NT = (T + gla_chunk::kBT - 1) / gla_chunk::kBT;
    const int64_t K = gla_chunk::kK;
    const int64_t V = gla_chunk::kV;
    return B * T * H * K
         + B * (NT + 1) * H * K * V
         + B * NT * H * K * V
         + B * NT * H * K
         + B * H * NT + 16;
}

torch::Tensor make_workspace(int64_t B, int64_t T, int64_t H, c10::Device device) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    return torch::zeros({workspace_floats(B, T, H)}, opts);
}

void chunk_gla_run(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gk,
    double scale,
    torch::Tensor& ws,
    torch::Tensor& o,
    double gk_normalizer = 0.0) {
    check_input(q,  "q",  gla_chunk::kK);
    check_input(k,  "k",  gla_chunk::kK);
    check_input(v,  "v",  gla_chunk::kV);
    check_input(gk, "gk", gla_chunk::kK);
    const int64_t B = q.size(0), T = q.size(1), H = q.size(2);
    TORCH_CHECK(k.sizes() == q.sizes(),  "k must match q shape");
    TORCH_CHECK(v.size(0) == B && v.size(1) == T && v.size(2) == H,
                "v must have [B,T,H,V] matching q's leading dims");
    TORCH_CHECK(gk.sizes() == q.sizes(), "gk must match q shape");
    TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kFloat16 && o.is_contiguous(),
                "o must be CUDA fp16 contiguous");
    TORCH_CHECK(o.dim() == 4 && o.size(0) == B && o.size(1) == T
                && o.size(2) == H && o.size(3) == gla_chunk::kV,
                "o must be [B,T,H,V=64]");
    const int64_t need_ws = workspace_floats(B, T, H);
    TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == torch::kFloat32 && ws.is_contiguous()
                && ws.numel() >= need_ws,
                "ws must be CUDA fp32 contiguous of size >= ", need_ws);
    const int64_t NT = (T + gla_chunk::kBT - 1) / gla_chunk::kBT;
    const int64_t K = gla_chunk::kK;
    const int64_t V = gla_chunk::kV;
    float* ws_ptr = ws.data_ptr<float>();
    float* dG  = ws_ptr;
    float* dh  = dG + B * T * H * K;
    float* dS  = dh + B * (NT + 1) * H * K * V;
    float* deL = dS + B * NT * H * K * V;
    float* dFl = deL + B * NT * H * K;
    auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
    const c10::cuda::CUDAStreamGuard guard(stream);
    gla_chunk::launch_chunk_gla(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gk.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
        dG, dh,
        static_cast<int>(B), static_cast<int>(T), static_cast<int>(H),
        static_cast<float>(scale),
        stream.stream(),
        dS, deL, static_cast<float>(gk_normalizer),
        nullptr, nullptr, 1e-6f, dFl);
}

void chunk_gla_run_gated(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gk,
    const torch::Tensor& g,
    const torch::Tensor& rms_w,
    double scale,
    double eps,
    torch::Tensor& ws,
    torch::Tensor& y,
    double gk_normalizer = 0.0) {
    check_input(q,  "q",  gla_chunk::kK);
    check_input(k,  "k",  gla_chunk::kK);
    check_input(v,  "v",  gla_chunk::kV);
    check_input(gk, "gk", gla_chunk::kK);
    check_input(g,  "g",  gla_chunk::kV);
    const int64_t B = q.size(0), T = q.size(1), H = q.size(2);
    TORCH_CHECK(k.sizes() == q.sizes(),  "k must match q shape");
    TORCH_CHECK(v.sizes() == g.sizes(),  "g must match v shape [B,T,H,V]");
    TORCH_CHECK(v.size(0) == B && v.size(1) == T && v.size(2) == H,
                "v must have [B,T,H,V] matching q's leading dims");
    TORCH_CHECK(gk.sizes() == q.sizes(), "gk must match q shape");
    TORCH_CHECK(rms_w.is_cuda() && rms_w.scalar_type() == torch::kFloat16
                && rms_w.is_contiguous() && rms_w.dim() == 1
                && rms_w.size(0) == gla_chunk::kV,
                "rms_w must be [V=", gla_chunk::kV, "] fp16 contiguous CUDA");
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat16 && y.is_contiguous(),
                "y must be CUDA fp16 contiguous");
    TORCH_CHECK(y.dim() == 4 && y.size(0) == B && y.size(1) == T
                && y.size(2) == H && y.size(3) == gla_chunk::kV,
                "y must be [B,T,H,V=", gla_chunk::kV, "]");
    const int64_t need_ws = workspace_floats(B, T, H);
    TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == torch::kFloat32 && ws.is_contiguous()
                && ws.numel() >= need_ws,
                "ws must be CUDA fp32 contiguous of size >= ", need_ws);
    const int64_t NT = (T + gla_chunk::kBT - 1) / gla_chunk::kBT;
    const int64_t K = gla_chunk::kK;
    const int64_t V = gla_chunk::kV;
    float* ws_ptr = ws.data_ptr<float>();
    float* dG  = ws_ptr;
    float* dh  = dG + B * T * H * K;
    float* dS  = dh + B * (NT + 1) * H * K * V;
    float* deL = dS + B * NT * H * K * V;
    float* dFl = deL + B * NT * H * K;
    auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
    const c10::cuda::CUDAStreamGuard guard(stream);
    gla_chunk::launch_chunk_gla(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gk.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        dG, dh,
        static_cast<int>(B), static_cast<int>(T), static_cast<int>(H),
        static_cast<float>(scale),
        stream.stream(),
        dS, deL,
        static_cast<float>(gk_normalizer),
        reinterpret_cast<const __half*>(g.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(rms_w.data_ptr<at::Half>()),
        static_cast<float>(eps), dFl);
}

torch::Tensor chunk_gla_impl(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gk,
    double scale,
    torch::Tensor& ws) {
    check_input(q,  "q",  gla_chunk::kK);
    check_input(k,  "k",  gla_chunk::kK);
    check_input(v,  "v",  gla_chunk::kV);
    check_input(gk, "gk", gla_chunk::kK);

    const int64_t B = q.size(0);
    const int64_t T = q.size(1);
    const int64_t H = q.size(2);
    TORCH_CHECK(k.sizes() == q.sizes(),  "k must match q shape");
    TORCH_CHECK(v.size(0) == B && v.size(1) == T && v.size(2) == H,
                "v must have [B,T,H,V] matching q's leading dims");
    TORCH_CHECK(gk.sizes() == q.sizes(), "gk must match q shape");

    const int64_t need_ws = workspace_floats(B, T, H);
    TORCH_CHECK(ws.is_cuda() && ws.scalar_type() == torch::kFloat32 && ws.is_contiguous(),
                "ws must be CUDA fp32 contiguous");
    TORCH_CHECK(ws.numel() >= need_ws, "ws too small: need ", need_ws,
                " floats, got ", ws.numel());

    auto opts = q.options();
    auto o = torch::empty({B, T, H, gla_chunk::kV}, opts);

    const int64_t NT = (T + gla_chunk::kBT - 1) / gla_chunk::kBT;
    const int64_t K = gla_chunk::kK;
    const int64_t V = gla_chunk::kV;

    float* ws_ptr = ws.data_ptr<float>();
    float* dG  = ws_ptr;
    float* dh  = dG + B * T * H * K;
    float* dS  = dh + B * (NT + 1) * H * K * V;
    float* deL = dS + B * NT * H * K * V;
    float* dFl = deL + B * NT * H * K;

    auto stream = at::cuda::getCurrentCUDAStream(q.device().index());
    const c10::cuda::CUDAStreamGuard guard(stream);

    gla_chunk::launch_chunk_gla(
        reinterpret_cast<const __half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(gk.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(o.data_ptr<at::Half>()),
        dG, dh,
        static_cast<int>(B), static_cast<int>(T), static_cast<int>(H),
        static_cast<float>(scale),
        stream.stream(),
        dS, deL,
        0.f,
        nullptr, nullptr, 1e-6f, dFl);
    return o;
}

torch::Tensor chunk_gla(const torch::Tensor& q, const torch::Tensor& k,
                       const torch::Tensor& v, const torch::Tensor& gk,
                       double scale) {
    auto ws = make_workspace(q.size(0), q.size(1), q.size(2), q.device());
    return chunk_gla_impl(q, k, v, gk, scale, ws);
}

torch::Tensor chunk_gla_with_ws(const torch::Tensor& q, const torch::Tensor& k,
                               const torch::Tensor& v, const torch::Tensor& gk,
                               double scale, torch::Tensor ws) {
    return chunk_gla_impl(q, k, v, gk, scale, ws);
}

void rmsnorm_gated_run(
    const torch::Tensor& x,
    const torch::Tensor& g,
    const torch::Tensor& w,
    torch::Tensor& y,
    double eps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16 && x.is_contiguous(),
                "x must be CUDA fp16 contiguous");
    TORCH_CHECK(g.is_cuda() && g.scalar_type() == torch::kFloat16 && g.is_contiguous(),
                "g must be CUDA fp16 contiguous");
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat16 && w.is_contiguous(),
                "w must be CUDA fp16 contiguous");
    TORCH_CHECK(y.is_cuda() && y.scalar_type() == torch::kFloat16 && y.is_contiguous(),
                "y must be CUDA fp16 contiguous");
    TORCH_CHECK(x.sizes() == g.sizes() && x.sizes() == y.sizes(),
                "x, g, y must have identical shapes");
    TORCH_CHECK(x.size(-1) == gla_norm_gate::kV,
                "last dim must be ", gla_norm_gate::kV, ", got ", x.size(-1));
    TORCH_CHECK(w.dim() == 1 && w.size(0) == gla_norm_gate::kV,
                "w must be [", gla_norm_gate::kV, "] fp16");

    const int64_t N = x.numel() / gla_norm_gate::kV;
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
    const c10::cuda::CUDAStreamGuard guard(stream);
    gla_norm_gate::launch_rmsnorm_gated(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(g.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        static_cast<int>(N), static_cast<float>(eps),
        stream.stream());
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_gla", &chunk_gla,
          "Chunk-based GLA forward (allocates workspace internally)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("gk"), py::arg("scale"));
    m.def("chunk_gla_with_ws", &chunk_gla_with_ws,
          "Chunk-based GLA forward (caller-provided workspace, CUDA-graph friendly)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("gk"), py::arg("scale"),
          py::arg("ws"));
    m.def("chunk_gla_run", &chunk_gla_run,
          "Chunk-based GLA forward writing into caller's output tensor (CUDA-graph friendly)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("gk"), py::arg("scale"),
          py::arg("ws"), py::arg("o"), py::arg("gk_normalizer") = 0.0);
    m.def("workspace_floats", &workspace_floats,
          "Number of fp32 floats required in workspace for given (B,T,H)");
    m.def("make_workspace", &make_workspace,
          "Allocate a workspace tensor of correct size on the given device");
    m.def("rmsnorm_gated", &rmsnorm_gated_run,
          "Fused RMSNorm + swish gate (D=64 hardcoded; in-place into y)",
          py::arg("x"), py::arg("g"), py::arg("w"), py::arg("y"), py::arg("eps"));
    m.def("chunk_gla_run_gated", &chunk_gla_run_gated,
          "Fused chunk_gla + RMSNorm + swish gate. Writes final y in one launch "
          "(skips the separate rmsnorm pass).",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("gk"),
          py::arg("g"), py::arg("rms_w"),
          py::arg("scale"), py::arg("eps"),
          py::arg("ws"), py::arg("y"),
          py::arg("gk_normalizer") = 0.0);
    m.attr("kK")  = py::int_(gla_chunk::kK);
    m.attr("kV")  = py::int_(gla_chunk::kV);
    m.attr("kBT") = py::int_(gla_chunk::kBT);
}
