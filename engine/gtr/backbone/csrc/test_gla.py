"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Correctness gate for the hand-written chunk_gla CUDA operator.

Reference is fla (Triton). The kernels are fp16 with a different tiling than fla,
so outputs are NOT bit-exact; we assert high cosine similarity and a bounded
relative error instead.

Run:  bash build.sh && python test_gla.py

Covers the full operator surface exposed by gla_torch_ext.cu:
  * chunk_gla_run        — chunk GLA forward (q,k,v,gk -> o)
  * rmsnorm_gated        — fused RMSNorm + swish gate (D=64)
  * chunk_gla_run_gated  — chunk GLA + RMSNorm + swish gate fused in one launch
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from fla.ops.gla import chunk_gla as fla_chunk_gla
from fla.modules import FusedRMSNormGated

import gla_torch_ext as ext  # built by build.sh

DEVICE = "cuda"
DTYPE = torch.float16
K = ext.kK   # 32
V = ext.kV   # 64
EPS = 1e-5

# T values: production seq len (2917), non-multiples of kBT=64, and the long
# end the parallel scan supports (kNTPadMax=128 -> T<=8192).
T_CASES = [65, 100, 127, 128, 256, 512, 1024, 2048, 2917, 4096, 8192]
H_DEFAULT = 6


def _metrics(ref: torch.Tensor, got: torch.Tensor):
    ref = ref.flatten().float()
    got = got.flatten().float()
    cos = F.cosine_similarity(ref, got, dim=0).item()
    rel = ((got - ref).norm() / (ref.norm() + 1e-12)).item()
    mad = (got - ref).abs().max().item()
    return cos, rel, mad


def _inputs(B, T, H, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
    # logsigmoid/normalizer keeps the cumulative gate from blowing up over T.
    gk_raw = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE) * 0.5 - 1.0
    g = F.logsigmoid(gk_raw) / 16.0
    return q, k, v, g


def test_chunk_gla(B, T, H):
    q, k, v, g = _inputs(B, T, H)
    scale = 1.0 / (K ** 0.5)
    o_ref, _ = fla_chunk_gla(q=q, k=k, v=v, g=g, output_final_state=False)

    ws = ext.make_workspace(B, T, H, q.device)
    o = torch.empty(B, T, H, V, dtype=DTYPE, device=DEVICE)
    ext.chunk_gla_run(q, k, v, g, scale, ws, o)
    return _metrics(o_ref, o)


def test_rmsnorm_gated(B, T, H):
    torch.manual_seed(1)
    o = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
    g = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
    w = (torch.randn(V, dtype=DTYPE, device=DEVICE).abs() + 0.5)

    ref_norm = FusedRMSNormGated(hidden_size=V, elementwise_affine=True, eps=EPS).to(DEVICE, DTYPE)
    ref_norm.weight.data.copy_(w)
    y_ref = ref_norm(o, g)

    y = torch.empty_like(o)
    ext.rmsnorm_gated(o, g, w, y, EPS)
    return _metrics(y_ref, y)


def test_chunk_gla_run_gated(B, T, H):
    q, k, v, g = _inputs(B, T, H)
    scale = 1.0 / (K ** 0.5)
    gate = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
    w = (torch.randn(V, dtype=DTYPE, device=DEVICE).abs() + 0.5)

    o_ref, _ = fla_chunk_gla(q=q, k=k, v=v, g=g, output_final_state=False)
    ref_norm = FusedRMSNormGated(hidden_size=V, elementwise_affine=True, eps=EPS).to(DEVICE, DTYPE)
    ref_norm.weight.data.copy_(w)
    y_ref = ref_norm(o_ref, gate)

    ws = ext.make_workspace(B, T, H, q.device)
    y = torch.empty(B, T, H, V, dtype=DTYPE, device=DEVICE)
    ext.chunk_gla_run_gated(q, k, v, g, gate, w, scale, EPS, ws, y)
    return _metrics(y_ref, y)


# (name, fn, cos_min, rel_max)
SUITE = [
    ("chunk_gla_run", test_chunk_gla, 0.99, 0.08),
    ("rmsnorm_gated", test_rmsnorm_gated, 0.999, 0.02),
    ("chunk_gla_run_gated", test_chunk_gla_run_gated, 0.99, 0.10),
]


def main():
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability()}")
    print(f"Config: B=1, H={H_DEFAULT}, K={K}, V={V}, dtype={DTYPE}\n")

    failures = 0
    for name, fn, cos_min, rel_max in SUITE:
        print(f"== {name} (cos>={cos_min}, rel<={rel_max}) ==")
        print(f"{'T':>6} | {'cos':>10} | {'rel_err':>10} | {'max_abs':>10} | {'status':>6}")
        print("-" * 56)
        for T in T_CASES:
            try:
                cos, rel, mad = fn(1, T, H_DEFAULT)
                ok = cos >= cos_min and rel <= rel_max
                failures += not ok
                print(f"{T:>6} | {cos:10.5f} | {rel:10.4e} | {mad:10.4e} | "
                      f"{'PASS' if ok else 'FAIL':>6}")
            except Exception as e:
                failures += 1
                print(f"{T:>6} | error: {e!r}")
        print()

    if failures:
        print(f"FAILED: {failures} case(s) out of tolerance")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
