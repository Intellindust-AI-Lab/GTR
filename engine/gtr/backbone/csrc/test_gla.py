"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Correctness gate for the hand-written chunk_gla CUDA operator.

Reference is fla (Triton). The kernels are fp16 with a different tiling than fla,
so outputs are NOT bit-exact; we assert high cosine similarity and a bounded
relative error instead. Single products through strong gates, which those norms
cannot see, are checked against a float64 recurrence.

Run:  bash build.sh && python test_gla.py

Covers the full operator surface exposed by gla_torch_ext.cu:
  * chunk_gla_run        — chunk GLA forward (q,k,v,gk -> o)
  * rmsnorm_gated        — fused RMSNorm + swish gate (D=64)
  * chunk_gla_run_gated  — chunk GLA + RMSNorm + swish gate fused in one launch
"""

import os
import sys
from functools import partial

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



GATES = {"weak": (-1.0, 0.5), "strong": (-24.0, 24.0)}


def _inputs(B, T, H, seed=0, gates="weak"):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
    k = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE)
    v = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE)
    if gates == "bursts":
        t = torch.arange(T, device=DEVICE).view(1, T, 1, 1)
        phase = torch.randint(0, 12, (B, 1, H, K), device=DEVICE)
        gk_raw = torch.where((t + phase) % 12 < 8, -160.0, 4.0).to(DTYPE)
    else:
        mean, std = GATES[gates]
        gk_raw = torch.randn(B, T, H, K, dtype=DTYPE, device=DEVICE) * std + mean
    g = F.logsigmoid(gk_raw) / 16.0
    return q, k, v, g


def test_chunk_gla(B, T, H, gates="weak"):
    q, k, v, g = _inputs(B, T, H, gates=gates)
    scale = 1.0 / (K ** 0.5)
    o_ref, _ = fla_chunk_gla(q=q, k=k, v=v, g=g, output_final_state=False)

    ws = ext.make_workspace(B, T, H, q.device)
    o = torch.empty(B, T, H, V, dtype=DTYPE, device=DEVICE)
    ext.chunk_gla_run(q, k, v, g, scale, ws, o)
    return _metrics(o_ref, o)


def test_chunk_gla_batch(B, T, H):
    return test_chunk_gla(8, T, H)


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


def test_chunk_gla_run_gated(B, T, H, gates="weak"):
    q, k, v, g = _inputs(B, T, H, gates=gates)
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


T_EXACT = 384   


def _recurrence_fp64(q, k, v, g):
    q, k, v, g = (x.double().cpu() for x in (q, k, v, g))
    B, T, H, _ = q.shape
    S = torch.zeros(B, H, K, V, dtype=torch.float64)
    o = torch.empty(B, T, H, V, dtype=torch.float64)
    for t in range(T):
        S = g[:, t].exp().unsqueeze(-1) * S + k[:, t].unsqueeze(-1) * v[:, t].unsqueeze(-2)
        o[:, t] = K ** -0.5 * (q[:, t].unsqueeze(-1) * S).sum(-2)
    return o


def _exact_case(q, k, v, g, row, gated=False, raw=False):
    """Relative error of output row `row`. raw passes logsigmoid's input (gk_normalizer=16) instead of g."""
    ref = _recurrence_fp64(q, k, v, F.logsigmoid(g.double()) / 16 if raw else g)
    if gated:   # RMSNorm with unit weight and a unit swish gate
        ref = ref * torch.rsqrt(ref.square().mean(-1, keepdim=True) + EPS) * F.silu(torch.ones_like(ref))
    ws = ext.make_workspace(1, T_EXACT, 1, q.device)
    o = torch.empty_like(v)
    if gated:
        ext.chunk_gla_run_gated(q, k, v, g, torch.ones_like(v), torch.ones(V, dtype=DTYPE, device=DEVICE),
                                K ** -0.5, EPS, ws, o, 16.0 if raw else 0.0)
    else:
        ext.chunk_gla_run(q, k, v, g, K ** -0.5, ws, o, 16.0 if raw else 0.0)
    ref, got = ref[0, row, 0], o[0, row, 0].double().cpu()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item(), bool(torch.isfinite(o).all())


def _blank(gate=-0.08):
    z = torch.zeros(1, T_EXACT, 1, K, dtype=DTYPE, device=DEVICE)
    return z, z.clone(), torch.zeros(1, T_EXACT, 1, V, dtype=DTYPE, device=DEVICE), torch.full_like(z, gate)


def exact_cases():
    """(name, rel_err, finite): every case must be finite and within 2% of the float64 recurrence."""
    out = []
    for gate in (-10.5, -12.0):
        for j in (0, 7, 23, 63):
            for gated, raw, exact in ((False, False, False), (True, False, False), (True, True, False),
                                      (True, False, True)):
                if raw and j:
                    continue
                q, k, v, g = _blank(-1.0 if raw else -0.08)
                q[0, j + 1, 0, 0], k[0, j, 0, 0], v[0, j] = 16.0, 16.0, 4.0
                g[0, j + 1, 0, 0] = gate * 16 if raw else gate
                if exact:
                    g[0, (j + 1) // 64 * 64 + 41, 0, 1] = -100.0
                out.append((f"history gate={gate} j={j} gated={gated} raw={raw} exact={exact}",
                            *_exact_case(q, k, v, g, j + 1, gated, raw)))

    for gate in (-8.7, -9.5):
        for row in (7, 8, 23, 24, 39, 40, 55, 56):
            q, k, v, g = _blank(gate)
            tiny_q = row % 16 == 7   # the query factor is the small one after the reference
            q[0, row, 0, 0] = 2.0 ** -24 if tiny_q else 32768.0
            k[0, row, 0, 0] = 32768.0 if tiny_q else 2.0 ** -24
            v[0, row] = 1.0
            out.append((f"subnormal operand gate={gate} row={row}", *_exact_case(q, k, v, g, row)))
    torch.manual_seed(0)
    q, k, v, g = _blank(-1000.0)
    q.normal_(), k.normal_(), v.normal_()
    ref = _recurrence_fp64(q, k, v, g)
    ws = ext.make_workspace(1, T_EXACT, 1, q.device)
    o = torch.empty_like(v)
    ext.chunk_gla_run(q, k, v, g, K ** -0.5, ws, o)
    out.append(("all gates -1000", ((o.double().cpu() - ref).norm() / ref.norm()).item(),
                bool(torch.isfinite(o).all())))
    return out


# (name, fn, cos_min, rel_max)
SUITE = [
    ("chunk_gla_run", test_chunk_gla, 0.99, 0.08),
    ("chunk_gla_run, strong gates", partial(test_chunk_gla, gates="strong"), 0.99, 0.08),
    ("chunk_gla_run, bursts", partial(test_chunk_gla, gates="bursts"), 0.99, 0.08),
    ("chunk_gla_run, batch 8", test_chunk_gla_batch, 0.99, 0.08),
    ("rmsnorm_gated", test_rmsnorm_gated, 0.999, 0.02),
    ("chunk_gla_run_gated", test_chunk_gla_run_gated, 0.99, 0.10),
    ("chunk_gla_run_gated, strong gates", partial(test_chunk_gla_run_gated, gates="strong"), 0.99, 0.10),
    ("chunk_gla_run_gated, bursts", partial(test_chunk_gla_run_gated, gates="bursts"), 0.99, 0.10),
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

    print(f"== single terms against a float64 recurrence, T={T_EXACT}, H=1 (finite, rel<=0.02) ==")
    for name, rel, finite in exact_cases():
        ok = finite and rel <= 0.02
        failures += not ok
        print(f"{name:<60} | rel_err {rel:10.4e} | {'PASS' if ok else 'FAIL':>6}")
    print()

    if failures:
        print(f"FAILED: {failures} case(s) out of tolerance")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
