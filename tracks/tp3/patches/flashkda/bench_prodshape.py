#!/usr/bin/env python3
"""Production-shaped FlashKDA vs Triton chunk-path benchmark.

Difference from bench_flashkda_vs_chunk.py: q/k/v are STRIDED views of one fused
qkv buffer, exactly as glm5next's _forward hands them over
(``qkv_ns.split(local_projection_size, dim=-1)`` then
``x.reshape(1, -1, H, D)`` -- a view, stride (.., 3*H*D, D, 1)).

The Triton chunk path consumes those strides directly.  FlashKDA hardcodes dense
q/k/v/g strides, so its wrapper must call ``.contiguous()`` on each -- three real
copies per KDA layer per step.  Here that cost is INSIDE the timed region for
the FlashKDA arm, so the comparison is the one production would see.
"""

import argparse
import json
import statistics
import sys
import time

import torch


def sync_time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=22)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--seqs", type=int, default=1)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lower-bound", type=float, default=-5.0)
    args = ap.parse_args()

    import vllm._flashkda_C  # noqa: F401
    from vllm.third_party.flash_linear_attention.ops.kda import (
        chunk_kda_with_fused_gate,
    )

    D, H, T, NS = 128, args.heads, args.tokens, args.seqs
    SL = T // NS
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    P = H * D  # local_projection_size

    free, _ = torch.cuda.mem_get_info()
    print(f"mem free={free / 2**20:.0f} MiB, H={H} T={T} seqs={NS}")

    # one fused conv output, as production produces it
    qkv = torch.randn(T, 3 * P, device=dev, dtype=torch.bfloat16)
    q_ns, k_ns, v_ns = qkv.split(P, dim=-1)

    def rearr(x):
        return x.reshape(1, -1, H, D)

    q, k, v = rearr(q_ns), rearr(k_ns), rearr(v_ns)
    print(
        f"q contiguous={q.is_contiguous()} strides={q.stride()} "
        f"(production hands the chunk kernel exactly this)"
    )
    g = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16) * 0.5
    beta_raw = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
    A_log = torch.randn(1, 1, H, 1, device=dev, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H * D, device=dev, dtype=torch.float32) * 0.1
    cu = torch.arange(0, T + 1, SL, device=dev, dtype=torch.int32)
    init = torch.randn(NS, H, D, D, device=dev, dtype=torch.float32) * 0.01
    lb = args.lower_bound
    row = {"heads": H, "total_tokens": T, "seqs": NS}

    # ---- FlashKDA, with the three .contiguous() copies inside the timed region
    ws = torch.empty(
        int(torch.ops._flashkda_C.get_workspace_size(T, H, NS)),
        device=dev,
        dtype=torch.uint8,
    )
    out_buf = torch.empty(1, T, H, D, device=dev, dtype=torch.bfloat16)
    fs_buf = torch.empty(NS, H, D, D, device=dev, dtype=torch.float32)
    A_flat = A_log.reshape(-1).contiguous()
    dtb = dt_bias.reshape(-1, D).contiguous()
    init_c = init.contiguous()
    cu_c = cu.contiguous()

    def run_fk(copies=True):
        torch.ops._flashkda_C.fwd(
            q.contiguous() if copies else q,
            k.contiguous() if copies else k,
            v.contiguous() if copies else v,
            g.contiguous() if copies else g,
            beta_raw,
            D**-0.5,
            out_buf,
            ws,
            A_flat,
            dtb,
            lb,
            init_c,
            fs_buf,
            cu_c,
        )

    run_fk()
    torch.cuda.synchronize()
    out_fk = out_buf.clone()
    fs_fk = fs_buf.clone()
    ts = sync_time(run_fk, args.iters, args.warmup)
    row["flashkda_with_copies_ms"] = statistics.median(ts)
    # copies alone, for the record
    tc = sync_time(lambda: (q.contiguous(), k.contiguous(), v.contiguous()), 5, 2)
    row["qkv_contiguous_ms"] = statistics.median(tc)

    # ---- Triton chunk path on the strided views (what production does today)
    def run_tri(clone):
        return chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v.clone() if clone else v,
            raw_g=g,
            beta=beta_raw.squeeze(0).float().sigmoid().unsqueeze(0),
            A_log=A_log,
            g_bias=dt_bias,
            initial_state=init,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu,
            safe_gate=True,
            lower_bound=lb,
        )

    o, s = run_tri(True)
    out_tri, fs_tri = o.clone(), s.clone()
    ts = sync_time(lambda: run_tri(False), args.iters, args.warmup)
    row["triton_ms"] = statistics.median(ts)
    row["speedup_prodshape"] = row["triton_ms"] / row["flashkda_with_copies_ms"]

    d = (out_tri.float() - out_fk.float()).abs()
    row["out_mean_abs_diff"] = d.mean().item()
    row["out_max_abs_diff"] = d.max().item()
    row["out_ref_absmean"] = out_tri.float().abs().mean().item()
    ds = (fs_tri.float() - fs_fk.float()).abs()
    row["state_mean_abs_diff"] = ds.mean().item()
    row["state_ref_absmean"] = fs_tri.float().abs().mean().item()
    print(json.dumps(row))


if __name__ == "__main__":
    main()
