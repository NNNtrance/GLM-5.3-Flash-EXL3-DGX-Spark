#!/usr/bin/env python3
"""FlashKDA (vllm._flashkda_C) vs Triton chunk_kda_with_fused_gate micro-benchmark.

Mirrors the exact call GLM-5.3-Flash's vllm/models/glm5next/nvidia/kda.py makes
in its chunked-prefill branch (bounded gate, in-kernel q/k l2norm, raw beta for
FlashKDA / pre-sigmoided fp32 beta for the Triton path).

Read-only w.r.t. the engine: single short-lived process, own CUDA context.
Usage:  python3 bench_flashkda_vs_chunk.py [--heads 22,16] [--lens 1024,2048,4096,8192]

RUN ONE ARM PER PROCESS ON A LOADED NODE.  Beside a serving engine there are
about 1.6 GiB free, and both arms in one process push the Triton path into
allocator pressure: at 8,192 tokens that reads 32.8 ms against the 9.44 ms the
same arm measures on its own -- a 3.5x artefact, and it would have been reported
as a 12x speedup.  Use --skip-triton / --skip-flashkda and two invocations, and
keep --min-free-mib armed so the script refuses rather than measures pressure.
The production-shaped comparison (strided q/k/v, so FlashKDA pays for its three
.contiguous() copies) is bench_prodshape.py -- that is the honest ratio, and it
is smaller than this one.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time

import torch


def mem():
    free, total = torch.cuda.mem_get_info()
    return free / 2**20, total / 2**20


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
        ts.append((time.perf_counter() - t0) * 1e3)  # ms
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="22")
    ap.add_argument("--lens", default="1024,2048,4096,8192")
    ap.add_argument("--seqs", type=int, default=1,
                    help="number of equal-length sequences; total tokens = seqs*len")
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--lower-bound", type=float, default=-5.0)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--skip-triton", action="store_true")
    ap.add_argument("--skip-flashkda", action="store_true")
    ap.add_argument("--min-free-mib", type=float, default=1200.0)
    ap.add_argument("--triton-clone-in-timing", action="store_true",
                    help="keep the v.clone() inside the timed region (biases the "
                         "Triton arm; production feeds a fresh v each step, so "
                         "the default excludes it)")
    args = ap.parse_args()

    D = 128
    dev = torch.device("cuda:0")
    torch.manual_seed(0)

    import vllm._flashkda_C  # noqa: F401
    from vllm.third_party.flash_linear_attention.ops.kda import (
        chunk_kda_with_fused_gate,
    )

    cap = torch.cuda.get_device_capability(0)
    print(f"device={torch.cuda.get_device_name(0)} cap={cap} torch={torch.__version__}")
    f, t = mem()
    print(f"mem free={f:.0f} MiB / total={t:.0f} MiB")
    print(f"flashkda ext = {vllm._flashkda_C.__file__}")

    results = []
    for H in [int(x) for x in args.heads.split(",")]:
        for T in [int(x) for x in args.lens.split(",")]:
            f0, _ = mem()
            if f0 < args.min_free_mib:
                print(f"ABORT: only {f0:.0f} MiB free before H={H} T={T}")
                break
            row = {"heads": H, "total_tokens": T, "seqs": args.seqs,
                   "seqlen_each": T // args.seqs, "head_dim": D}
            try:
                # ---- inputs, exactly as glm5next's kda.py hands them over ----
                # q/k/v: [1, T, H, D] bf16 (post short-conv slices, _rearr'd)
                q = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
                k = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
                v = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
                # g1: [1, T, H, D] bf16 raw gate logits (f_b_proj output)
                g = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16) * 0.5
                # beta: [1, T, H] bf16 RAW logits
                beta_raw = torch.randn(1, T, H, device=dev, dtype=torch.bfloat16)
                # A_log: [1, 1, H, 1] fp32 ; dt_bias: [H*D] fp32
                A_log = torch.randn(1, 1, H, 1, device=dev, dtype=torch.float32) * 0.1
                dt_bias = torch.randn(H * D, device=dev, dtype=torch.float32) * 0.1
                # cu_seqlens int32 [NS+1] for NS equal-length sequences
                NS = args.seqs
                assert T % NS == 0, "len must be divisible by seqs"
                SL = T // NS
                cu = torch.arange(
                    0, T + 1, SL, device=dev, dtype=torch.int32
                )
                # recurrent state: fp32 [NS, H, D, D] (kda_state_dtype -> fp32)
                init_state = torch.randn(
                    NS, H, D, D, device=dev, dtype=torch.float32
                ) * 0.01

                out_tri = fs_tri = None
                out_fk = fs_fk = None

                # ---------------- FlashKDA ----------------
                if not args.skip_flashkda:
                    ws_size = torch.ops._flashkda_C.get_workspace_size(T, H, NS)
                    row["workspace_bytes"] = int(ws_size)
                    workspace = torch.empty(
                        int(ws_size), device=dev, dtype=torch.uint8
                    )
                    out_buf = torch.empty(
                        1, T, H, D, device=dev, dtype=torch.bfloat16
                    )
                    fs_buf = torch.empty(
                        NS, H, D, D, device=dev, dtype=torch.float32
                    )
                    qc, kc, vc, gc_ = (
                        q.contiguous(),
                        k.contiguous(),
                        v.contiguous(),
                        g.contiguous(),
                    )
                    A_flat = A_log.view(-1).contiguous()
                    dtb = dt_bias.view(-1, D).contiguous()
                    isc = init_state.contiguous()
                    cuc = cu.contiguous()

                    def run_fk():
                        torch.ops._flashkda_C.fwd(
                            qc,
                            kc,
                            vc,
                            gc_,
                            beta_raw,
                            D**-0.5,
                            out_buf,
                            workspace,
                            A_flat,
                            dtb,
                            args.lower_bound,
                            isc,
                            fs_buf,
                            cuc,
                        )

                    run_fk()
                    torch.cuda.synchronize()
                    out_fk, fs_fk = out_buf.clone(), fs_buf.clone()
                    ts = sync_time(run_fk, args.iters, args.warmup)
                    row["flashkda_ms_median"] = statistics.median(ts)
                    row["flashkda_ms_min"] = min(ts)
                    row["flashkda_ms_all"] = [round(x, 3) for x in ts]

                # ---------------- Triton chunk path ----------------
                if not args.skip_triton:
                    def run_tri(clone=True):
                        # the chunk kernel writes into v's buffer. Correctness
                        # runs feed a clone; timed runs feed v directly (as
                        # production does: v is a fresh conv-output slice), so
                        # no clone cost is charged to the Triton arm.
                        o, s = chunk_kda_with_fused_gate(
                            q=q,
                            k=k,
                            v=v.clone() if clone else v,
                            raw_g=g,
                            beta=beta_raw.squeeze(0).float().sigmoid().unsqueeze(0),
                            A_log=A_log,
                            g_bias=dt_bias,
                            initial_state=init_state,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                            cu_seqlens=cu,
                            safe_gate=True,
                            lower_bound=args.lower_bound,
                        )
                        return o, s

                    o, s = run_tri(clone=True)
                    out_tri, fs_tri = o.clone(), s.clone()
                    _c = bool(args.triton_clone_in_timing)
                    row["triton_clone_in_timing"] = _c
                    ts = sync_time(lambda: run_tri(clone=_c), args.iters, args.warmup)
                    # cost of one v.clone() alone, for reference
                    tc = sync_time(lambda: v.clone(), 5, 2)
                    row["v_clone_ms_median"] = statistics.median(tc)
                    row["triton_ms_median"] = statistics.median(ts)
                    row["triton_ms_min"] = min(ts)
                    row["triton_ms_all"] = [round(x, 3) for x in ts]
                    del o, s
                    gc.collect()
                    torch.cuda.empty_cache()

                # ---------------- numerics ----------------
                if out_tri is not None and out_fk is not None:
                    a = out_tri.float()
                    b = out_fk.float()
                    d = (a - b).abs()
                    row["out_mean_abs_diff"] = d.mean().item()
                    row["out_max_abs_diff"] = d.max().item()
                    row["out_ref_absmean"] = a.abs().mean().item()
                    ds = (fs_tri.float() - fs_fk.float()).abs()
                    row["state_mean_abs_diff"] = ds.mean().item()
                    row["state_max_abs_diff"] = ds.max().item()
                    row["state_ref_absmean"] = fs_tri.float().abs().mean().item()
                if "triton_ms_median" in row and "flashkda_ms_median" in row:
                    row["speedup"] = row["triton_ms_median"] / row["flashkda_ms_median"]

            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"ERROR H={H} T={T}: {row['error']}", file=sys.stderr)

            f1, _ = mem()
            row["mem_free_after_MiB"] = round(f1)
            results.append(row)
            print(json.dumps(row), flush=True)

            for name in list(locals().keys()):
                pass
            try:
                del q, k, v, g, beta_raw, A_log, dt_bias, cu, init_state
            except Exception:
                pass
            for nm in ("out_tri", "fs_tri", "out_fk", "fs_fk", "workspace",
                       "out_buf", "fs_buf", "qc", "kc", "vc", "gc_",
                       "A_flat", "dtb", "isc", "cuc"):
                if nm in locals():
                    del locals()[nm]
            gc.collect()
            torch.cuda.empty_cache()

    print("\n===== SUMMARY =====")
    hdr = f"{'H':>3} {'Ttot':>6} {'triton ms':>10} {'flashkda ms':>12} {'x':>6} {'mean|d|':>10} {'max|d|':>10}"
    print(hdr)
    for r in results:
        if "error" in r:
            print(f"{r['heads']:>3} {r['total_tokens']:>6}  ERROR: {r['error']}")
            continue
        print(
            f"{r['heads']:>3} {r['total_tokens']:>6} "
            f"{r.get('triton_ms_median', float('nan')):>10.3f} "
            f"{r.get('flashkda_ms_median', float('nan')):>12.3f} "
            f"{r.get('speedup', float('nan')):>6.2f} "
            f"{r.get('out_mean_abs_diff', float('nan')):>10.3e} "
            f"{r.get('out_max_abs_diff', float('nan')):>10.3e}"
        )
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\njson -> {args.json_out}")


if __name__ == "__main__":
    main()
