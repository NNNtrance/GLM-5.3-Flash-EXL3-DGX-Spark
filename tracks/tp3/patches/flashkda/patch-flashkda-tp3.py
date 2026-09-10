#!/usr/bin/env python3
"""HAREM-TP3 FlashKDA KDA chunked prefill for GLM-5.3-Flash.

IN PRODUCTION since 10 September 2026.  Registered unconditionally by the
prelude; the behaviour is env-gated and default OFF (see ENV GATE below), so an
unset knob leaves the image byte for byte upstream on this path.  Measurements,
gates and the A/B that promoted it: results/gates/flashkda-ab-10sep.md.

Adapted port of vLLM PR #55737 ("[Perf][GLM-5.3-Flash] Use FlashKDA for KDA
chunked prefill") onto our pinned vLLM (487ecf187, image exl3-zeus:754421f).

WHAT IT DOES
------------
``vllm/models/glm5next/nvidia/kda.py`` runs its chunked-prefill branch through
``chunk_kda_with_fused_gate`` -- a ~10-kernel Triton chain (gate cumsum, two
scaled-dot-kkt passes, wy recompute, the chunk h/o pair, the 16x16->64x64
inverse, two l2norms, the beta sigmoid).  ``vllm._flashkda_C`` is already built
into this image (Kimi-K3 uses it) and implements the same bounded-gate KDA
recurrence in ONE fused CUDA kernel, including the q/k l2norm and the gate
``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, taking raw beta logits.

Measured on the head node (GB10, sm_121, 22 KDA heads/rank, head_dim 128,
bf16), engine idle, one short-lived process:

    tokens/step   Triton chain    FlashKDA    speedup
    512             0.449 ms      0.152 ms      2.9x
    1024            1.064 ms      0.326 ms      3.3x
    2048            2.238 ms      0.673 ms      3.3x   <- our max-num-batched-tokens
    4096            4.462 ms      1.350 ms      3.3x
    8192            9.437 ms      2.682 ms      3.5x

Output agreement with the Triton path: mean|d| 1.2e-5, max|d| 7.3e-4 on the
attention output (ref |mean| 2.5e-3, i.e. ~1 bf16 ULP); recurrent state
mean|d| 1.0e-4 (ref |mean| 2.9e-2).

ENV GATE -- DEFAULT OFF
-----------------------
  HAREM_KDA_FLASHKDA unset or 0   -> the Triton chain, byte for byte upstream.
  HAREM_KDA_FLASHKDA=1            -> FlashKDA when supported (CUDA SM90/SM10x/
                                     SM12x, head_dim 128, bf16, bounded gate);
                                     refuses loudly when it is not.

WHY THE UPSTREAM DIFF DOES NOT APPLY AS-IS
------------------------------------------
1. ``torch.ops._flashkda_C.fwd`` in THIS image takes 14 arguments and ends at
   ``cu_seqlens``:
     fwd(q, k, v, g, beta, scale, out!, workspace!, A_log, dt_bias,
         lower_bound, initial_state?, final_state!?, cu_seqlens?)
   The PR's base (main 94848eda) has two extra trailing optionals
   (``checkpoint_state``, ``checkpoint_offsets``) and passes ``None, None``.
   Both are dropped here -- same semantics, and passing them would raise.
2. The PR reads ``additional_config["kda_prefill_backend"]``.  Our production
   launcher passes no ``--additional-config``, so that knob can only be set by
   changing the unit.  Replaced by the HAREM env gate above (the
   ``additional_config`` knob is still honoured when present, so an
   ``--additional-config '{"kda_prefill_backend": "triton"}'`` still forces the
   old path).
3. Our ``A_log`` is ``[1, 1, H, 1]`` fp32 and ``dt_bias`` is ``[H*D]`` fp32;
   both are flattened with ``.contiguous()`` here (the PR's ``.view()`` alone
   happens to work, but the copy is free and survives a layout change).
4. Our file already defines ``ns_out`` above the prefill/decode branch (it is
   read by the final non-spec copy).  The FlashKDA branch assigns it, which is
   what makes that copy correctly skipped; spelled out rather than shadowed.
5. Our recurrent state dtype is fp32 (``kda_state_dtype(bf16, "auto")`` ->
   ``(bf16, fp32)``).  Measured: FlashKDA accepts an fp32 initial/final state.

NOT PORTED
----------
The PR also rewrites the spec/non-spec merge to ``index_copy_`` straight into
``core_attn_out`` (dropping one ``torch.empty`` + copy).  That is an
independent micro-optimisation of the Triton path as well; it is left out so
this patch changes exactly one thing.

FAIL-CLOSED
-----------
Five anchors, each required exactly once in the installed file.  A missing or
duplicated anchor exits non-zero instead of guessing.  Re-running is a no-op.

HOW TO INVOKE IT -- READ THIS, BOTH HALVES BIT US
-------------------------------------------------
    run python3 "$TP3_DIR/patch-flashkda-tp3.py" \\
        --root "$(dirname "$VLLM_PY")" --in-place

1. ``--root`` is the DIST-PACKAGES root, not ``$VLLM_PY``.  REL above starts
   with ``vllm/``, unlike every other patch script in this tree, because this
   one was written to run against a throwaway copy of the source layout.
   ``--root "$VLLM_PY"`` makes the path ``.../vllm/vllm/models/...`` and raises
   ``FileNotFoundError``; the prelude's fail-closed wrapper then stops the rank,
   which is the correct outcome and cost one boot to read.
2. Without ``--in-place`` this script is a DRY RUN: it prints "dry run OK",
   exits 0, and patches nothing.  Registered that way it produces a healthy
   boot that quietly still runs the Triton chain -- an A/B comparing nothing
   with nothing.  The boot-log line below is what makes that visible.

The resolver prints its choice once per process:

    [HAREM-FLASHKDA] kda_prefill_backend=flashkda  (HAREM_KDA_FLASHKDA='1')
    [HAREM-FLASHKDA] kda_prefill_backend=triton    (HAREM_KDA_FLASHKDA='0')

A boot log with no such line is a boot where this patch did not run.  PR #55737
prints nothing equivalent, which is why the line was added here.
"""

import argparse
import os
import sys

REL = "vllm/models/glm5next/nvidia/kda.py"
MARK = "HAREM-FLASHKDA"

# --- anchor 1: the import block ----------------------------------------------
A1_OLD = "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"
A1_NEW = (
    "from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata\n"
    "from vllm.v1.worker.workspace import current_workspace_manager\n"
)

# --- anchor 2: backend resolver, right after _cast_sigmoid -------------------
A2_OLD = "    return x.float().sigmoid()\n"
A2_NEW = '''    return x.float().sigmoid()


_HAREM_KDA_BACKEND_LOGGED = set()


def _harem_log_kda_backend(backend: str) -> str:
    """HAREM-FLASHKDA.  Say once per process which chunked-prefill kernel won.

    The resolver runs per KDA layer (34 of them), so the decision is printed the
    first time only; a boot log without this line means the patch is not
    installed at all.
    """
    if backend not in _HAREM_KDA_BACKEND_LOGGED:
        _HAREM_KDA_BACKEND_LOGGED.add(backend)
        print(
            f"[HAREM-FLASHKDA] kda_prefill_backend={backend} "
            f"(HAREM_KDA_FLASHKDA={os.environ.get('HAREM_KDA_FLASHKDA', '')!r})",
            flush=True,
        )
    return backend


def _harem_kda_prefill_backend(
    additional_config,
    head_dim: int,
    dtype: "torch.dtype",
    lower_bound: float | None,
) -> str:
    """HAREM-FLASHKDA.  Pick the chunked-prefill kernel.

    ``HAREM_KDA_FLASHKDA=1`` asks for FlashKDA (``vllm._flashkda_C``, the fused
    CUDA kernel Kimi-K3 already uses); unset/0 keeps the Triton
    ``chunk_kda_with_fused_gate`` chain, byte for byte upstream.
    ``additional_config["kda_prefill_backend"]`` (auto/triton/flashkda) is
    honoured too and wins over the env gate, so the PR's knob keeps working.
    """
    backend = os.environ.get("HAREM_KDA_FLASHKDA", "").strip().lower()
    backend = {"1": "auto", "flashkda": "flashkda", "": "triton", "0": "triton"}.get(
        backend, backend
    )
    if isinstance(additional_config, dict):
        backend = additional_config.get("kda_prefill_backend", backend)
    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    if backend == "triton":
        return _harem_log_kda_backend("triton")
    capability = current_platform.get_device_capability()
    supported = (
        current_platform.is_cuda()
        and capability is not None
        # SM90 is architecture-specific; SM10x and SM12x are built as family
        # binaries ("10.0f"/"12.0f" in cmake/external_projects/flashkda.cmake),
        # so the sm_120 cubin in this image runs on GB10 (sm_121) -- measured.
        and capability.major in (9, 10, 12)
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )
    if not supported:
        raise RuntimeError(
            "HAREM-FLASHKDA: FlashKDA requires CUDA SM90/SM10x/SM12x, bfloat16, "
            f"head_dim=128 and a bounded KDA gate; got capability={capability}, "
            f"dtype={dtype}, head_dim={head_dim}, lower_bound={lower_bound}"
        )
    return _harem_log_kda_backend("flashkda")
'''

# --- anchor 3: end of __init__ (vllm_config is in scope here) ---------------
A3_OLD = "        self._conv_state_dim_first = is_conv_state_dim_first()\n"
A3_NEW = '''        self._conv_state_dim_first = is_conv_state_dim_first()

        # HAREM-FLASHKDA.  Resolve the chunked-prefill kernel once, and size the
        # three workspace buffers FlashKDA needs (recurrent final state, scratch,
        # and an output buffer for steps that also carry spec-decode tokens).
        self.kda_prefill_backend = _harem_kda_prefill_backend(
            vllm_config.additional_config,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.kda_lower_bound,
        )
        self._flashkda_buffer_specs = None
        if self.kda_prefill_backend == "flashkda":
            import vllm._flashkda_C  # noqa: F401

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            workspace_size = torch.ops._flashkda_C.get_workspace_size(
                max_tokens, self.local_num_heads, max_seqs
            )
            self._flashkda_buffer_specs = (
                (
                    (max_seqs, self.local_num_heads, self.head_dim, self.head_dim),
                    self.get_state_dtype()[1],
                ),
                ((workspace_size,), torch.uint8),
                (
                    (1, max_tokens, self.local_num_heads, self.head_dim),
                    vllm_config.model_config.dtype,
                ),
            )
'''

# --- anchor 4: insert the kernel wrapper just before forward() --------------
A4_OLD = "    def forward(\n"
A4_NEW = '''    def _harem_flashkda_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HAREM-FLASHKDA.  Fused KDA chunked prefill.

        Takes the RAW gate logits ``g`` and RAW ``beta`` logits, l2-normalizes
        q/k in-kernel and applies the bounded gate
        ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` -- the same thing
        ``chunk_kda_with_fused_gate(..., safe_gate=True)`` computes.  Writes the
        attention output into ``out`` (a workspace buffer when ``None``) and
        returns ``(out, final_state)``.

        NOTE the 14-argument call: this image's ``_flashkda_C::fwd`` ends at
        ``cu_seqlens``; upstream's newer ``checkpoint_state`` /
        ``checkpoint_offsets`` tail does not exist here.
        """
        assert self._flashkda_buffer_specs is not None
        final_state, workspace, workspace_out = current_workspace_manager(
        ).get_simultaneous(*self._flashkda_buffer_specs)
        final_state = final_state[: initial_state.shape[0]]
        if out is None:
            out = workspace_out[:, : q.shape[1]]
        # FlashKDA hardcodes dense q/k/v/g strides; beta may be row-strided.
        torch.ops._flashkda_C.fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta,
            self.head_dim**-0.5,
            out,
            workspace,
            self.A_log.reshape(-1).contiguous(),
            self.dt_bias.reshape(-1, self.head_dim).contiguous(),
            self.kda_lower_bound,
            initial_state.contiguous(),
            final_state,
            cu_seqlens.contiguous(),
        )
        return out, final_state

    def forward(
'''

# --- anchor 5: the chunked-prefill call --------------------------------------
A5_OLD = '''            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                raw_g=g1_ns,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_ns is raw bf16 from forward.
                beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
'''
A5_NEW = '''            if self.kda_prefill_backend == "flashkda":
                # HAREM-FLASHKDA.  Non-spec step: write straight into the layer
                # output buffer (dense token order, no merge copy) -- ns_out
                # stays non-None so the tail copy below is skipped.  A step that
                # also carries spec-decode tokens writes to the workspace buffer
                # and is scattered by non_spec_token_indx in the merge below.
                ns_out = None if use_spec else core_attn_out[:, :num_actual_tokens]
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = self._harem_flashkda_prefill(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    g=g1_ns,
                    beta=beta_ns,
                    initial_state=initial_state,
                    cu_seqlens=non_spec_query_start_loc,
                    out=ns_out,
                )
            else:
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_kda_with_fused_gate(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    raw_g=g1_ns,
                    # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                    # don't sigmoid); beta_ns is raw bf16 from forward.
                    beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc,
                    safe_gate=safe_gate,
                    lower_bound=lower_bound,
                )
'''

ANCHORS = [
    ("A1-import", A1_OLD, A1_NEW),
    ("A2-resolver", A2_OLD, A2_NEW),
    ("A3-init", A3_OLD, A3_NEW),
    ("A4-wrapper", A4_OLD, A4_NEW),
    ("A5-prefill-call", A5_OLD, A5_NEW),
]


def apply(src: str, where: str) -> str:
    if MARK in src:
        print(f"patch-flashkda: already applied ({where})")
        return src
    if "import os\n" not in src:
        # kda.py imports os lazily inside __init__ in the HAREM-FULLSCOPE build;
        # the resolver needs it at module scope.
        src = src.replace("import torch\n", "import os\n\nimport torch\n", 1)
    for name, old, new in ANCHORS:
        n = src.count(old)
        if n != 1:
            print(
                f"patch-flashkda: {name} count={n} (expected 1) in {where} "
                "-- refusing",
                file=sys.stderr,
            )
            sys.exit(3)
    for _name, old, new in ANCHORS:
        src = src.replace(old, new, 1)
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dist-packages root")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--out", default="", help="write here instead of in place")
    a = ap.parse_args()
    p = os.path.join(a.root, REL)
    src = open(p).read()
    out = apply(src, REL)
    if out is src:
        return
    dst = a.out or (p if a.in_place else "")
    if not dst:
        print(
            "patch-flashkda: dry run OK (pass --in-place or --out to write)",
        )
        return
    open(dst, "w").write(out)
    print(f"patch-flashkda: applied to {dst} (HAREM_KDA_FLASHKDA honoured)")


if __name__ == "__main__":
    main()
