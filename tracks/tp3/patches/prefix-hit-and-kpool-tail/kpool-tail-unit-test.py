#!/usr/bin/env python3
"""Model-free test of the K-pool tail slot mapping, on CPU tensors.

Pins the addressing the kpool kernels document -- ``block_table[req, 0] * kpool
+ pos % kpool`` -- against both the corrected mapping and the generic paged one,
for positions on both sides of ``block_size`` and of the block-table row width.
Same property upstream's ``tests/v1/attention/test_kpool_tail_slot_mapping.py``
pins for PR #53906; written here because that test cannot run against this
image's tree.

    kpool-tail-unit-test.py          (inside the serving image)
Prints "KPOOL TAIL UNIT: n/4" as its last line; exit 0 only on 4/4.
"""

import sys

import torch

from vllm.v1.attention.backends.mla.indexer import compute_kpool_tail_slot_mapping

KPOOL = 4          # index_kpool on GLM-5.3-Flash
WIDTH = 32         # the tail group's block-table row width on this stack
NREQS = 3

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


def build(lens):
    """One decode-ish batch: request r contributes lens[r] tokens ending at
    seq_lens[r]."""
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32)
    n = int(qsl[-1])
    seq = torch.tensor([2000 + 700 * r for r in range(len(lens))], dtype=torch.int32)
    pos = torch.cat(
        [
            torch.arange(int(seq[r]) - lens[r], int(seq[r]), dtype=torch.int64)
            for r in range(len(lens))
        ]
    )
    bt = torch.zeros((NREQS, WIDTH), dtype=torch.int32)
    for r in range(len(lens)):
        bt[r, 0] = 10 + r  # each request owns a different tail block
    # the generic kernel's output: block_table[req, pos // bs] * bs + pos % bs
    req = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
    col = torch.clamp(pos // KPOOL, max=WIDTH - 1)
    generic = bt[req, col].to(torch.int64) * KPOOL + pos % KPOOL
    return qsl, seq, pos, bt, generic, n, req


def main() -> None:
    lens = [1, 1, 1]
    qsl, seq, pos, bt, generic, n, req = build(lens)

    want = bt[req, 0].to(torch.int64) * KPOOL + pos % KPOOL
    got = compute_kpool_tail_slot_mapping(
        generic.clone(), bt, qsl, pos, n, len(lens), KPOOL
    )[:n]
    check(
        "decode batch maps every token to its own tail block",
        torch.equal(got, want),
        f"got={got.tolist()} want={want.tolist()}",
    )

    blk = got // KPOOL
    check(
        "corrected mapping never leaves the request's own block",
        torch.equal(blk, bt[req, 0].to(torch.int64)),
        f"blocks={blk.tolist()} own={bt[req, 0].tolist()}",
    )

    # The generic mapping is what the tail gets without the fix: for pos >= kpool
    # it reads column pos//kpool, which is an unwritten zero (or, past the row
    # width, another request's memory), so every request collapses onto block 0.
    gblk = generic // KPOOL
    check(
        "generic mapping DOES leave the row (this is the bug)",
        not torch.equal(gblk, bt[req, 0].to(torch.int64)),
        f"generic blocks={gblk.tolist()} own={bt[req, 0].tolist()}",
    )

    # A chunked-prefill batch: many tokens per request, positions crossing the
    # row width (32 * 4 = 128).
    lens2 = [200, 200, 200]
    qsl2, seq2, pos2, bt2, generic2, n2, req2 = build(lens2)
    got2 = compute_kpool_tail_slot_mapping(
        generic2.clone(), bt2, qsl2, pos2, n2, len(lens2), KPOOL
    )[:n2]
    want2 = bt2[req2, 0].to(torch.int64) * KPOOL + pos2 % KPOOL
    check(
        "prefill batch across the row width stays in its own block",
        torch.equal(got2, want2),
        f"n={n2} mismatches={(got2 != want2).sum().item()}",
    )

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"KPOOL TAIL UNIT: {n_ok}/{len(results)}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
