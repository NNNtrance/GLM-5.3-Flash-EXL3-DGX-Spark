# Audit — check your install against ours, and check ours against itself

Two audiences, one folder.

**If you have built the stack:** run `audit/run-audit.sh` and compare what you get against the
expected ranges on this page. The script prints numbers; it does not grade them. Grading is this
page.

**If you are reviewing what we published:** §5 is the provenance table — every headline number, the
date it was measured, the configuration it was measured on, and the file it traces to. §6 is the
retraction index: everything we published and then withdrew, with what replaced it. A reviewer who
reads nothing else should read §6.

```
audit/run-audit.sh
```

About 25 minutes end to end; the category-speed step is most of it. One section at a time:

```
audit/run-audit.sh health kv gates c1 prefill memory fabric
```

Sections: `health` `kv` `gates` `c1` `prefill` `category` `memory` `fabric` `versions`.

---

## 1. Settings every number below assumes

Change any row and the expected values change with it.

| | |
|---|---|
| Image | `exl3-zeus:754421f` (`cuda-exl3` `754421f`, carrying `f3e3090`) |
| Checkpoint | `turboderp/GLM-5.3-Flash-exl3`, branch `4.05bpw`, revision `2a30229e` — **full scope** |
| Parallelism | TP=3, expert parallelism on, three nodes |
| KV | `kv-cache-dtype fp8`, **and an fp8 draft cache** (`HAREM_DRAFT_KV_DTYPE=fp8`), `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256` |
| Speculative | DFlash2, k=7 |
| Memory | `--gpu-memory-utilization` **0.88**, `SETTLE_MIN_GIB=112` |
| Guard patches | `HAREM_SM12_ITEMS=pdl,kpool` (the sm_12x correctness set) and `HAREM_INDEXER_WS_MODE=bound` (the sparse-indexer K-gather workspace, 5,036.40 → 513.00 MB), with `VLLM_DEBUG_WORKSPACE=1` for the per-boot proof of the second |
| Batching | `--max-num-seqs 8`, `--max-num-batched-tokens 2048` |
| Fabric | mesh plugin `19924dcc` + patches `0004`/`0005`/`0006`, `NCCL_MAX_NCHANNELS=8`, `NCCL_MESH_LINKS_PER_PEER=0`, `NCCL_MESH_PTR_CUDA=1` |
| Boot | per-rank fast-load sidecar, warm `CUDA_EXL3_TUNE_CACHE` |
| Sampling | temperature 0, thinking on, reasoning effort **low** |
| Date | 6 September 2026 |

**This is production configuration 12 — what our three nodes serve, start at boot and were rebooted
into as a whole cluster with the gates read afterwards. §2 grades a fresh boot against it.** Anything
you publish from this script must carry that table with it. A tok/s figure without its configuration
is not a measurement.

**The line, in one line.** Production 10 (5 September 2026) is `gpu-memory-utilization` **0.83** →
production 11 (6 September 2026) is **0.87** plus the sm_12x correctness set → production 12
(6 September 2026, promoted 16:53) is **0.88** plus the indexer workspace bound. Each step is a
memory fraction and one env-gated patch set, and nothing else moves: same image, same checkpoint,
same fabric settings, same batching, same sampling. Production 9
(5 September 2026, 0.80) is the configuration every *analysis* page in this repository was written
on, and it applies unchanged — 9 through 12 differ by a memory fraction, two guard patches and one
buffer size.

**Production 9's expectations are kept beside production 12's below**, in a `production 9 [history]`
column or a collapsed block, so a number quoted from an older commit can still be located. Grade
against the production 12 column.

**One thing below was not re-derived on production 12, and it says so**: §4's MMLU sample, which is
production 9's 86.47 ±0.74 carried forward `[not tested]` on 10, 11 and 12. §2's category speed was
re-measured on production 12 on 7 September.

**Everything here is at reasoning effort `low`.** Max effort would cost 5–12× the tokens and days of
cluster time. Nothing on this page is a max-effort number and none of it should be quoted as one.

---

## 2. What to expect — production configuration 12

Work down this list in order. Each step rules out the ones after it, and there is a reason for that
ordering: **a fast engine that answers wrong is worthless, and a machine that is swapping produces
speed numbers that mean nothing.**

**Every expected value below is production 12's, and every speed row carries the band it is graded
in** — the declared boot-to-boot bands for this stack, from
[docs/09](../docs/09-measurement-protocol.md) §1.2: **C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %**,
and **±6 % on the KV pool**. C4 is the noisiest level on this stack and C1 and C8 are the two that
carry a verdict; a C4 or C6 reading from a single boot is not evidence of anything. Speed figures are
the **pool of six sweep rounds over two boots** — the load boot and the clean boot after a
whole-cluster reboot — which is the protocol configurations 11 and 12 were both promoted under.

### versions — before anything else

The audit's first section prints your environment against §5 of
[docs/00](../docs/00-hardware-and-os.md). Nothing here is graded automatically; you are looking for
the two that bite:

- `ibv_devinfo | grep -c PORT_ACTIVE` must be **4** on every node.
- `lspci -vv` must report `Speed 32GT/s, Width x4` for each ConnectX endpoint. If yours reads `x2`,
  your SBIOS is below `0104` and **you are measuring a different machine from ours** — fix that
  before you read another number ([docs/00](../docs/00-hardware-and-os.md) §2.1).

### health

`GET /health` returns 200 within a few milliseconds once the engine is serving. **Only the head
(rank 0) serves the API**; the workers have no HTTP endpoint at all, and a `curl` against a worker
timing out is correct behaviour, not a fault.

### fabric — the check almost everyone skips `[measured-here]`

`PORT_ACTIVE` is link state, not traffic. Read the transmit counters as a **delta** across a
benchmark:

```
for p in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do echo "$p $(cat $p)"; done
```

| | Expected |
|---|---|
| Ports whose counter moves | **4 of 4** per node |
| Ports flat at a value that never changes | **0** |

Two flat ports per node is the pre-patch condition, and it cost us a factor of two on the ceiling.
It is what patch `0005` fixes. See [docs/06](../docs/06-nccl-mesh.md) §6.

### KV pool — `[measured-here]`

Read from the engine's own start-up log line `GPU KV cache size: N tokens`, on a **load** boot with
a settled memory baseline.

| | Expected (production 12) |
|---|---|
| **KV pool** | **7,041,322 tokens** — three boots of this configuration read 7,170,798 / 7,088,154 / 7,041,322 |
| **Pass range**, ±6 % of the headline | **6,618,843 – 7,463,801** |
| Available KV, per rank | 50.75 / 50.89 / 51.09 GiB |
| Consumed per node (weights + non-torch) | **54.28 – 54.62 GiB** |
| Locked indexer workspace | **513.00 MB**, and **exactly one** resize line per rank |
| Concurrent 1M-token requests | about **7.0** |

The pool at each earlier configuration, so a number quoted from an older commit can be located:

| Configuration | `gpu-memory-utilization` | KV pool (tokens) |
|---|---|---|
| Production 7 (routed-experts-only, fp8 draft cache) | 0.80 | 4,696,969 |
| Production 8 (image `62f53e6`) | 0.80 | 4,696,969 |
| Production 9 (full-scope checkpoint) | 0.80 | 5,165,289 (5,168,044 on the verification boot) |
| Production 10 (production 9 + one line) | 0.83 | 5,619,834 |
| Production 11 (0.87 + sm_12x correctness set) | 0.87 | 6,382,920 |
| **Production 12 (0.88 + indexer workspace bound)** | **0.88** | **7,041,322** |

**Where production 12's pool comes from, and why it is not a percentage sum.** The workspace bound
was priced at **+644,628 tokens** by its own A/B and the 0.87 → 0.88 rung at **+179,063** by the
memory ladder; the two add **in absolute tokens**, predicting 7,209,365 against a measured
7.04–7.17 M — inside the ±6 % boot-to-boot band. They do not add in percent, because the second lands
on a base the first has already enlarged, which is why the headline is +10.3 % against production 11
and not the +13.4 % a percentage sum would promise
([`results/memory/indexer-workspace-ab.md`](../results/memory/indexer-workspace-ab.md) §7.1).

**The workspace bound is checkable from the engine's own log, and should be checked on every boot.**
`VLLM_DEBUG_WORKSPACE=1` is carried in production for exactly this: upstream's own `WorkspaceManager`
prints `Resized workspace ... 0.00 MB -> 513.00 MB`, once per rank, independently of anything our
patch claims about itself. **More than one resize line on a rank, or a figure other than 513.00 MB,
means the buffer grew after `lock_workspace()` and the pool you just read is not the one you will
keep.** Ours reads one line per rank on the load boot, on the systemd-started engine, on the clean
boot after a whole-cluster reboot, and again after a 969,468-token request and eight concurrent ~128K
lanes.

A pool within a few percent of ours is a match. A pool well below it usually means one of: no fp8
draft cache (`HAREM_DRAFT_KV_DTYPE`), `--block-size` left at a large value,
`--max-num-batched-tokens` raised, the workspace bound not armed (`HAREM_INDEXER_WS_MODE`), or —
most likely — **a boot taken without the memory settle gate**. That last one is not a real difference
in your machine; it is the instrument. See [docs/07](../docs/07-kv-and-draft-page.md) §1.1.

<details>
<summary>The 0.85 verdict this section used to carry, and what replaced it</summary>

The table here used to end with a rejected `0.85 | 5,256,198` row and the sentence **"0.85 will not
be attempted on this stack"**, both resting on `MemFree` against a 4 GiB floor, on a boot that
predates the fast-load page-cache fix. Both are `[retracted]` (§6). The ladder was climbed rung by
rung on 6 September against swap **traffic** instead: 0.85, 0.87 and 0.88 all pass, 0.90 is rejected,
0.87 shipped as production 11 and **0.88 shipped as production 12** hours later — the same
measurement, a different decision about what the cluster is for
([`results/memory/ladder-6sep.md`](../results/memory/ladder-6sep.md), [docs/11](../docs/11-open-issues.md) §2.4).
**0.89 was never measured**, and the workspace bound does not change that: it gives memory back on
the GPU side, not the host side.

</details>

**Read the pool from a boot whose baseline had settled.** vLLM sizes the pool from a difference
between two `/proc/meminfo` readings minutes apart, and it runs backwards: a node that starts with
less memory free awards itself a *larger* pool. Ours was polluted by 9 GiB on the last node started
until `SETTLE_MIN_GIB` was added.

### gates — expect 10/10 and 12/12, cold **and** warm `[measured-here]`

```
python3 ../scripts/correctness-probe.py $API
python3 ../scripts/code-exam.py $API
```

| | Expected (production 12) |
|---|---|
| correctness probe | **10/10**, `requests with EMPTY content: 0` |
| code exam | **12/12** |

Production 12 read both full **cold and warm on the load boot, on the systemd-started engine, and on
the clean boot after a whole-cluster reboot** — three independent boots, six readings. Two gates the
script does not run read full on the same boots and are worth borrowing if you have them: tool-call
**8/8** and needle-lite **6/6** at 64K and 128K.

**Run both twice: once cold after the boot, and again after the full benchmark.** The warm run is the
one that carries weight. The class of defect this stack has actually produced — a kernel writing rows
nothing initialises, and a combine summing them — **hides on a fresh engine**, because a fresh
caching allocator hands out zeroed pages ([docs/09](../docs/09-measurement-protocol.md) §5).

This gate is not cosmetic. A decode kernel that silently computes the wrong thing for our head count
produced fluent, confident nonsense here while the engine reported no error at all, and arms with a
mis-shaped kernel scored 0/12 to 10/12 on the code exam while still reading fluently. **If you score
below full marks on a warm engine, stop and find out why before you measure anything else.**

Note the code exam executes model-written Python locally, in a temp file, with a 25-second timeout
and no sandbox.

### boot gates — read these before the numbers

A production boot prints four things that tell you the engine built the shape you asked for. If any
is missing, nothing measured afterwards means anything:

| Gate | Expected |
|---|---|
| `[padload]` line | present |
| Patch anchors | **10** matched |
| Pad audit | `assert 5` passes |
| `CUDA_EXL3_DEBUG_NAMES` tally | **203 EXL3 / 113 bf16**, nothing heavy in the "→ unquantized" list |

Production 11 and 12 add two more, one for each configuration's guard patch. Both must appear on
**every** rank, on a fast-load boot and on a cold boot alike — the prelude applies them either way:

| Gate | Expected | Came with |
|---|---|---|
| sm_12x correctness set | `SM12 items=pdl,kpool` | production 11 |
| Indexer workspace bound | the `HAREM-IDXWS bound` line, `headroom 2.03x`, `saved 4.42 GiB` | production 12 |
| Workspace resize, upstream's own line | **exactly one** per rank, `0.00 MB -> 513.00 MB` | production 12 |

### c1 — cold/warm single stream `[measured-here]`

| | Expected (production 12) | Band | Pass range | Production 9 `[history]` |
|---|---|---|---|---|
| C1 aggregate | **69.72** tok/s | ±4 % | **66.93 – 72.51** | 69.9 |
| C1 per stream | **75.55** tok/s | ±4 % | 72.53 – 78.57 | 75.9 |
| TTFT at C1 | **0.25** s | — | — | 0.280 |
| Draft acceptance | **~62.5 %** — the gate is ≥ 60 % | — | — | ~61.9 |
| Accepted tokens per step | **~5.35** | — | — | ~5.34 |

Production 12's C1 is production 11's to within **+0.08 %** pooled over six rounds and two boots, so
a reading anywhere in that pass range grades the same against 9, 10, 11 or 12. The single-boot
readings behind the pool were 69.63 on the load boot and 72.77 on the clean boot — a 4.5 % spread at
the *least* noisy level, which is why the headline is a pool and not a boot.

**A note on acceptance, because we got this wrong on our own front page.** The C1 median reads
61.9 % on production 9 against 64.4 % on production 8, and we published that 2.4-point gap as the
price of the full-scope checkpoint. **It is an artefact of our own harness** `[retracted]`:
`bench-sweep.py` cycles `prompts[i % 12]`, so C1 and C2 see only the first **8** of the 12 prompts
while C4–C8 see all twelve, and the two groups differ by 8 points of acceptance. Pooled by draft
token across all five levels and three independent boots, production 9 is **62.27 %** against
production 8's **62.09 %** — **+0.18 points**, well inside that arm's own ±1.4-point boot spread. A
700-token cold probe reads identically on both arms (42.53 % against 42.51 %).

Two things follow for anyone comparing acceptance numbers. **Pool by draft token across levels**, not
by taking a per-level median. And note that `accept_len = 1 + k × acceptance` holds on all 90 of our
rows to ±0.005 — so "acceptance −2.4 points" and "tokens per step −3 %" were never two costs, they
are **the same number written twice**.

The first of the three requests is genuinely cold if you run this right after a boot; it will be
lower, and that is the point of running it first. If you run the audit against an engine that has
been serving for a while, the "cold" line is not cold.

**Do not compare any of this against a synthetic "count to 200" number.** On synthetic prompts the
drafter accepts almost everything and the same engine runs far faster. That is the ceiling, not the
working speed.

### concurrency — `[measured-here]`

| Level | **Production 12** | Band | Pass range | Production 9 `[history]` |
|---|---|---|---|---|
| C1 total | **69.72** | ±4 % | **66.93 – 72.51** | 69.9 |
| C2 total | 101.20 | ±6 % | 95.13 – 107.27 | 99.17 |
| C4 total | 146.14 | ±9 % | 132.99 – 159.29 | 140.72 |
| C6 total | 176.00 | ±6 % | 165.44 – 186.56 | 172.40 |
| C8 total | **196.06** | ±3 % | **190.18 – 201.94** | 197.2 |
| C1 / C8 per stream | 75.55 / 28.44 | — | — | 75.9 / 28.6 |
| TTFT C1 / C8 | 0.25 / 0.80 s | — | — | 0.280 / 0.826 |

<details>
<summary>Production 9 against production 8, the comparison this table used to hold</summary>

| | Production 9 | Production 8 |
|---|---|---|
| C1 total / per stream | 69.9 / 75.9 | 56.9 / 62.4 |
| C8 total / per stream | 197.2 / 28.6 | 175.4 / 26.7 |
| TTFT C8 | 0.826 s | 0.906 s |

That step — **+22.9 % at C1, +12.5 % at C8** — is the full-scope checkpoint, and it is the largest
single move this stack has made. Everything between production 9 and production 12 is memory: three
memory fractions, two guard patches and one buffer bound, and none of them moved C1 or C8 outside a
tenth of a percent.

</details>

**Pool six rounds over two boots, and read the verdict off C1 and C8.** Three rounds of one boot is
enough only **because the MLA tuner cache is persisted and warm**; on an image without one the rule is
still five rounds with the first two discarded, because this stack's tuner warm-up has made a winning
arm look 25–45 % worse on the first pass ([docs/12](../docs/12-tuner-cache.md),
[docs/09](../docs/09-measurement-protocol.md)).

**C4 and C6 will disagree with this table on a single boot, and that is not a finding.** Production
12's own two boots read C6 at 167.8 and 177.4 (5.7 % apart) and C4 at 140.8 and 151.6 (7.7 %), while
C1 and C8 pooled to within a tenth of a percent of the reference. Production 11 made the same point
from the other side and we published a 6.4 % C4 "cost" that a repeat deleted (§6). Boot-to-boot spread
on this stack has reached **16 % at C8** on older images. A single pair of rounds is not evidence for
anything; we published a kernel conclusion drawn from one pair and had to withdraw it (§6).

### prefill — measure it on a prompt the engine has never seen `[measured-here]`

| | Expected (production 12) | Pass range (±3 %) | Production 9 `[history]` |
|---|---|---|---|
| Fresh, unseen ~8.5K prompts | **1,744** tok/s — 1,737 on the load boot, 1,750 on the clean boot | **1,692 – 1,796** | 1,738 |
| Warm, repeated 7K prompt | 1,622 / 1,632 tok/s | — | 1,575 |

**A prefill number measured on a repeated prompt is not a prefill measurement.** The second run reads
whole blocks out of the prefix cache and overstates by up to 55 %. `scripts/prefill-7k.py` reports
the warm number and says so; `bench/prefill-fresh.py` draws a new seed per request and is the one to
quote.

Prefill has not moved since production 9: production 8, 9, 10, 11 and 12 all sit inside the ±3 %
equality band. The full-scope checkpoint bought its gain in **decode step time**, not in prefill, and
the three memory configurations after it bought pool, not speed.

### category speed — `[measured-here]` on production 12

**Re-measured on production 12 on 7 September**, with the same prompt set and the same script as the
production-9 row, one warm-up round plus three measured rounds on a warm engine with no boot inside
the window
([`../results/speed/category-speeds-production-12.md`](../results/speed/category-speeds-production-12.md)):

| Category | C1 mean decode | C1 acceptance | C4 total | Production 9 `[history]` |
|---|---|---|---|---|
| code | **61.5** tok/s | 46 % | 115.2 tok/s | 61.7 · 46 % · 116.1 |
| math | **76.2** tok/s | 57 % | 120.6 tok/s | 79.6 · 58 % · 129.4 |
| JSON | **73.1** tok/s | 53 % | 108.8 tok/s | 72.8 · 54 % · 110.1 |
| **prose** | **29.0** tok/s | **13 %** | 52.2 tok/s | 29.1 · 13 % · 50.7 |

Every category is inside its own round-to-round spread against production 9 — math's −4.3 % against
an 8.4 % round spread is the largest — so the three memory rungs and the workspace bound between the
two arms bought **+37.8 % of KV pool at no cost in tokens per second**. The production-7 lineage read
code 47.9 · math 59.0 · JSON 57.7 · prose 22.4, so every category gained 30–35 % at production 9 and
has held it since.

**The prose row is the honest headline of this whole stack**, and it is not a setting: the DFlash2
drafter barely fires on free prose, so prose falls back to roughly the unspeculated rate while code,
math and JSON get 2–2.8×. Acceptance on **non-English** prose is worse still, 10–13 %. That is a
property of the drafter's training distribution, and nothing on our side fixes it. Every speed table
in this repository was measured with **English** prompts; read them as English-workload numbers.

Note that this run came from a standalone category script, **not** from the sweep record — the
quick-arm harness the production 8 and 9 comparisons ran does not include a category step at all
([docs/14](../docs/14-troubleshooting.md) §8.12).

### memory — `[measured-here]`

**Host memory is the pass/fail section on this stack, and the rule is swap traffic, not free RAM.**
At `gpu-memory-utilization 0.88` (**production 12**), sampled with `vmstat -n -t 1` on all three
nodes for the whole window between "engine up" and "battery done" — 1,765 s on the arm that also ran
both long-context stress cases:

| | head | worker-1 | worker-2 |
|---|---|---|---|
| **Swap in**, summed over the window | **0** | **0** | **0** |
| Swap out, stress arm | 1,340 KiB | 5 KiB | 10 KiB |
| — in how many samples | 8 of 353, longest unbroken run **10 s** | 1 | 1 |
| Swap out, clean boot after reboot | 5 KiB | 10 KiB | 10 KiB |
| Swap **used**, clean boot | **0.000 GiB** | 0.000 | 0.000 |
| `MemAvailable` min, stress arm | **1.52 GiB** | 3.37 | 3.36 |
| `MemAvailable` min, clean boot | 3.15 GiB | 4.49 | 4.45 |
| OOM killer | **0** | 0 | 0 |
| Consumed per node (weights + non-torch) | **54.62** | **54.48** | **54.28 GiB** |

**The thresholds, as the ladder actually applied them.** A rung passes on **three conditions
together** ([`results/memory/ladder-6sep.md`](../results/memory/ladder-6sep.md) §1):

1. **Swap traffic ≈ 0 under load** — `si` and `so` per second from `vmstat -n -t 1`, summed over the
   benchmark window on every node. Not swap *used*: that is a stock, it sits at ~0.04 GiB from boot
   at **every** rung including the one that failed, and it discriminates nothing.
2. **C1 inside ±4 % and C8 inside ±3 %** of a same-session reference.
3. **Both gates full, cold and warm.**

**What "≈ 0" means in practice, with both ends of the scale.** Production 12's stress arm paged out
1,340 KiB on the head node — 134× its own 0.87 reference's 10 KiB — and passed, on two facts: swap-in
was **exactly zero in every sample on every node**, so no page was ever asked for again, and the
traffic occupied 8 of 353 samples with a longest unbroken run of **10 s**. The rejected 0.90 rung, for
scale, paged **1,519 MiB out and 143 MiB back in**, in 250 of 598 samples, with an unbroken run of
**85 s**. Between those two there is no ambiguity. **If your swap grows through the rounds, or reads
back at all, step `gpu-memory-utilization` down and re-audit.**

**`MemFree` is not the ruler, and `MemAvailable` is a budget rather than a gate.** This page used a
4 GiB `MemFree` floor until 6 September; it is `[retracted]` (§6), because on a unified-memory part
most of what the kernel holds at that moment is reclaimable page cache. `MemAvailable` is the honest
number, but it does not decide a rung — it tells you what is left for **anything running beside the
engine**. At 0.88 that budget is about **2 GiB**, which is why a profiling run on this cluster now
has to stop the engine first. Every 1 % of `gpu-memory-utilization` costs about **1.2 GiB** of host
headroom on this hardware, so the same arithmetic sizes any rung you are considering.

**0.90 is rejected, 0.89 is unmeasured, and throughput will not warn you.** At 0.90 every concurrency
level is still inside its band while the head node pages 1.5 GB out and reads 143 MB back; what
surfaces at the client is the arm's **first prefill going 5.0 → 9.8 s**, not a lower median. A ladder
judged on tok/s would have taken that rung. 0.89 was never measured, and the indexer workspace bound
does not change it: that patch gives memory back on the **GPU** side, not the host side.

<details>
<summary>The same measurements at 0.80 and 0.83, kept for older commits</summary>

At `gpu-memory-utilization 0.80` (**production 9**), engine idle:

| | head | worker-1 | worker-2 |
|---|---|---|---|
| Free | 12.1 GiB | 13.5 GiB | 13.4 GiB |
| Swap used | ~0.1 GiB | ~0.1 GiB | ~0.1 GiB |

At `gpu-memory-utilization 0.83` (**production 10**), after a full benchmark:

| | head | worker-1 | worker-2 |
|---|---|---|---|
| `MemAvailable` | 8.8 GB | 10.0 GB | 10.1 GB |
| `MemFree` | 0.9–1.2 GiB, reclaimable page cache | | |
| Swap used | ~0.11 GiB | ~0.09 GiB | ~0.08 GiB, flat through the rounds |

At **0.87** (production 11), `MemAvailable` min under load was 3.40 / 4.66 / 4.68 GiB with swap
traffic exactly zero on the clean boot. Consumed memory per node was **58.3–59.1 GiB** on production
9, 10 and 11 alike; production 12's 54.3–54.6 is the 4.42 GiB the workspace bound releases, measured
on the far side.

</details>

`vm.swappiness` should read **60**. Do not set it to 0 — see
[docs/00](../docs/00-hardware-and-os.md) §9.1 for the incident that cost three power cycles.

### boot time — `[measured-here]`

| | Expected (production 12) | Production 9 `[history]` |
|---|---|---|
| Container start → API ready, fast-load boot | **272 s** (weights 80 s) | 251 s (weights 58 s) |
| `systemctl start` → `/health` 200 | **205 s** | — |
| **Power-on to `/health` 200**, all three rebooted together, autostart enabled | **311 s** by the wall clock | — |
| The one-off dump boot that produces the sidecar | **590 s** | 620 s |
| Without a fast-load sidecar at all | ~618 s | ~618 s |

The power-on figure is the planning number and it has not moved: 311 s at production 12 against
312 s at production 11, by the same wall clock. Note that production 10's equivalent is published
twice — 242 s by the harness counter and 315 s by the wall clock in the same log — and the larger is
the one to plan on (§5).

---

## 3. If the audit disagrees with this page

In this order, because each step rules out the ones after it:

1. **Versions and fabric.** `x2` instead of `x4`, or two flat ports per node, means you are measuring
   a different machine. Nothing below matters until those are right.
2. **Correctness.** If the probe or the code exam fails, compare your image and patch tree against
   `patches/` — every failure we have seen here traced to a missing patch, not to a setting.
3. **The KV pool**, from the start-up log. It tells you whether the engine built the shape you asked
   for, before any timing is involved. Check the settle gate before you believe a small pool, and
   check the workspace resize line — one per rank at 513.00 MB — before you believe a large one.
4. **Memory.** A machine that is swapping produces speed numbers that mean nothing. Read swap
   *traffic* under load, on every node, not `MemFree` and not swap used.
5. **Speed**, and only against the matching row, with the matching concurrency, the matching prompt
   category, and the matching number of sweep rounds.

If it is a failure rather than a disagreement, [docs/14](../docs/14-troubleshooting.md) indexes them
by symptom with the exact log line.

---

## 4. What this audit does **not** check

Named so you do not mistake silence for a pass:

- **Quality beyond the two gates.** The MMLU sample (1,995 questions) takes hours and is not in the
  script. Ours reads **86.47 ±0.74** at TP=3, measured on production 9 and carried forward through
  10, 11 and 12 `[not tested]` on each.
- **Long context.** `max_model_len` is 1,000,000 and the pool supports about **7.0** such requests at
  production 12, but the audit sends nothing near that. Production 12 was promoted on two cases the
  script does not run — one 969,468-token request, correct in 569.6 s, and eight concurrent ~128K
  lanes, 8/8 with 640,904 prompt tokens in 227.5 s — because a bounded indexer workspace is only safe
  if the bound holds under exactly those. **If you change `max_num_seqs` or the K-pool size, that
  bound is recomputed at boot and these are the two cases to re-run.**
- **Anything at max reasoning effort.** See §1.
- **Numerical equivalence against BF16.** We never ran it.

---

## 5. Provenance — every headline number, and where it comes from

| Claim | Value | Config | Date | Traces to |
|---|---|---|---|---|
| C1 total / per stream | 69.9 / 75.9 tok/s | prod 9 | 5 Sep 2026 | `results/speed/concurrency-sweeps.csv` |
| C8 total / per stream | 197.2 / 28.6 tok/s | prod 9 | 5 Sep 2026 | `results/speed/concurrency-sweeps.csv` |
| TTFT C1 / C8 | 0.280 / 0.826 s | prod 9 | 5 Sep 2026 | `results/speed/concurrency-sweeps.csv` |
| Draft acceptance · accepted/step | 61.9 % · 5.34 | prod 9 | 5 Sep 2026 | `results/speed/concurrency-sweeps.csv` |
| Prefill, fresh | 1,738 tok/s | prod 9 | 5 Sep 2026 | `results/speed/category-prefill-and-mixed-load.md` |
| KV pool | 5,165,289 tokens | prod 9 @ 0.80 | 5 Sep 2026 | `results/boot/boot-ledger.md`, engine log |
| Consumed memory per node | 58.3–59.1 GiB | prod 9 | 5 Sep 2026 | `results/boot/boot-ledger.md` |
| Quality gates | 10/10 · 12/12 cold and warm | prod 9 | 5 Sep 2026 | `results/gates/quality-gates.md` |
| MMLU sample, 1,995 q | 86.47 ±0.74 | prod 9, TP=3 | 5 Sep 2026 | `results/gates/quality-gates.md` |
| Cold boot | 251 s (weights 58 s) | prod 9, fast-load | 5 Sep 2026 | `results/boot/boot-ledger.md` |
| Step breakdown, prefill / C1 / C8 | see `charts/step-breakdown-prod9.svg` | **prod 9** | 5 Sep 2026 | `results/profile/step-breakdown.csv` |
| Step breakdown, the production-7 control | see `docs/10` §5 | prod 7 | 5 Sep 2026 | `results/profile/measured-prod7.md` |
| Production 10 (0.83 rung), boot `L083` | KV 5,619,834; C1 70.5 / C4 144.6 / C8 194.0; prefill fresh 1,769; TTFT 0.282 / 0.811 s | prod 10 | 5 Sep 2026 | `results/configs/kv-pool-progression.csv`, `results/speed/concurrency-sweeps.csv` |
| Production 10, host memory under load | `MemAvailable` 8.8 / 10.0 / 10.1 GB; swap ~0.1 GB flat | prod 10 | 5 Sep 2026 | `results/configs/kv-pool-progression.csv` |
| `NCCL_ALGO=Ring,Tree`, five-round engine arm, boot `ALG5` | C1 70.6 / C4 143.4 / C8 195.6, all within ±1 % of `Ring`; KV 5,702,479 (+1.5 %) | prod 10 | 5 Sep 2026 | `results/mesh/algo-sweep.md`, `docs/06` §12.3 |
| Boot from power-on, all three rebooted, autostart enabled | units finished +98/+98/+103 s; `/health` 200 at +242 s by the harness, **+315 s** by the log's wall clock; KV 5,652,892; gates 10/10 · 12/12 | prod 10 | 5 Sep 2026 | `results/boot/boot-ledger.md`, `systemd/README.md` |
| **Production 11** — configuration 10 at 0.87 **plus** the sm_12x correctness set (`HAREM_SM12_ITEMS=pdl,kpool`) | KV **6,382,920**; C1 **69.6 / 74.7**; C8 **201.1 / 29.5**; prefill fresh 1,760; TTFT 0.278 / 0.806 s; acceptance 61.3 % · 5.29; boot 271 s (`docker run` → API ready), 590 s for the dump boot the patches force. Against a same-session 0.83 reference: KV **+12.1 %**, C1 +0.6 %, C8 +1.8 %, C4 −2.2 % | prod 11 @ 0.87 | 6 Sep 2026 | `results/configs/production-configurations.csv` row 11, `results/speed/concurrency-sweeps.csv` |
| Production 11, host memory and swap **traffic** under load | `MemAvailable` min **3.40 / 4.66 / 4.68 GiB** against 8.28 / 9.05 / 9.57 at 0.83; swap traffic `si`+`so` **exactly 0** on all three nodes on the clean boot, and 0.08 / 0 / 0.02 MiB on the load boot against the reference's own 0.7 / 0.9 / 0.4 MiB floor | prod 11 | 6 Sep 2026 | `results/configs/production-configurations.csv` row 11, `results/memory/ladder-6sep.md` |
| Production 11, quality and the whole-cluster reboot | Gates **10/10 · 12/12 cold and warm on both boots**, tool-call **8/8**, needle-lite **6/6** at 64K and 128K; `/health` 200 at **312 s** from the reboot command by wall clock against production 10's 315 s by the same clock, ConnectX-7 4/4, KV 6,382,920, swap traffic zero. MMLU is production 9's 86.47 ±0.74, carried forward and `[not tested]` here | prod 11 | 6 Sep 2026 | `results/configs/production-configurations.csv` row 11, `results/boot/boot-ledger.md` |
| The memory ladder, four rungs in one session against a same-session 0.83 reference | KV pool 0.83 **5,674,931** · 0.85 **6,016,528** (+6.0 %) · **0.87 6,363,636** (+12.1 %, **shipped**) · 0.88 **6,542,699** (+15.3 %, passed and not taken) · 0.90 **6,870,523** (+21.1 %, **rejected**). No speed number at any rung is outside its band — including the rejected one; gates full at every rung; OOM killer 0 everywhere | one line per rung, same tree | 6 Sep 2026 | `results/memory/ladder-6sep.md` |
| Why 0.90 was rejected, and what a rung costs | Head node **1,519.4 MiB paged out and 142.6 MiB read back**, 250 of 598 load seconds non-zero, longest unbroken run 85 s, swap in use at the end 2.65 / 0.57 / 0.52 GiB against 0.03 at the reference. What surfaces at the client is the arm's **first prefill going 5.0 → 9.8 s**, not a lower median. `MemAvailable` min on the head node across the five rungs: 8.35 / 5.99 / **3.49** / 1.86 / 1.04 GiB — every 1 % of the fraction costs about **1.2 GiB** of host headroom | one line per rung, same tree | 6 Sep 2026 | `results/memory/ladder-6sep.md` §3–§4 |
| The per-node memory ledger, read with the engine left running | Engine CUDA allocation **99.06 GiB** = weights 51.62 + KV pool 40.12 + non-torch 7.28 + CUDA graph **0.00**; host anonymous 5.60; page cache 5.80; slab 2.36; free 5.08; residual **3.49**, which reads **3.12** with the engine down and is therefore the driver's fixed reserve. Genuinely unavailable at idle: **4.84 GiB** of 121.63 | prod 10 @ 0.83, read-only, engine up | 6 Sep 2026 | `docs/17-memory-ledger.md` §2 |
| What a KV block is, and the KDA speculation slots inside the divisor | Block = 3,328 tokens, **20,934,400 B**: MLA latent fp8 **89.53 %**, indexer k fp8 5.77 %, DFlash2 draft 4.70 %. A 1M-token request costs **363 blocks = 7.078 GiB on every node** and the pool holds **5.67** of them. Under `mamba_cache_mode=align` every KDA layer holds **2 + 7 = 9** state slots per request — **36 blocks, 9.9 % of the divisor at TP=3 and 12.9 % at TP=2**, 5.61 GiB (14 % of the pool) at 8-way concurrency. Taking the slots to two projects **+8.0 %** (TP=3) and **+9.6 %** (TP=2) `[estimate]`, against an arithmetic worst case of C8 197 → about 168 tok/s; **not run** `[not tested]` | prod 10 @ 0.83 | 6 Sep 2026 | `docs/17-memory-ledger.md` §3–§5 |
| **Production 12 candidate** — the sparse indexer's K-gather workspace, bounded | Locked workspace **5,036.40 → 513.00 MB** (−4.42 GiB); KV pool **6,289,256 → 6,933,884, +10.25 %** at an unchanged memory fraction. C1 69.69 → 70.69, C8 199.76 → 196.81, prefill fresh 1,794 → 1,778, TTFT equal — every level inside its band. Gates 10/10 · 12/12 cold and warm on both arms, tool-call 8/8, needle-lite 6/6, one **969,468-token** request correct in 572.4 s, **eight** concurrent long-context lanes (640,904 prompt tokens) with every needle correct, swap 0.000 GiB, no safety layer fired. Both arms are eager boots, so the production pool with fast-load restored was an `[estimate]` of ≈7.03 M here — **settled by the promotion the same day at 7.04–7.17 M**, the row below | prod 11's 0.87, fast-load off, one line between the arms | 6 Sep 2026 | `results/memory/indexer-workspace-ab.md`, `docs/17` §2.5 |
| **Production 12 — shipped, and what §2 grades against** — configuration 11 at 0.88 **plus** the indexer workspace bound (`HAREM_INDEXER_WS_MODE=bound`) | KV **7,041,322** on the reboot boot, three boots reading 7,170,798 / 7,088,154 / 7,041,322; C1 **69.72 / 75.55**; C8 **196.06 / 28.44**; C2 101.20, C4 146.14, C6 176.00; prefill fresh 1,744 (1,737 load, 1,750 clean); TTFT 0.25 / 0.80 s; acceptance 62.5 / 62.4 % · 5.35 per step; consumed per node **54.62 / 54.48 / 54.28 GiB**, available KV 50.75 / 50.89 / 51.09 GiB, locked workspace 513.00 MB at exactly one resize per rank. Against a same-session 0.87 reference: KV **+10.3 %** (+12.3 % on the best boot), **C1 +0.08 %, C8 −0.08 %** — every level inside its band. Boot 272 s fast-load, 205 s from `systemctl start`, **311 s** from power-on with all three rebooted together, 590 s for the dump boot the patch forces | prod 12 @ 0.88, pool of six rounds over two boots | 6 Sep 2026 | `results/configs/production-configurations.csv` row 12, `results/memory/indexer-workspace-ab.md` §7.1 |
| Production 12, host memory and swap **traffic** under load | Swap **in exactly 0** in every sample on every node, in both windows; swap out **1,340 / 5 / 10 KiB** over the 1,765 s stress window (8 of 353 samples, longest unbroken run 10 s) against the reference's own 10 / 5 / 5, and 5 / 10 / 10 KiB with swap used **0.000 GiB** on the clean boot. `MemAvailable` min **1.52 / 3.37 / 3.36 GiB** on the arm that also ran both long-context stress cases, **3.15 / 4.49 / 4.45** on the clean boot. OOM killer 0 on all three | prod 12 | 6 Sep 2026 | `results/configs/production-configurations.csv` row 12, `results/memory/ladder-6sep.md` |
| Production 12, quality and the whole-cluster reboot | Gates **10/10 · 12/12 cold and warm** on the load boot, on the systemd-started engine and on the clean boot after a whole-cluster reboot; tool-call **8/8**; needle-lite **6/6** at 64K and 128K. Long-context stress the 0.87 reference never ran: one **969,468-token** request correct in **569.6 s**, and **eight** concurrent ~128K lanes 8/8, 640,904 prompt tokens in 227.5 s (2,817 tok/s aggregate prefill), each lane carrying its own needle. After both, every rank still logs exactly one workspace resize and zero assertions — none of the patch's four safety layers fired. MMLU is production 9's 86.47 ±0.74, carried forward and `[not tested]` here | prod 12 | 6 Sep 2026 | `results/configs/production-configurations.csv` row 12, `results/boot/boot-ledger.md` |
| Quality battery on production 12 — three benchmarks against the NVFP4 sibling recipe | **GSM8K 97.5 %** against 94.0, **IFEval 80.0 % prompt / 86.0 % instruction** against 78.9 / 85.1, **tool-eval-bench 85.5 ±1.3** against 87.8 ±0.9. The tool-eval difference is real (Welch t = 4.02, permutation p = 0.0048) and is **four scenarios out of 88** — remove TC-51, TC-21, TC-74 and TC-87 and this stack is +0.22 ahead on the other 84; nine of fourteen categories are identical digit for digit; `max_points` 176 in both files. The chat-template explanation was tested as its own engine arm and **refuted**. The **1M needle and the full 14,042-question MMLU were deferred on time and did not run** `[not tested]` | prod 12, harness `2.6.1.dev39`, same seed, temperature 0, effort low, one engine, no restart between tests | 6–7 Sep 2026 | `results/gates/quality-battery-production-12.md`, `results/gates/quality-gates.md` |
| Category speed on production 12 | C1 code **61.5** · math **76.2** · JSON **73.1** · prose **29.0** tok/s, acceptance 46 / 57 / 53 / **13 %**; C4 total 115.2 / 120.6 / 108.8 / 52.2. Every category inside its own round-to-round spread against production 9, so **+37.8 % of KV pool cost no category any tokens per second**. The step rate is the same in all four columns (14.58–15.55 /s, 6.4 % spread) — the whole 2.6× spread is accepted tokens per step, which is the drafter's hit rate | prod 12, single stream and four in parallel, one warm-up plus three measured rounds, warm engine, no boot in the window | 7 Sep 2026 | `results/speed/category-speeds-production-12.md` |
| **TP=2 production candidate B** (full scope), against candidate A (routed experts only) | KV pool at 1M **2,128,571** against 1,500,000 (**+41.9 %**); C1 **58.50 / 62.55** against 48.76 / 54.72; C8 **155.75** against 137.41; TTFT 0.407 / 1.077 s; consumed per node **84.8 GiB** against 89.3 (**−4.5**); prefill fresh 1,400 against 1,444 (equal); boot 272 s on both; gates 10/10 · 12/12 cold and warm, tool-call 8/8, needle-lite 6/6 on both; MMLU **86.02 ±0.75** against 86.37 ±0.74, inside one error bar | **two** nodes, TP=2, EP off, `exl3-zeus:754421f`, 0.85, median of rounds 2–4 of four | 6 Sep 2026 | `results/speed/tp2-production-candidate.md`, `docs/15` §5 |
| **TP=2 candidate C** (candidate B **plus** the indexer workspace bound), against a same-session control | Locked workspace 5,036.40 → **513.00 MB** on both ranks — the three-node numbers to the decimal, because no term of the bound knows the rank count. KV pool **1,800,000 → 2,378,571, +32.14 %** eager, and **2,128,571 → 2,692,857, +26.5 %** once the sidecar is back; predicted 2,689,285 from candidate B's own eager→fast-load difference, measured **+0.13 %** off. Consumed per node 84.7/85.0 → 79.5/80.4 GiB. Gates 10/10 · 12/12 cold **and** warm on both arms, tool-call 8/8, needle-lite 6/6; one **969,468**-token request correct, eight concurrent ~128K lanes **8/8** (640,904 prompt tokens, 2,225 tok/s prefill), one resize per rank after the stress, none of the four safety layers fired, swap 0.000 GiB. Speed: every level inside its band, **but all five the same sign** (−1.16 … −2.63 %, mean −2.0 %) where three ranks read mixed signs — **recorded as unexplained**, no clock/temperature/power telemetry was sampled and the arm order was fixed. The 1M gate first read FAIL on an **empty** answer (never a wrong one) because the probe scored `content` alone; re-run identical, it passed, and the harness now scores both fields | **two** nodes, TP=2, EP off, `exl3-zeus:754421f`, 0.85, one environment line between the arms, both eager; production row fast-load | 6 Sep 2026 | `results/speed/tp2-production-candidate.md`, `docs/15` §5.9 |
| The draft KV page at two ranks (`HAREM_SW_BLOCK_SIZE=256`) | KV pool **601,562 → 1,303,571, +116.7 %** (+109.6 % normalised); C8 127.54 → 135.59 (+6.3 %); TTFT −23 % at C1 and −27 % at C8; C1 aggregate equal; gates full on both arms. **The headline is the cliff, not the pool**: without it a 6,253-token prompt is never scheduled at all (`Running: 0, Waiting: 1`), with it 8,268 tokens serve in 6.3 s at 1,478 tok/s fresh prefill. The drafter takes **60.2 %** of the blocks-per-request divisor at TP=2 against 53 % at TP=3. Cost: +9.1 % memory per block, acceptance −1.9 points | **two** nodes, TP=2, EP off, `exl3-zeus:62f53e6`, experts-only `b20c49ba`, 0.85, three rounds, one boot each | 6 Sep 2026 | `results/speed/tp2-draft-page.md`, `docs/15` §4 |
| KDA/MLA quantization gate bench, **cold, on the target GPU** — **this is the live row** | KDA arms EXL3/bf16 **1.023** at M=8 (neutral, +0.050 ms/step); `kv_b_proj` **0.291×**; three families together **+1.368 ms/step** against a 1.5 ms gate; prefill not re-measured | model-free, **GB10**, both arms rotated over ≥ 4× L2 | 5 Sep 2026 | `results/kernels/kda-gate-bench-gb10.md`, `results/kernels/gb10-coldbench/`, `bench/kda_gate_bench_gb10.py` |
| KDA/MLA quantization gate bench, first pass | KDA arms EXL3/bf16 1.58–1.76× at M=8; whole family 0.851 ms of a 72.5 ms step; `kv_b_proj` 0.391× but no batched kernel exists — **the ratios are `[retracted]`, warm; the file is kept as published** | model-free, **workstation GPU** | 5 Sep 2026 | `results/kernels/kda-gate-bench.md`, `bench/kda_gate_bench.py`, `docs/11` §1.11 |
| Device read bandwidth ruler | 225.2 GB/s | model-free | 5 Sep 2026 | `bench/bw.py`, `results/profile/step-breakdown.md` |
| BF16 GEMM ruler | 97.3 TFLOP/s | model-free | 5 Sep 2026 | `bench/gemmpeak.py`, `results/profile/step-breakdown.md` |
| Mesh all-reduce, channel sweep | see `results/mesh/` | model-free | 4–5 Sep 2026 | `results/mesh/all-reduce-sweep.md` |
| Multilink sweep, 9 arms | see `results/mesh/` | model-free | 5 Sep 2026 | `results/mesh/multilink-sweep.csv` |
| Fabric raw, `ib_write_bw` | 98.0 Gb/s per link | model-free | 29 Aug 2026 | `docs/00` §4.4 |
| PCIe link state | `32GT/s, Width x4`, 12/12 | — | 5 Sep 2026 | `docs/00` §4.3 |
| sm_12x stack patches — the PDL race question, by logit divergence | Between-arm p95 **6.2–6.5e-02** against a within-arm pooled floor p95 of **7.7–8.0e-02**: two arms differ *less* than two runs of one arm. Between-arm max 3.028e-01 = **3.9×** the floor against a K = 4 threshold, **no outlier prompt**, and in the first comparison the arms diverge in token sequence later (position 5) than one arm does from itself (position 3). **No race detected** — on one prompt set, 128 generated tokens, concurrency 1 | diagnostic image `exl3-zeus:e7e345e-dflash`, separate in-container tree, TP=3+EP, 0.83, `CUDA_EXL3_DETERMINISTIC=1`, three rounds; production 10 never modified | 6 Sep 2026 | `results/kernels/sm12-stack-patches-ab.md` §4 |
| sm_12x stack patches — what the four items cost | PDL on against off, six rounds pooled against three: C1 69.54 / 70.12 (**+0.8 %**), C8 197.85 / 199.19 (**+0.7 %**), both in band. The one reading outside its band — C1 per-stream at +5.4 % — was deleted by a repeat: 69.75, then **76.17** on the identical configuration. Items 2–4 on against off: C1 −1.1 %, C8 +0.9 %, prefill-fresh −0.9 %, KV pool −0.3 %, so **the per-chunk memset cost the preparation note priced did not appear**. Gates 10/10 · 12/12 in every arm with one warm 11/12 flake that did not reproduce in three re-runs; KV pool across the five arms 5,666,666–5,694,214, **0.5 %** | as above | 6 Sep 2026 | `results/kernels/sm12-stack-patches-ab.md` §5–§6 |
| sm_12x — what was adopted with production 11, and what was not | Items **2 + 3** (`patch-kpool-init.py`) and item 1 **as PDL off** (`patch-pdl-gate.py`) moved out of `patches-optional/sm12/` into the track's own patch tree behind `HAREM_SM12_ITEMS=pdl,kpool`, fail-closed on an unrecognised entry. **Item 4 was not taken**: what it removes could not be measured from the client in either direction. `patch-kpool-init.py` cannot be applied from inside the container — one target is bind-mounted read-only from `$OVERLAY_DIR` and needs a pre-patched host-side copy, verified with `diff -r` | prod 11 | 6 Sep 2026 | `tracks/tp3/patches/README.md`, `results/kernels/sm12-stack-patches-ab.md` §9 |
| All-reduce latency sweep, 8 B → 64 MiB doubling, 10 warmup + 50 timed per size | **8 KiB 74.68 µs · 64 KiB 86.40 µs · 1 MiB 275.12 µs · 16 MiB 1,096.79 µs = 20.40 GB/s busBW = 98.1 % of our own measured 20.8 GB/s wire · 64 MiB 3,954.98 µs / 22.62 GB/s.** From 8 B to 32 KiB — a **4,096×** range — the time never leaves **72–85 µs**: decode is latency-bound and prefill is at the wire. The 128 KiB point (172.48 µs, twice its neighbours) reproduced in all three repetitions and is the surviving trace of the old cliff. **132 of 132** correctness checks passed | model-free, production plugin build and production NCCL environment, engine up and idle | 6 Sep 2026 | `results/mesh/nccl-latency-sweep.md`, `bench/nccl-latency-bench.py` |
| `NCCL_MAX_NCHANNELS=8` and `NCCL_PROTO`, measured size by size | The channel cap does **nothing** at decode sizes (8 and 64 KiB inside noise) and is worth up to **11×** in the cliff: 1 MiB 275 µs against 1,388, 4 MiB 306 against 3,554, 16 MiB 1,097 against 3,046, and no difference at all at 64 MiB where bandwidth is already saturated. `NCCL_PROTO` auto is at least as good as every forced protocol at every size and clearly better between 128 KiB and 1 MiB, where `Simple` is **4–7× slower** | model-free, as above | 6 Sep 2026 | `results/mesh/nccl-latency-sweep.md` §6 |
| **Three of our own all-reduce harnesses disagree at small sizes, and all three readings stay** | 64 KB: `ar_bench.py` **61.3 µs**, this sweep **86.4 µs**, `mesh_sweep.py` **143 µs**. 8 KB: **38.6** against **74.68**, a factor of two. They differ in more than one variable at once, so **no cause can be assigned from the data that exists** and none of them is corrected against the others. The new page's conclusions rest on the curve's flatness and on a 16 MiB point where the harnesses agree, not on an absolute small-size value | model-free, three harnesses and three configurations | 4–6 Sep 2026 | `results/mesh/nccl-latency-sweep.md` §6.3, `results/mesh/all-reduce-sweep.md` §5, `docs/06` §12.1 |
| MLA prefill at production selection overlap, on the target part | At 262K context production runs **1.013×** a fully cache-resident arm at our own 22-head TP=3 shape (+1.3 %) and **1.054×** at the kernel author's 16-head shape (+5.4 %), against an independent-selection arm at **1.820×** and **2.468×**. Across all six cells (three contexts × two head counts) production closes **96.0–98.4 %** of the independent→drifting distance, and its excess over drifting at 262K — 223 µs at 22 heads, 575 at 16 — is **below one cold read of its own 187 MiB working set** (807 µs) | model-free, **one GB10 node**, kernel `cuda-exl3 754421f` inside the production image, engine down, 3-round medians | 6 Sep 2026 | `results/kernels/mla-prefill-falsification-gb10.md`, `bench/mla-prefill/` |
| The new item that falsification produced, and the check it did not run | At **22 heads** the kernel's compute-only floor costs **13–16 % more per head** than at 16 — 1.586 / 1.557 / 1.589× against a linear 1.375×, at all three contexts — and it is isolated to the compute path, because the traffic-bound arm's ratio sits *below* 1.375×. The fixture's correctness gate ran at a **2-head** smoke shape: the 22-head numeric check against the reference is **`[not tested]`**, its CPU-only self-test having passed 24/24 | as above | 6 Sep 2026 | `results/kernels/mla-prefill-falsification-gb10.md`, `HELP-WANTED.md` §8 |
| Other published recipes, quoted as their authors publish them | A dozen recipes, every figure `[reported]` exactly as its repository states it with the conditions it states — none re-derived, rescaled or averaged. The rows worth the page are §3.4 and §4.2: **two other people quantized this model's dense path independently, one at two nodes and one at three, by two different routes, and both measured a gain in the same band as ours** | — | 5 Sep 2026, §5.3 corrected 6 Sep | `docs/16-comparison-with-published-recipes.md` |
| Repository restructure and rename — **no measurement changed** | `patches/tp3full/` → `tracks/tp3/patches/` and `patches/tp2full/` → `tracks/tp2/patches/`, file for file; one byte-identical duplicate deleted; every `docs/NN` page gains an "Applies to" line and all **86** troubleshooting entries a track tag (45 both, 26 both but measured at TP=3 only, 13 TP=3 only, 2 TP=2 only). `preflight-tp3.py`'s vocabulary gate gained the per-rank condition it had never tested: `lcm(128, tp)` is right at `tp=3`, right by luck at `tp=2` and **wrong at `tp=4`**, where the unit is `128 × tp` = 512 — and `tp=4` itself is **`[not tested]`**. The repository is now `GLM-5.3-Flash-EXL3-DGX-Spark` | — | 6 Sep 2026 | `CHANGELOG.md`, `tracks/README.md`, `docs/00-start-here.md` |

**Five provenance facts a reviewer should not have to dig for:**

1. **Production 9 was profiled**, on the live server, all three ranks, no restart. Two of its rows —
   NCCL and CPU gap at C1 — are **inflated by the profiler itself**, because that arm launches 2,738
   kernels per step and CUPTI charges roughly a microsecond per kernel boundary. With the profiler off
   the two together are ≤17.19 ms rather than the 29.1 the trace prints, and the step wall is 72.52 ms
   rather than 84.44. **Subtracting the instrument properly on this arm has not been done**
   `[not tested]`; the prefill column, where the overhead is +1.4 %, is safe at face value.
2. **The MMLU figures for production 7 and 8 (86.4 ±0.7) were measured at TP=2** and carried forward.
   Only production 9's **86.47 ±0.74** was measured at TP=3. The gates are identical between the two
   arrangements, which is exactly why someone should check.
3. **The production 8 headline is a pool of two same-day runs** (56.9 / 175.4), because that arm's
   documented boot-to-boot spread is about 7 %. The single p8 boot alone read 56.8 / 172.8.
4. **Every headline speed figure is the median of one boot's rounds, and the spreads are not equal
   across metrics.** On the four production 9 and 10 boots, C1 boot medians span **1.1 %**, C8
   **2.5 %** and C4 **7.4 %**; within a single boot, round-to-round peak-to-peak reaches 5.7 % at C1
   and 9.8 % at C4. **Production 10's C4 of 144.6 against production 9's 134.6 is that spread, not a
   gain** — a memory fraction does not buy 7 % of four-way throughput. Every round of every boot is
   printed in `docs/10` §1.1 so this can be checked rather than taken on trust.
5. **The boot-from-power-on figure is published twice because its own log disagrees with itself.**
   The harness printed `+242 s`; the wall-clock stamps in the same file give **315 s**, and 242 s
   before the health check matches no event recorded. We could not reconstruct the counter's origin,
   so both are printed and the larger is the planning figure. The decomposition is not in doubt:
   units finished at +98 to +103 s, container to served token 212 s. `results/boot/boot-ledger.md`.

---

## 6. The retraction index

Everything we published here and then withdrew. The full text of each, with the measurement that
overturned it, is in [docs/11](../docs/11-open-issues.md) §1 unless the `#` column names another
page — this is the index, so a reviewer can see the shape of our error rate in one place.

The count is **thirty-eight**, and [docs/11](../docs/11-open-issues.md) §1's opening paragraph defines
it: the 32 rows of that page's audit table plus the six withdrawn findings that are not rows of it.
The rows below are the ones with a story worth telling, not the whole 38.

| # | What we claimed | What it actually was |
|---|---|---|
| 1.1 | The missing `n_rows` also costs the non-expert-parallel path | It does not; the tail is handled there |
| 1.2 | The extra masking pass is under 1 % of the MoE layer | 2.9 % at M=8 rising to **15.8 % at M=2048** |
| 1.3 | One upstream build is ~10 % slower end to end, and we do not know why | Boot-to-boot spread: **16 % at C8 with nothing changed at all** |
| 1.4 | NCCL picks the LL protocol at 16 MB and leaves half the link unused | The tuner plainly does not choose it; the real finding underneath was a different one |
| 1.5 | The MLA tuner re-tunes on every prefill chunk, wasting 2.6 % of prefill | A bounded warm-up repeat, not continuous re-tuning; the proposal to disable tuning was withdrawn |
| 1.6 | The mesh ceiling is ~13 GB/s against a 25 GB/s link, and GPUDirect is the fix | Two errors in one sentence, both ours |
| 1.7 | A pair of cables is worth 50 GB/s, so the collective runs at 28 % of the fabric | 50 GB/s is the **wire**; the card sits in a **PCIe Gen5 x4** slot, so the real ceiling is ~15 GB/s per card and ~30 GB/s per node |
| 1.8 | Two smaller ones — a retired patch, and a CUDA-graph capture claim | See docs/11 |
| 1.12 | "The 36/9 drafter sidecar removed the `22 % 4` head-count obstacle" — **our own correction of row 8, and it was wrong too** | Graphs captured on that boot because the draft KV was **bf16** (FlashAttention, no head check). The sidecar changed nothing: at three ranks it makes the division `22 % 3`. Since production 7 (fp8 draft KV → FlashInfer) graphs are **off** at TP=3 and **on** at TP=2 on the same image; the declaration is filed as [vllm#55581](https://github.com/vllm-project/vllm/issues/55581) |
| 1.13 | "A too-small indexer workspace ends in upstream's locked-workspace `AssertionError`" — published in our patch docstring, in HELP-WANTED §9 and on an upstream thread, with that assertion as the load-bearing safety layer | **Corrected by the issue's author, not by us** ([@drakosha](https://github.com/drakosha) on [vllm#55221](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5561194190)). Both indexers request a **static** size, so the assertion cannot fire from this path at all. The real failure mode is a **silent clamp** producing wrong answers, and the load-bearing layer is ours — the startup refusal. **The first of our retractions a reader outside this stack caught** |
| 1.10 | Three rows of the step-time breakdown, and the ranked target list built on them | `exl3_moe_combine` at 1.5 % of a chunk — **the kernel does not exist in this build**; `_zero_kv_blocks` at 1.3 % measures **0.09 %**; the "5.45 ms C1 idle" is **3.47 ms** once the profiler's own cost is subtracted, so graph coverage is worth +1.5–2 % rather than +6 % |
| 2.3 | 8.2 GiB per worker is stranded; equalising ranks is worth 8–26 % of pool | **The instrument.** The pool is a difference between two `/proc/meminfo` readings and runs backwards; acting on this would have over-committed the head node |
| 2.17 | The mHC triple shares a buffer, so the fusion is blocked | Thread-local dataflow, not a shared buffer |
| 2.18 | (came with production 7) | See docs/11 §2.18 |
| 2.22 | The checkpoint's scope was a quality decision | It was never a quality decision — the premise was wrong |
| **new** | **Production 9 cost 2.4 points of draft acceptance — "this is the cost"** | **A harness artefact.** Pooled by draft token over five levels and three boots it is **+0.18 points**. `bench-sweep.py` cycles `prompts[i % 12]`, so C1/C2 see 8 of 12 prompts and C4–C8 see all twelve; a two-group model explains all five levels at R² = 0.97, and the sign reverses at C6. Net effect on tok/s: **+0.24 %** |
| **new** | Full scope is "equal both ways" on prefill | True of the **wall** and it hides the class underneath: the dense stage in prefill is **+17.3 ms, +10.4 %** at M=1,792. The plugin author reproduced it on his own card and withdrew a "cuBLAS parity" claim from his README |
| 1.11 | **"EXL3 is 1.58–1.76× slower than BF16 on the KDA shapes at M=8" — the sentence that closed the quantization item** | **A warm number, on the wrong card.** The ~300 MB weight bank was sized for the large shapes and left the 0.72 MB KDA arm resident in a 101 MB L2. Cold on the **target** GPU the same shape reads **1.023**; GB10's own warm arm reproduces the withdrawn 1.596 at **1.605**. Seven of nine shapes reverse sign and the family goes from **−0.584 to +0.050 ms/step**. Two companions go with it: "GB10's ratios will be worse" (every family came out **better**) and "not bandwidth-bound, so bytes are not the cost" (right, but the cost is **two dependent launches**, so the remedy is a fusion). **The closure survives, re-scoped:** the arms stay BF16 for want of a gain, `kv_b_proj` still holds 96 % of it and still needs a kernel, and prefill was never re-measured |
| 2.4 | **"0.85 will not be attempted on this stack"**, and "0.88 was never attempted" | **The wrong ruler, on a machine that no longer exists.** Both rested on `MemFree` against a 4 GiB floor, and the 0.85 boot they came from predates the fast-load page-cache fix. Climbed rung by rung on 6 September against **swap traffic under load** instead: 0.85, 0.87 and 0.88 all pass, **0.90 is rejected**, 0.87 shipped as production configuration 11 — and **0.88, the rung that ladder passed and then declined on diagnostic headroom, shipped hours later as production configuration 12**. The measurement did not change; the decision about what the cluster is for did |
| 2.27 | The MLA-prefill "21–26 % overlap gap, worth about 2 % of a prefill chunk" | **It does not exist, and the item closes at zero.** Production was never between the kernel author's two arms — its selection turns over about **152 keys per row**, 76× his low-turnover arm — and on our own 48-SM part production runs within **1.3 %** of a fully cache-resident arm at 262K context. The only lever left on that kernel is less work, not less traffic |
| docs/17 §2.4 | "The driver takes 14.2 GiB" | **Stale by about 10 GiB.** It came from an NVFP4-era ceiling scan whose best observed free was 107.43 of 121.63; the three ranks start at **112.01 / 112.97 / 113.15** and an idle node offers **116.79**. The driver's fixed reserve is **3.12–3.49 GiB** and the total genuinely unavailable at idle is **4.84**. What limits the memory fraction now is the host's share at run time — a ruler correction, not a lever |
| docs/17 §4.1 | **Eight** of the nine KDA state slots exist so a rejected speculative step can be rolled back | **Seven.** The nine can be read as `1 + 8` off the non-align branch of that `if` and our first pass did read it that way; **align is the branch we run**, so it is `2 + 7`. No total moves — nine either way — but it is why the compact-rollback alternative goes to **two** slots and not to one |
| 1.14 | **"The vision tower cannot run at three ranks", plus "1.05 GiB of BF16 vision weights wasted on every rank at TP=2" and "`--mm-encoder-tp-mode data` would keep carrying it"** — in docs/03, docs/14, the README quick start and the TP=3 environment template | **All three withdrawn, by running it.** Production configuration 13 serves **4 images + 2 videos per request** at TP=3 with the text gates unmoved and no measurable cost. The 1.05 GiB is the *routed-experts-only* checkpoint; on the production full-scope checkpoint the tower is **6-bit EXL3, 0.557 GiB total / 0.416 GiB per rank**, readable from the safetensors headers in seconds. And the replication flag alone does **not** make the tower load: vLLM builds it with `quant_config=None` while this checkpoint's tower is quantized — the real obstacle, never in the sentence. [docs/18](../docs/18-vision-at-three-ranks.md) |
| docs/15 §3.3 | Full scope at TP=2 "is a rig and not a serving configuration" — it cannot boot at `max_model_len` 1,000,000 — and is "~10 GiB heavier per node" | **Neither reading survives two changes made since.** The draft page fix cuts blocks-per-request 640 → 280 and the launcher's settle gate turns 0.73 GiB of available KV memory into **16.07**; candidate B boots at 1M with a **2,128,571**-token pool and is the recommended two-node configuration. And it is **4.5–4.7 GiB lighter**, not 10 heavier: the old figure came from a boot with no settle gate |
| **new** | Production 11 costs **6.4 % of C4** | **Boot noise, and a repeat is what showed it.** The load boot read C4 at **135.0** against the reference's 144.2; the clean boot after a whole-cluster reboot read **145.9**. C4's six-round spread on this configuration is **11.1 %** against 2.5 % at C1 and 5.3 % at C8, and the same morning's ladder read C4 at 0.87 *without* these patches as 144.18. Printed rather than smoothed |

**Two retractions of the same number in two days** (1.6 and 1.7) is its own lesson, and it is the one
worth carrying away from this folder: **the ruler gets measured too.** "Two links × port rate" is not
a ceiling, it is the capacity of the wire; the path the card uses to reach the machine has to be
counted before a roofline is written. The same class of mistake put a 273 GB/s catalogue figure in
place of the 225 GB/s the device actually delivers.

Row 1.11 is that lesson a third time and in its hardest form: the ruler **was** measured, the artefact
**was** found — it read 210 % of peak — and a weight bank was added because of it. The bank was then
sized against the wrong shape on the wrong card, and the conclusion it protected was about the one
shape it did not cover. **Measuring the instrument once is not a property the instrument keeps.**

Nothing in `results/` was edited to match a retraction. The mistakes stay on the record with the
measurement that overturned them next to them.
