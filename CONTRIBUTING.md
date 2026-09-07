# Contributing — help us close the gaps

If you own comparable hardware, the items below are the ones a second cluster could actually settle.
Every one of them is something we could not run, not something we could not be bothered to run — the
reason is given with each.

**Before you open anything, four things will save you time:**

- [docs/00 — Start here](docs/00-start-here.md) asks how many nodes you have and says which pages are
  yours. Every `docs/NN` page carries an **Applies to** badge on its first line.
- [docs/14 — Troubleshooting](docs/14-troubleshooting.md) indexes every failure we hit by symptom,
  with the exact log line. If your stack is misbehaving, look there before you open an issue.
- [audit/](audit/README.md) is a post-install self-check with our own numbers next to each step. Run
  it before reporting that something is slow: it tells you *which* thing differs from ours.
- [STYLE-GUIDE.md](STYLE-GUIDE.md) governs anything written here, including evidence tiers.

## How to contribute a measurement

**[HELP-WANTED.md](HELP-WANTED.md) is the ranked list**, and it is the one to read first: it carries
the **expected effort** for every item, what to measure, and — for the items where it matters — what
a contributor with fewer nodes than the item needs can and cannot check. The list further down this
page is the older, unranked form of the same thing, kept because several of its entries carry detail
that would not fit there.

Three ways to send a result, in increasing order of how much we can do with it:

| | When to use it | What it becomes |
|---|---|---|
| **An issue**, using the *Measurement contribution* template | You have numbers and would rather not open a pull request | We ask for anything missing and add it ourselves, with credit |
| **A pull request** adding `results/community/<your-handle>/<item>/` | You have the raw output | It sits beside our own raw files and anyone can re-median it |
| **A pull request that also edits the page** | Your measurement changes what a page says | The best outcome. Say which claim of ours it corrects |

The issue and pull request templates in [`.github/`](.github/) are the measurement protocol from
[docs/09](docs/09-measurement-protocol.md) turned into checklists: rounds and which were discarded,
the per-metric noise band your difference has to clear, the prompt set named and labelled, the KV
pool read from a load boot on a settled host, the gates cold **and** warm, the full settings block,
an evidence tier, and what the gain cost. There are three issue templates — *did not reproduce*,
*measurement contribution* and *question* — and the first of them asks for the env **diff** against
your track's example rather than the whole file.

**Two things worth saying plainly.** A report that a step of ours did **not** reproduce is as
welcome as a measurement, and a silent failure — no log line at all — is worth more to us than a
loud one; §11 of [docs/14](docs/14-troubleshooting.md) is the index of the twenty we hit that
produced no error message. And **a pull request that withdraws one of our numbers is worth more than
one that adds a number**: thirty-eight claims of ours did not survive their own raw data, they are
kept in place with what replaced them ([docs/11](docs/11-open-issues.md) §1, whose opening paragraph
says what that number counts), and we do not ask for a replacement figure as the price of a
correction.

If you run one of the items below, please open a pull request adding your raw output under
`results/community/<your-handle>/<item>/` plus a short Markdown summary. Every number needs its
settings — image tag and which `cuda-exl3` commit, TP and EP, quantization, KV dtype, draft method
and `k`, `gpu-memory-utilization`, `--block-size`, `HAREM_SW_BLOCK_SIZE`,
`--max-num-batched-tokens`, `--max-num-seqs`, `NCCL_MAX_NCHANNELS`, the mesh plugin build and its
`NCCL_MESH_*` settings, whether `CUDA_EXL3_TUNE_CACHE` was set and warm, temperature, reasoning
effort, `max_tokens`, concurrency, prompt type, how many sweep rounds and which were discarded, date
— and an evidence tier as defined in [STYLE-GUIDE.md](STYLE-GUIDE.md). We will credit you in
[CREDITS.md](CREDITS.md).

## Items we did not run (most useful first)

1. **A large-M hyper-connection fusion kernel.** The single largest kernel-side item left in
   prefill: fusing the post-mapping kernel with the tf32 prenorm GEMM stops the residual stream being
   re-read once per call and is worth **−2.5 to −2.7 % of prefill**. It needs a new kernel that tiles
   over `block_m`; the existing fused one grids per token and is +32 % worse at M=2048. TileLang
   work, and everything needed to price it is in [docs/11](docs/11-open-issues.md) §2.16 and
   [docs/10](docs/10-results-and-roofline.md) §5.5. If you would rather have the cheap half:
   two keyword arguments at the call site and one constant in the third kernel are −0.4 % between
   them, measured, unclaimed.
2. **`NCCL_MAX_NCHANNELS=12` over two cables.** It was indistinguishable from 8 on a single cable and
   was never taken to the engine; over two cables the arithmetic changed and 16 turned out to be
   2.5× worse on the decode-sized message. 12 is now an open question, not a carried-forward
   equivalence. One boot settles it, one token in `EXTRA_ENV`. See [docs/06](docs/06-nccl-mesh.md)
   §8 and §14 item 1.
3. **`--max-num-batched-tokens 3072`.** 2048 and 4096 are both measured; the value between them is
   not. Report the KV pool, fresh prefill, mixed-load TTFT and C1–C8.
4. **The mesh plugin patches on a second cluster.** `patches/kernel/0005` and `0006` are worth +73 %
   on a 64 MB all-reduce and +4–6 % end to end here, and the idle-second-cable finding underneath
   0005 only exists on a fabric with more than one cable per node pair. If yours has one cable per
   pair, `NCCL_MESH_LINKS_PER_PEER=1` makes 0005 a no-op and 0006 is still worth measuring on its
   own. Start by reading your own `port_xmit_data` counters — see [docs/06](docs/06-nccl-mesh.md) §6.
5. **The full MMLU.** The 1,995-question sample **is** a TP=3 figure — 86.47 ±0.74, measured on the
   production checkpoint on 5 September — so the gap is not the rank count, it is the sample. The
   full **14,042-question** run has been staged twice and deferred on time both times, most recently
   in the production-12 battery ([`results/gates/quality-battery-production-12.md`](results/gates/quality-battery-production-12.md)).
   It is roughly a day of engine time here and it is the single largest quality number this
   repository does not have `[not tested]`.
6. **The newer checkpoint revision** (`aba59d21`, four days newer than the `b20c49ba` we pinned). A
   clean A/B against ours with the correctness probe, the code exam and an MMLU sample.
7. **The memory ladder above 0.88, and the same ladder at two ranks.** The "0.85 was rejected" that
   stood here is **retracted** ([docs/11](docs/11-open-issues.md) §2.4): it rested on a free-memory
   rule measured before the page-cache remedy in [docs/08](docs/08-fast-boot.md). Re-climbed on
   6 September against **swap traffic under load**, 0.85, 0.87 and 0.88 all pass and 0.90 is rejected;
   production is 0.88 ([`results/memory/ladder-6sep.md`](results/memory/ladder-6sep.md)). What is open
   is what sits between 0.88 and 0.90, and the whole ladder at **two** ranks, which must be re-derived
   rather than copied ([docs/15](docs/15-tp2-track.md) §6).
8. **The fast-load read path.** `HAREM_FASTLOAD_READ=mmap` exists in the code and has never been run.
   The sidecar currently reads at 0.88–1.04 GB/s where the same NVMe gives another loader 3.1 GB/s.
   Weight load could go from 67 s to about 20.
9. **`block_m` under expert parallelism.** The alignment needs the global expert count and the
   block-size heuristic is about rows per expert. A sweep would say whether the two uses want
   different numbers.
10. **The two-node gaps that are left.** Two Sparks are no longer an afterthought here: the track
    page is [docs/15](docs/15-tp2-track.md), the files are [`tracks/tp2`](tracks/tp2/README.md), and
    **candidate C** is a complete measured configuration. What is genuinely missing is in
    [docs/15](docs/15-tp2-track.md) §6, and the four we would most like are a **two-node reboot
    test**, the **memory ladder at two ranks**, **expert parallelism on at two ranks** (legal, never
    measured) and a **second boot of each candidate** so the speed rows carry a boot-to-boot spread
    rather than one boot's median `[not tested]`.
11. **The whole fabric story on an NVFP4 stack.** `NCCL_MAX_NCHANNELS=8` (+13 % at C8 for one line),
    then the two plugin patches. All three are properties of the plugin and the wiring, not of the
    quantization, so they should transfer — and if they do, most of the EXL3-versus-NVFP4 gap in
    [docs/10](docs/10-results-and-roofline.md) §3 transfers with them.
12. **Anything at max reasoning effort.** Everything here is at `low`. Expect 5–12× the time.

Two more that we would take from anyone, out of order because they are larger than most of the list:
**overlapping the all-reduce with compute** ([docs/11](docs/11-open-issues.md) §2.17 — the collective
is 16.5 % of a prefill chunk and is serialised against compute at 99.3 % occupancy, so it is the
biggest number on the board with nobody on it), and **a torch-profiler run on a production
configuration** with `--profiler-config` set at launch, which would replace the 2.8 % reconciliation
residual in [docs/10](docs/10-results-and-roofline.md) §5 with a measurement and settle the C8 decode
split.

## What we would rather you did not send

- A single pair of sweep rounds as evidence for anything. Boot-to-boot spread on this stack is up to
  16 % on C8; see [docs/09](docs/09-measurement-protocol.md) §2. We have published a kernel
  conclusion drawn from one pair and had to withdraw it.
- A three-round median from an image **without** a warm MLA tuner cache. Three rounds is what the
  persisted cache bought and it is conditional on it; without one, the rule is still five rounds with
  the first two discarded. Say which you ran. See [docs/12](docs/12-tuner-cache.md).
- A prefill number measured on a repeated prompt. It reads the prefix cache and overstates by up to
  55 %.
- A speed number without the quality gates from the same boot, cold **and** after the benchmark.

## House rules for a pull request

- Keep our documents unchanged unless you are correcting an error; in that case cite the raw file
  that shows the correction. We keep the retraction, we do not delete the mistake — see
  [docs/11](docs/11-open-issues.md) §1.
- **Never include host names, LAN addresses, user names, home paths or tokens** in anything you
  upload. Grep before you push; we do, before every commit.
- Report what a gain cost — speed, quality and memory together. If it cost nothing, say that you
  looked and give the numbers that show it.
