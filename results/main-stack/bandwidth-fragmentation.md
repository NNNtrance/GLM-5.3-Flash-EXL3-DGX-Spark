# After many engine start-and-stop cycles the GB10 memory bandwidth falls 7–11 %; only a reboot restores it (7–8 October 2026)

**Applies to: any measurement or deployment on GB10 nodes (DGX Spark), both tracks.** It is a finding about
the instrument that every other file in this directory depends on: a baseline taken on a freshly rebooted
node is not comparable with a measurement taken after several engine sessions. In production it is
harmless as long as the engine starts once at boot and nothing restarts it without a reboot.

## 0. Settings and the probe

| | |
|---|---|
| Hardware | 3× DGX Spark (GB10), one 121.63 GiB unified memory each; kernel 6.17.0-1032-nvidia, driver 580.178.04 |
| Probe | one container per node, no network, GPU otherwise idle, run before and after arms. Per-second classification of **continuous bf16 matrix-vector** throughput (6×5120 @ 5120×16384: 168 MB read per call, 10 s blocks, two rounds), a **64 MB device copy**, and burst-with-gaps variants of both. The pattern (alternating burst/continuous blocks classified per second) follows a public report of a GB10 "slow mode" (`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`, issue 1, called "gpuflip" there) `[reported]`; the code is ours |
| Companion readings | `/proc/buddyinfo` (free blocks per order, Normal zone), `/proc/vmstat` compaction counters, `nvidia-smi` clocks and throttle reasons, top processes |
| Raw | [`raw/bandwidth-probe-series.csv`](raw/bandwidth-probe-series.csv): 34 probe samples × 3 nodes with the number of engine sessions since the last reboot |

All rows `[measured-here, private harness]`; the probe is a 60-line PyTorch script that the repository does
not carry. Note on the probe's own verdict: it classifies a second as "fast" above 150 GB/s, so **it said
"fast" for every second at 219 GB/s**; read the absolute number, not the label.

## 1. What we saw

**Fresh node:** head 233.4–236.5 GB/s (median 234.5), workers 243.6–245.5 GB/s (median 244.3), 24 fresh-reboot
samples across the night (about ±0.7 % sample to sample). Free 32 MB blocks (order 13): 3,654–3,723 on every
node. 64 MB copy 239–244 GB/s.

**After many engine sessions** (a night of ~10 engine starts in about four hours):

| Node | Right after a reboot | After ~10 sessions, no reboot | Change | Free 32 MB blocks |
|---|---|---|---|---|
| head | 234.3 GB/s | **218.9** | **−6.6 %** | 3,707 → **368** |
| worker-1 | 244.7 | **219.3** | **−10.4 %** | 3,711 → **345** |
| worker-2 | 244.1 | **219.2** | **−10.2 %** | 3,712 → **237** |

(Probe at 00:51 on 8 October against the probe right after the reboot at 00:57. Against the 20:33 reboot of
the evening before, the probe at 00:28–00:32 had read −7 % / −10 % / −11 %.) The 64 MB copy fell the same way,
240–244 → 219–220 GB/s.

**After one session:** −3.7 % on the head and −6 to −7 % on the workers (head 234–235 → 224.5–226.8, workers
244 → 226.6–230.5; six samples per node, in six different arms), with order-13 blocks 3,7xx → 2,070–2,472. The
earlier estimate in our own notes, ~1–1.5 % per start, was wrong: **one session is already most of it** on
the workers.

**After 6–7 sessions in 95 minutes:** head 236.1 → 221.7 (−6.1 %), workers 244.9 → 220.4 (−10.0 %) and
244.6 → 220.0 (−10.1 %); blocks 3,654 → 1,030, 3,665 → 722, 3,664 → 657. At the same time the first-boot
swap-out on the head grew from 21–24 pages (the first boots) to 1,731–9,570 pages per boot in the same night.

**What does not restore it:**

| Intervention | Band after (head / worker-1 / worker-2, GB/s) | Order-13 blocks |
|---|---|---|
| none (degraded) | 218.9 / 219.3 / 219.2 | 368 / 345 / 237 |
| `sync; echo 3 > drop_caches; echo 1 > compact_memory` | 219.1 / 219.2 / 219.1 (**no change**) | 505 / 543 / 410 |
| **triple reboot (about 100 s)** | **234.3 / 244.7 / 244.1** | **3,707 / 3,711 / 3,712** |

`compact_stall` was ~6,000 and `compact_fail` 99 % of attempts.

**Eliminated as causes:** a leftover process (top-CPU list empty, `nvidia-smi` 0.8 %); heat or clocks (GPU
50 °C, no throttle reason); the image build, test and distribution containers (a probe after a reboot
followed by an image-build container (2 min 20 s), the 620-test suite (4 min 49 s) and a Docker image load:
**236.5 / 245.1 / 244.2**, unchanged, although worker-2's free 32 MB blocks were already 3,274 against its usual 3,712).

**Mechanism (unproven):** physical memory fragments at the 32 MB scale, so the GPU/SMMU mappings that engine
sessions create lose large contiguous blocks (on an ARM64 4K-granule kernel the contiguous hint covers 32
MB), raising translation misses; a reboot returns the blocks. `[estimate]`

## 2. How it nearly cost us a real result

The first test of an attempt to turn on CUDA graphs for the draft model ([rejected-draft-cuda-graph.md](rejected-draft-cuda-graph.md))
was compared with a baseline from earlier that evening, on a fresh node. After the night's sessions the
nodes were at 219 GB/s. The same MoE GEMM kernel took **379 µs per call in the baseline and 407 µs (+7.5 %)
in the test**, which hid about two points of the gain:

| Comparison | One-user step change |
|---|---|
| test vs baseline measured earlier on fresh nodes | **−0.6 %** (verdict: reject) |
| test vs a control booted 25 minutes later on the same degraded nodes | **−2.56 %** |

A baseline from a different boot state was wrong by 2 points on an effect of the same size. The same night's
other runs show it in the other direction: the speed-table baseline, taken after several engine starts the
previous evening, read 2.6–6.5 % per step slower than a fresh-boot control of the same configuration.

## 3. The measurement protocol it forced

1. **Triple reboot of the cluster before every arm** (about 100 s; head 94–105 s, workers 39–42 s). On the
   night this was written, about 25 such reboots all came back first time.
2. **A probe before and after each arm**, with the absolute number and the order-13 count recorded; a probe
   under **230 GB/s** gates a second reboot (never needed after the protocol started).
3. **A baseline in the same chain** — A/B/A/B in the same session, not an old one. A baseline from another
   boot state is secondary evidence only.
4. **Validity of an arm is its own probe, not a probe bracketing two arms.** The first wording of this rule
   (one probe before arm 1 and one after arm 2, shift ≤ 2 %) was written when arms ran back to back; once
   every arm began after its own reboot, the bracketing pair was meaningless. It was changed before the
   data that used it ([kl-health.md](kl-health.md) §6): control-to-candidate band shift 241.3 → 241.1 GB/s
   (−0.08 %).
5. **Prefer step time over token rate** for deciding an engine change, because token rates carry the draft's
   acceptance noise (±5–12 % per cell) and the step does not (±1–3 %).

It also bears on [docs/09](../../docs/09-measurement-protocol.md): its five-round, discard-two rule governs
engine comparisons; this is the *boot-state* rule that goes with it.

## 4. What this cost

- About 100 s and a whole-cluster outage **per arm**, and about 25 reboots in the night — plus the
  discipline to never reuse an old baseline.
- **In production, a rule:** the unit starts once at boot on fresh memory; a restart without a reboot starts
  the next engine session on memory that has already lost 4–11 % of its bandwidth, until the next reboot. The
  watchdog's action is therefore a triple reboot, not an engine restart ([boot-and-watchdog.md](boot-and-watchdog.md)).

## 5. Rejected, open, retracted

**Rejected explanations:** leftover processes, heat, clocks, build and test containers, `drop_caches` and
`compact_memory`.

**Open**

- **The mechanism is a hypothesis**: fragmentation correlates with the loss on 34 samples but a causal test
  (for example deliberately fragmenting a fresh node, or forcing large-page allocation) was not run.
- Whether a kernel or driver setting prevents it; none was tried, because changing the system on shared
  nodes was out of scope.
- **When** the loss develops: every reading was taken with the engine stopped, so we cannot say whether it
  accumulates *during* a session (production runs for days) or at teardown. The eight-hour run did not
  re-probe at its end ([soak-8h.md](soak-8h.md)) `[not tested]`.
- Why the head loses less after one session (−3.7 %) than the workers (−6 to −7 %): it starts lower (234
  against 244 GB/s) and is the node carrying the API; not investigated.
- Whether earlier results in this repository are affected: any that compared an arm against a baseline
  from a different boot state, which this stack's earlier files did not always separate.

**Retracted:** nothing published. The internal first estimate ("about 1–1.5 % per engine start") was
superseded by the measurement above (one start is already −4 to −7 %).
