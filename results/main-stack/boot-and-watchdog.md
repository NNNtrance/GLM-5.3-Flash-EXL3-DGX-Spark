# Boot from power-on in 274 s, a watchdog that reboots all three nodes, and the drill that tested it (8 October 2026)

**Applies to: TP=3, the main (vLLM `main`) stack, production configuration.** Three things measured the same
afternoon: the engine starting by itself at boot, a compliance exam of the running engine, and a deliberate
engine kill to see whether the watchdog recovers the cluster without a person.

## 0. Settings

| | |
|---|---|
| Image / engine | `a4i-21d93d0d8-448f1d6`; TP=3 + EP, `gpu-memory-utilization` 0.84, `--max-model-len 1000000`, `--max-num-seqs 5`, draft k=7 with NODROP and the `[[1,1,7],[2,8,3]]` schedule, vision 16 images / 4 videos with `--mm-encoder-tp-mode data`, fast-load LOAD sidecar (53 GB per node), `NCCL_MESH_TIMEOUT_SEC=180`, served on the production port under the name `glm-5.3-flash` |
| Start path | a systemd oneshot unit per node (enabled), whose `ExecStartPre` is a fail-closed precondition script and whose `ExecStart` runs the engine container; `TimeoutStartSec` 1500 |
| Preconditions (any miss blocks the start) | Docker up; ConnectX-7 ports **4 of 4** active; each fabric neighbour answers a jumbo-frame ping; page cache dropped and `MemAvailable` ≥ **112 GiB** afterwards; the mesh plugin and chat-template files match their recorded hashes; the pinned environment/image/sidecar manifests match; on the head, the **JIT-cache gate** (the three nodes' compile caches must list the same files, else a node that compiles for 49 s while the others wait trips the 30 s mesh timeout that hangs silently — the failure behind `NCCL_MESH_TIMEOUT_SEC=180`) |
| Warning, not a block | fewer than 2,000 free 32 MB blocks in `/proc/buddyinfo` order 13 (a booted-fresh node has ~3,700; [bandwidth-fragmentation.md](bandwidth-fragmentation.md)) |

All rows `[measured-here, private harness]` (the install, exam and drill scripts are not in this repository;
the unit shape is in [`systemd/`](../../systemd/README.md)).

## 1. Boot from power-on

The three nodes were rebooted together (`systemctl reboot` on all three at 16:33:18) and the units started
the engine by themselves. Time is measured **from the reboot command**.

| | |
|---|---|
| Nodes back (ssh, Docker, fabric 4/4, neighbour jumbo pings, MTU 9000 / 200 Gb/s) | head 103 s, worker-1 41 s, worker-2 42 s |
| **`/health` 200** | **274 s** (target ≤ 360, acceptance limit ≤ 600) |
| KV pool | **6,190,735** tokens (6,188,010 on the test door earlier the same day; the KL API flags are off in production) |
| Router-bias boot gate | `GECTI` **42 of 42** layers on each of 3 ranks; non-passing gate lines 0; `Traceback` 0 |
| Start arguments | gmu 0.84, production port, `disable_eagle_block_drop: True`, `[[1, 1, 7], [2, 8, 3]]`, `{'image': 16, 'video': 4}`, `mm_encoder_tp_mode: data`, `max_model_len 1000000`, `max_num_seqs 5`; NODROP warning line present |
| `NVRM Xid` | 0 on all nodes |
| `MemAvailable` just after boot | 7.4 / 9.2 / 9.2 GiB (head / worker-1 / worker-2) |
| Precondition script | 12 s on the head, 62 s on the workers (the JIT-cache gate is head-only) |

One red line in the acceptance script — a pattern error (it looked for `'served_model_name': 'glm-5.3-flash'`
and the argument is printed as a list `['glm-5.3-flash']`) — was judged an exam defect after confirming the
model list answers the right id. We list it because the script said "failed" and the engine was fine.

For orientation, other boot times in this repository: the previous stack's whole-cluster boot-to-`/health`
was **311–321 s** ([`results/boot/boot-ledger.md`](../boot/boot-ledger.md)); the fast-load LOAD path on the
test door of this stack took **169–187 s** from `docker run`, the dump boot that writes the sidecar **380 s**.
The 274 s and the previous stack's figure are not like-for-like (different image, vision limit and memory
fraction), but both are the same quantity: reboot command to `/health` 200.

## 2. The compliance exam: 7 of 8

An eight-item exam from an independent client against the running unit (317.5 s of a 900 s limit):

| Item | Result | Detail |
|---|---|---|
| health | pass | `/health` 200; `max_model_len` 1,000,000 |
| chat | pass | answer correct, finish `stop` |
| streaming | pass | 3 chunks, first token 0.20 s, `[DONE]`, usage present |
| tool call | pass | one tool call correct; **strict tool-call gate 30 turns, 0 failed, 0 corrupted** (longest prompt 34,565 tokens; 206.6 s) |
| JSON output | pass | `json_object` and `json_schema` both valid |
| needle at 60k | pass | **3 of 3** found at depths 10 % / 50 % / 90 % (prompts 59,992–59,994 tokens; 94.3 s) |
| vision | pass | 2 images, 1 video, and 2 images + 1 video in one request, all correct, 13.4 s |
| thinking effort | **fail — exam defect** | see below |

**The failing item was the exam's fault.** It asked an easy question at effort `high` and required visible
reasoning; on an easy question the model does not think at either effort (3 completion tokens both times). A
harder question produced **365 characters of reasoning at `low` and 589 at `high`**, so the effort setting
works. That second check was run by hand and is not in an archived output file `[measured-here, raw lost]`.
No rule was reversed; the exam item is to be rewritten with a question that forces thought.

## 3. The watchdog (v2)

The previous watchdog restarted the engine services (no reboot; at most two restarts in two hours) when
`/health` had been unhealthy for three minutes. Two things made that wrong for this cluster: **`/health` keeps returning 200 while the mesh is silently hung**
(a silent hang went unseen for about 33 minutes), and **restarting one engine is not a safe recovery** — the
cluster's rule is that a crashed engine means all three nodes reboot together, because one node rebooting
alone takes down its neighbour's port. The new one:

| Rule | Value |
|---|---|
| Runs | on the head, every 60 s (systemd timer, first run 10 min after boot) |
| Probe | `/health` must be 200, **and** a one-token-class completion (`max_tokens` 8, temperature 0, 1+1=) must return within **60 s** |
| Busy exemption | if the probe times out but the token counters in `/metrics` advanced during it, the engine is "busy" (five slots full), not stuck; **at most 10 exemptions in a row**, then it counts as stuck. A frozen counter is a real hang |
| Action after | **3 consecutive failures** (≈ 3 minutes) |
| Action | collect `docker logs --tail 300` from each node as evidence, then **reboot the workers (ssh), then the head** |
| Rate limit | at most **1 reboot per 30 min and 3 per 6 h**; beyond that it writes an `ALARM` file and does nothing until a person deletes it |
| Boot grace | does nothing in the first **900 s** of uptime or while the engine container is younger than **1,500 s** |
| Refuses to act | if a peer is unreachable over ssh (a one-machine reboot would kill the opposite fabric port); if the unit is `inactive` (stopped on purpose); a `failed` unit counts as a failure; if the pause file exists |
| Dry run | `--kuru` logs what it *would* do and triggers nothing |

Tests before installing: 33 of 33 cases against a mock engine with **3 mutations of the script all caught**;
install/rollback in a sandbox 35/35 then 38/38 (resolved systemd state checked); on the live engine a
dry-run probe returned in 1–3 s; a forced-failure dry run (probe timeout 0.01 s, exemption off, three
rounds) logged "would reboot worker-1, worker-2, head" on the third and rebooted nothing.

**The install failed once.** The first install attempt on the three nodes aborted when a root-owned,
mode-600 JIT-seed file could not be read, and the installer **rolled all three nodes back automatically**
(no half-install); a one-word fix (`sudo -n cp -a`) and the second attempt passed. Both the roll-back and
the install-time checks are therefore exercised in earnest.

## 4. The drill: killing the engine

Activated at 16:45 (the old timer disabled, the new one enabled, the pause file removed last). At 17:01:08
`docker kill` was run on the engine container of worker-2.

| Time | Event |
|---|---|
| 17:01:08 | engine container killed on worker-2 |
| 17:01:49 | watchdog failure 1 of 3 |
| 17:02:49 | watchdog failure 2 of 3 |
| 17:03:49 | failure 3 → evidence collected → **triple reboot: workers first, then the head** |
| +316 s (from the kill) | all three nodes reachable again |
| **+494 s (8.2 min) from the kill** | **the units started the engine by themselves; `/health` 200** |

Evidence directory written; **no alarm**. Measured from the reboot command, `/health` came back at about
333 s (17:03:49 → 17:09:22, derived) against 274 s in the clean power-on test; the 59 s difference was not
investigated (the killed container, a different cache state and worker-first sequencing are candidates).
The time from fault to first failure count is the watchdog's own 3-minute tolerance; the engine was out for
~8 minutes in total, and the next eight hours ran on the containers that drill started ([soak-8h.md](soak-8h.md)).

## 5. What this cost

- **A reboot of three nodes for every real engine failure** — about 8 minutes of outage and a loss of every
  in-flight request and every cache. The old behaviour (restart the engine) was faster but did not reliably
  recover a hung mesh, and repeated restarts degrade memory bandwidth ([bandwidth-fragmentation.md](bandwidth-fragmentation.md)).
- **False-positive exposure:** a probe that merely queues behind five busy slots would be misread as a hang;
  the exemption handles it only while token counters advance.
- **Install ceremony:** a precondition script of 12–62 s at every boot.
- **Rate-limited help:** after one reboot in 30 minutes the watchdog stops and waits for a human.

## 6. Rejected, open, retracted

**Open**

- **The drill tested only the "container killed" path.** The silent-hang path (`/health` 200, probe times out)
  was verified in dry run only, never against a live hang; the rate-limit and ALARM path, and the
  peer-unreachable refusal, were exercised only in the mock.
- **The watchdog lives on the head.** If the head node itself dies, nothing watches; the workers do not run it
  `[not tested]`.
- The 59 s gap between the drill's boot (≈ 333 s) and the clean power-on boot (274 s).
- The mesh plugin does not abort the process on its own timeout: a rank more than the configured timeout
  behind hangs the cluster silently instead of failing loudly. The 180 s setting and the JIT-cache gate
  prevent the case we hit; the underlying behaviour is the same as on the previous stack, where it also had a
  30 s default.
- The thinking-effort exam item (§2).

**Rejected:** restarting only the engine on failure as the watchdog's action (§3).

**Retracted:** none.
