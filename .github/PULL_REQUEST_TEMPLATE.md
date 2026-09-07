<!--
Read CONTRIBUTING.md and STYLE-GUIDE.md before you fill this in. HELP-WANTED.md is the ranked
list of what a second cluster could settle. Delete the sections that do not apply to your change,
but do not delete section 5.
-->

## 1. Which track

- [ ] **Both tracks** — a shared page, a shared script, a kernel patch, the measurement protocol
- [ ] **TP=3 only** — three nodes
- [ ] **TP=2 only** — two nodes
- [ ] **Neither** — a node count we do not own. Say which, and mark every number in it `[reported]`
      or `[measured-here]` on **your** cluster, never as ours.

`docs/00-start-here.md` has the badge scheme and the per-document table. If your change touches a
page whose badge would now be wrong, change the badge in the same pull request.

## 2. What this changes, in one paragraph

What it does, and what it replaces. If it corrects something of ours, say which file and which line
was wrong — we keep the mistake and add the retraction, we do not delete the mistake
(`docs/11-open-issues.md` §1).

## 3. Measured numbers, with the protocol

**No number without a log.** Every figure in a pull request here carries where it came from and how
it was taken. The rules are in `docs/09-measurement-protocol.md`; the short form:

- Rounds: 1 warm-up + 3 measured with a **warm** tuner cache, or 5 with rounds 1-2 discarded.
  Say which, and give the individual rounds, not only the median.
- The noise band the difference has to clear: C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 % within a
  boot; **15.9 % on C8** between boots. Under about 5 % is not a result on one boot.
- Prompt set named, and labelled realistic / synthetic / fresh. Prefill on a repeated prompt is not
  a prefill measurement.
- KV pool read from a load boot on a settled host, not from a dump boot.
- The gates, cold **and** warm: correctness probe __/10, code exam __/12, empty completions __.
- The full settings block: image tag and `cuda-exl3` commit, TP/EP, checkpoint and revision, KV
  dtype, draft method and `k`, `gpu-memory-utilization`, `--block-size`, `HAREM_SW_BLOCK_SIZE`,
  `--max-num-batched-tokens`, `--max-num-seqs`, `NCCL_MAX_NCHANNELS`, mesh plugin build,
  temperature, reasoning effort, `max_tokens`, concurrency, date.

Raw output goes under `results/community/<your-handle>/<item>/` with a short Markdown summary. A
number in prose with no file behind it will be asked for the file.

Every measured claim carries an evidence tier from `STYLE-GUIDE.md`: `[measured-here]`,
`[measured-here, raw lost]`, `[reported]`, `[estimate]`, `[not tested]`, `[retracted]`.

## 4. What it cost

Speed, quality and memory, together, for every gain. If it cost nothing, say that you looked for the
price and give the numbers that show it. A pull request whose cost line is empty will be asked for
it before anything else.

## 5. Before you push

- [ ] **No host names, LAN addresses, user names, home paths or tokens** anywhere in the diff.
      We grep before every commit and so should you:
      `grep -rnI -iE "your-hostname|192\.168\.|10\.0\.|/home/[a-z]+|ghp_|hf_" <your files>`
      Documentation addresses only: `192.0.2.10`, `192.0.2.11`, `192.0.2.12`, `192.0.2.100`.
      Machines are `head`, `worker-1`, `worker-2`.
- [ ] Relative links resolve. No emojis, no exclamation marks, English.
- [ ] Commands are copy-paste runnable on a node that has only this repository and the pinned
      upstreams. One command per fenced block, no `$` prompt.
- [ ] If you touched a patch script, `python3 -m py_compile` is clean and the anchors still match
      exactly once.
- [ ] If you added or removed a file in a patch tree, you know it invalidates the fast-load sidecar
      identity on every node and costs a dump boot (`docs/08-fast-boot.md` §4).

## 6. Retractions are welcome

**A pull request that withdraws a number is worth more here than one that adds a number, and it will
be merged with thanks.** This repository has published and then withdrawn thirty-eight of its own
claims; they are kept, in place, with what replaced them (`docs/11-open-issues.md` §1 says what that
number counts). If you measured something of ours properly
and it does not hold, open the pull request. Say what you ran, what you got, and which claim it
overturns. We do not ask for a replacement number as the price of a correction.
