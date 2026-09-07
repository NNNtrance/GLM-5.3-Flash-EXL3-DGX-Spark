# Style guide for authors (agents and humans) — read before writing anything

Audience: people (and their AI coding agents, e.g. Claude Code) who own 3× NVIDIA DGX Spark (GB10)
and want to run `zai-org/GLM-5.3-Flash` through the EXL3 path with this exact stack.
Language: **English**. Tone: factual, no marketing. Every number carries its settings. Every claim
carries its evidence tier.

This guide is the same one we wrote for the NVFP4 recipe, with the EXL3-specific rules added at the
end. Where the two repositories disagree, this file governs this repository.

## Naming (privacy rule — NO exceptions)

- Machines: `head` (rank 0, serves the API), `worker-1`, `worker-2`. Never the real hostnames.
- IPs: documentation addresses only — `192.0.2.10` (head), `192.0.2.11` (worker-1),
  `192.0.2.12` (worker-2), workstation/client `192.0.2.100`. Never real LAN addresses, never the
  fabric `/24`s.
- Users and paths: `$USER` and `$HOME`; `~/exl3-zeus/` is fine (it is the install directory).
  Never a real username, never `/home/<name>`.
- Never mention: our vault, our internal file paths, e-mail addresses, remote collaborators,
  tokens or keys of any kind.
- The word **HAREM** appears inside the stack: patch markers (`HAREM-TP3`, `HAREM-GB10-TOPK`),
  environment variables (`HAREM_SW_BLOCK_SIZE`, `HAREM_DISABLE_PERSISTENT_TOPK`, `HAREM_FASTLOAD_*`,
  `HAREM_EP_ZERO_MODE`, `HAREM_EP_FILTER_SUFFIXES`), function names (`_harem_*`), the module
  `harem_fastload.py`, some image tags and log lines. **Keep them** — the patch scripts match those
  strings exactly and fail closed when an anchor stops matching. The README explains the name once.
  Do not explain it again anywhere else, and do not use it in prose.

## Evidence tiers (put one on every measured claim)

- `[measured-here]` — we measured it on this cluster and the raw data is in `results/`.
- `[measured-here, raw lost]` — we measured it but the raw file did not survive; say so.
- `[reported]` — someone else reported it (link).
- `[estimate]` — our estimate, not measured.
- `[not tested]` — we did not test it.
- `[retracted]` — we published it, then measured it properly and it was wrong. Say what replaced it.

## Every number needs its settings

Image tag (which `cuda-exl3` commit), TP/EP, quantization, KV dtype, draft on/off and `k`,
`gpu-memory-utilization`, `--block-size`, `HAREM_SW_BLOCK_SIZE`, `--max-num-batched-tokens`,
`--max-num-seqs`, `NCCL_MAX_NCHANNELS`, temperature, reasoning effort, `max_tokens`, concurrency,
prompt type (synthetic / realistic / fresh), number of sweep rounds and which were discarded, date.
Put it in the table caption or in a settings block above the table.

## Speed: synthetic vs realistic — always separated, always labelled

- **Synthetic** — "count 1→200" and friends: this measures the speculative-decoding CEILING. Label it.
- **Realistic** — the 12 short English code prompts of `hizset-v2`, plus the four category prompt sets
  (prose / code / math / JSON).
- **Fresh** — for prefill only: a prompt the engine has never seen. A repeated prompt reads two whole
  3,328-token blocks out of the prefix cache and reports up to 1,596 tok/s where the honest number is
  1,025. Prefill measured on a repeated prompt is not a prefill measurement.
- State plainly: synthetic numbers will disappoint in real use; prose acceptance is ~13 %, code
  ~46–50 %.

## Quality and benchmarks

Always say: all runs at reasoning effort **low**, temperature 0 unless stated, and explain why (max
effort would take days on this cluster). Where a max-effort number would change a conclusion, give an
`[estimate]` and mark it.

## Honesty sections (mandatory)

- "What we tried and rejected", with reason and evidence tier.
- "Open problems" (unsolved) and "Retracted" (numbers we withdrew, and what replaced them). This
  stack has produced **thirty-eight** retractions already; they are in
  [docs/11](docs/11-open-issues.md) §1, which defines the count, and they stay there.
- A **"what this cost"** line for every gain — speed, quality and memory together. No gain is
  reported without its price. If the price is genuinely zero, say that it was looked for.

## Credits and licences

Every external component: name, link, exact revision (commit / HF sha / image digest), license, and
what we use it for. Our own patches: "written by us for this recipe; use freely (Apache-2.0); a
credit is appreciated" — and say when a patch was adapted from someone else's idea (name and link).
Where a patch of ours was **superseded by upstream**, say so and retire it in writing rather than
quietly deleting it. Where upstream turned out to be right and we were wrong, say that too.

## Formatting

Markdown; tables for numbers; fenced code blocks for commands, one command per block, no `$` prompt;
relative links between documents; no emojis; no exclamation marks. Commands must be copy-paste
runnable on a node with only this repository and the pinned upstreams.

## EXL3-specific rules

- Always name the `cuda-exl3` commit an image was built from. "the current build" is not a revision.
- The upstream kernel project moves fast. When you record a kernel measurement, record the commit on
  both sides of the comparison, and check before publishing whether upstream has since adopted or
  refuted it.
- Never compare two engine arms on fewer than five sweep rounds with the first two discarded. This
  stack's MLA tuner warm-up has made a winning arm look 25–45 % worse on the first pass — see
  [docs/09](docs/09-measurement-protocol.md).
- Model-free first. Every conclusion in this repository that survived was reached with the engine
  down, on a micro-benchmark with the real shapes; every conclusion drawn from two engine sweeps
  alone has since been retracted.


## The opening statement

Every repository of ours opens, directly under the title, with the "below the engine" block: what we
did beneath the flags — kernel, transport, loader — in this specific repository, each item true of
this repository's contents, followed by the measurement-and-retraction sentence. Update the examples
to what the new repository actually contains; never carry an example over from another repository.
The claim is about depth of work, not about other people's recipes — name no one, and keep the
comparison page ([docs/16](docs/16-comparison-with-published-recipes.md)) as the only place other
recipes are discussed.
