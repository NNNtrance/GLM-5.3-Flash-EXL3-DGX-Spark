# Stopwatch speeds on the production configuration (7 September 2026)

Three DGX Sparks, TP=3 + EP, the full-scope EXL3 4.05 bpw checkpoint, DFlash2 drafter k=7, vision
enabled (4 images + 2 videos per request), `gpu_memory_utilization=0.88`, KV pool 7,016,528 tokens,
`max_num_batched_tokens=2048`, API on one node. Nothing else was running on the cluster.

These are **stopwatch** numbers: the vLLM counters `generation_tokens_total` and
`prompt_tokens_total` were read twice, 60 s apart, and divided by 60. They are therefore sustained
averages over a full minute of steady load, not the harness numbers in the README's headline table
(which use a fixed prompt set and per-request timing) and not the momentary readings of a dashboard.

| Load | Sustained, 60 s stopwatch | Notes |
|---|---|---|
| Decode: 12 code-generation requests submitted at once (8 running, 4 queued), 2,000 max tokens each | **205 tok/s** aggregate | Draft acceptance 55–62 % in this regime. The dashboard's one-second EMA peaked at **261 tok/s** during the same load; treat that as a peak, not a rate. |
| Prefill: 24 fresh, unseen ~7K-token prompts kept in flight, 8 output tokens each | **1,914 tok/s** aggregate | Only 2–3 requests are "running" at any moment: the scheduler fills the 2,048-token step budget with the first waiting prompts, so prefill throughput does not rise with concurrency the way decode does. |

Two things worth knowing if you reproduce this:

* `prompt_tokens_total` is incremented when a request's prefill **finishes**, not chunk by chunk.
  One-second samples therefore read 0, 0, 0, 17,000, 0 while the engine is steadily reading ~1,900
  tok/s. Average over at least ten seconds before quoting a prefill rate.
* Prefill is bounded by the step budget, not by the number of concurrent prompts. Raising
  `max_num_batched_tokens` to 4096 lifts it, at a cost in KV pool we chose not to pay
  (see [docs/10](../../docs/10-results-and-roofline.md)).

Prose-heavy generation is slower than the code regime above (the k=7 drafter accepts about one token
in eight on prose); see `category-speeds-production-12.md` in this folder for the per-category
single-stream numbers.
