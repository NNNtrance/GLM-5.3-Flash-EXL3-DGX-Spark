#!/usr/bin/env python3
"""Cached-path equality test.

The question this answers is the one the 8 September night could not settle:
when a prompt is served partly from the prefix cache, does the model give the
*same* answer it gave when it computed the whole thing?  Any prefix-cache or
K-pool-tail change that corrupts recall would show up exactly here -- a fluent,
different, wrong answer on the second pass.

Design
------
24 needle-style prompts: a unique fact planted in unique filler, eight at each
of three sizes (~8k, ~60k, ~128k tokens), at eight different depths.  Every
prompt has its own random seed, so no two prompts share a prefix beyond the
chat template -- pass 1 is genuinely cold for each of them.

Two orderings, and the difference between them turned out to matter:

  --order passes      pass 1 = all 24 cold, pass 2 = the same 24 repeated.
  --order interleaved cold and repeat back to back, one prompt at a time.

`passes` is the obvious design and it does NOT work on this stack: by the time
prompt #1 comes round again, twenty-three long requests have gone through and
its blocks are gone, so the "repeat" recomputes everything and the test proves
nothing about the cached path.  `interleaved` is what actually holds the cache,
and it is the mode that answers the question.

The engine's own Prometheus counters are read either side of every request, so
each request's prefix-cache hit belongs to exactly one request.  Pass 2 must be
a hit; if it is not, the test says so rather than quietly passing.

Verdict
-------
PASS when, for all 24: the repeat answer is byte-identical to the cold answer
AND the answer is correct (the planted code appears in `content`).  One pair
may differ in wording while both are correct -- that is the tolerance the
protocol allows; a second such pair fails the test.  Any wrong answer, on
either pass, fails.

Thinking stays on (`reasoning_effort: low`, the production setting);
`enable_thinking=false` is never sent -- it is banned on this stack.  Scoring is
on `content`, as correctness-probe.py and needle-lite6.py do, with `reasoning`
reported separately.

    cached-equality.py --api http://HOST:8001 --out equality.json
"""

import argparse
import json
import random
import sys
import time
import urllib.request

WORDS = (
    "harbour lantern gravel meadow cinder ledger anchor thistle furnace pebble "
    "willow cavern marble drifting quarry beacon thicket ember lattice hollow "
    "granite tumbling saffron mackerel bramble kestrel obsidian juniper "
    "tessellate meridian lodestone parapet vellum cistern brackish gantry "
).split()

# One distinct code per prompt.  Deliberately unguessable and unrelated to each
# other, so an invented answer cannot accidentally score.
CODES = [
    "QX7-4412", "ZM3-9087", "KR8-2251", "TV5-6630", "BN2-3374", "LP9-8105",
    "HD4-5719", "WS6-1928", "JC2-8460", "RF8-3095", "GY5-7284", "NM1-6613",
    "AE9-2047", "PU3-9556", "XO7-1382", "DK4-4870", "VT2-3169", "CB6-7725",
    "SL8-5031", "MQ1-8894", "IZ5-2607", "FW9-4318", "OH3-6952", "EN7-1476",
]

# words -> tokens is about 1.3 on this filler; measured against prompt_tokens
# and reported per row, so the labels below are nominal sizes only.
SIZES = [("8k", 6_200), ("60k", 46_000), ("128k", 98_000)]
DEPTHS = [0.02, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.97]

COUNTERS = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total")


def scrape(api: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(api + "/metrics", timeout=timeout) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name, _, rest = line.partition("{")
        if rest:
            name = name.strip()
            val = rest.rpartition("}")[2].strip()
        else:
            name, _, val = line.partition(" ")
        try:
            out[name.strip()] = out.get(name.strip(), 0.0) + float(val)
        except ValueError:
            continue
    return {k: out.get(k, 0.0) for k in COUNTERS}


def build(idx: int, seed_offset: int = 0) -> tuple:
    """Return (prompt, code, nominal_size_label)."""
    size_i, depth_i = divmod(idx, 8)
    label, n_words = SIZES[size_i]
    code = CODES[idx]
    rnd = random.Random(90_000 + seed_offset + idx * 7919)
    words = [rnd.choice(WORDS) for _ in range(n_words)]
    at = int(len(words) * DEPTHS[depth_i])
    words[at:at] = f"The vault code for room {idx + 1} is {code} .".split()
    hay = " ".join(words)
    prompt = (
        "You are given a long document. Somewhere inside it there is a line of the "
        f"form 'The vault code for room {idx + 1} is XXX-NNNN'. Find it.\n\n"
        "=== DOCUMENT ===\n" + hay + "\n=== END DOCUMENT ===\n\n"
        f"Reply with the vault code for room {idx + 1} and nothing else."
    )
    return prompt, code, label


def ask(api: str, prompt: str, timeout: int) -> dict:
    body = {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
    }
    req = urllib.request.Request(
        api + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    return {
        "content": (msg.get("content") or "").strip(),
        "reasoning": (msg.get("reasoning_content") or "").strip(),
        "prompt_tokens": int(d["usage"]["prompt_tokens"]),
        "seconds": round(time.monotonic() - t0, 2),
    }


def settle(api: str, before: dict, tries: int = 8) -> dict:
    """Poll until the counters move -- the wire finishes before the scrape."""
    last = before
    for _ in range(tries):
        cur = scrape(api)
        if cur["vllm:prefix_cache_queries_total"] > before["vllm:prefix_cache_queries_total"]:
            return cur
        last = cur
        time.sleep(1.0)
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://192.0.2.10:8001")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="equality")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--order", choices=("passes", "interleaved"), default="interleaved")
    ap.add_argument("--seed-offset", type=int, default=0)
    a = ap.parse_args()

    print(f"#### CACHED-PATH EQUALITY [{a.label}] order={a.order} "
          f"{time.strftime('%H:%M:%S')}", flush=True)
    cases = [build(i, a.seed_offset) for i in range(24)]

    rows = [{"idx": i, "code": c, "size": s} for i, (_, c, s) in enumerate(cases)]

    def one(i: int, pas: str) -> None:
        prompt, code, label = cases[i]
        b = scrape(a.api)
        try:
            r = ask(a.api, prompt, a.timeout)
            err = None
        except Exception as e:  # noqa: BLE001
            r = {"content": "", "reasoning": "", "prompt_tokens": 0, "seconds": 0.0}
            err = f"{type(e).__name__}: {e}"
        af = settle(a.api, b)
        dq = af["vllm:prefix_cache_queries_total"] - b["vllm:prefix_cache_queries_total"]
        dh = af["vllm:prefix_cache_hits_total"] - b["vllm:prefix_cache_hits_total"]
        ratio = (dh / dq) if dq > 0 else 0.0
        rows[i][pas] = {
            "content": r["content"],
            "correct": code in r["content"],
            "either": (code in r["content"]) or (code in r["reasoning"]),
            "prompt_tokens": r["prompt_tokens"],
            "seconds": r["seconds"],
            "hit_queries": int(dq),
            "hit_tokens": int(dh),
            "hit_ratio": round(ratio, 4),
            "error": err,
        }
        print(
            f"  [{pas}] #{i + 1:02d} {label:>4} n={r['prompt_tokens']:>7,} "
            f"hit={ratio * 100:5.1f}% {r['seconds']:6.1f}s "
            f"{'OK ' if code in r['content'] else 'BAD'} {r['content'][:32]!r}"
            + (f" ERR {err}" if err else ""),
            flush=True,
        )

    if a.order == "passes":
        for pas in ("cold", "repeat"):
            print(f"-- pass: {pas} {time.strftime('%H:%M:%S')}", flush=True)
            for i in range(24):
                one(i, pas)
    else:
        for i in range(24):
            one(i, "cold")
            one(i, "repeat")

    # ---- verdict -----------------------------------------------------------
    identical = 0
    differing_but_correct = []
    wrong = []
    no_hit = []
    for r in rows:
        c, p = r["cold"], r["repeat"]
        if not c["correct"] or not p["correct"]:
            wrong.append(r["idx"] + 1)
        elif c["content"] == p["content"]:
            identical += 1
        else:
            differing_but_correct.append(r["idx"] + 1)
        # a repeat that hit no more of its prompt than the cold pass did is not
        # exercising the cached path at all, and the test would prove nothing
        if p["hit_ratio"] <= c["hit_ratio"] + 0.01:
            no_hit.append(r["idx"] + 1)

    print()
    print(f"identical and correct .......... {identical}/24")
    print(f"differing but both correct ..... {len(differing_but_correct)}  {differing_but_correct}")
    print(f"wrong on either pass ........... {len(wrong)}  {wrong}")
    print(f"repeat did NOT hit the cache ... {len(no_hit)}  {no_hit}")
    cold_mean = sum(r["cold"]["hit_ratio"] for r in rows) / 24
    rep_mean = sum(r["repeat"]["hit_ratio"] for r in rows) / 24
    print(f"mean hit ratio: cold {cold_mean * 100:.1f}%  repeat {rep_mean * 100:.1f}%")

    ok = (not wrong) and (not no_hit) and len(differing_but_correct) <= 1
    print(f"EQUALITY TEST: {'PASS' if ok else 'FAIL'} "
          f"({identical + len(differing_but_correct)}/24 correct, "
          f"{identical} byte-identical)")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(
                {
                    "label": a.label,
                    "rows": rows,
                    "identical": identical,
                    "differing_but_correct": differing_but_correct,
                    "wrong": wrong,
                    "no_hit": no_hit,
                    "mean_hit_cold": cold_mean,
                    "mean_hit_repeat": rep_mean,
                    "pass": ok,
                },
                f,
                indent=1,
            )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
