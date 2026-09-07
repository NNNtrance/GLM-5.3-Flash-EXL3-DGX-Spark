#!/usr/bin/env python3
"""Exact-repeat prefix-cache hit ratio, TTFT and draft acceptance, per request.

Why not the API's ``cached_tokens``: that field needs the server to be started
with ``--enable-cache-report``, which production is not.  The engine's own
Prometheus counters need no flag and are the same numbers the scheduler uses:

    vllm:prefix_cache_hits_total    / vllm:prefix_cache_queries_total
    vllm:spec_decode_num_accepted_tokens_total / ..._num_draft_tokens_total

Requests are issued one at a time and the counters are read either side of each
one, so every delta belongs to exactly one request.  A short settle poll covers
the gap between the last token arriving on the wire and the engine publishing
the counter.

CEILING.  A request of ``n`` tokens can never hit all of them -- at least one
token has to be recomputed -- and hits land on the coordinator's alignment
granularity ``G`` (3,328 tokens on this stack: the target MLA group's block).
So the best achievable ratio is ``floor((n-1)/G) * G / n``.  The probe reports
the raw ratio, that ceiling, and the quotient, because at 8k the ceiling is
83 % and a raw "83 %" there is a perfect score, not a failure.

    prefix-hit-probe.py --api http://HOST:8001 --label taban --out out.json

Sizes and repeat counts are fixed so two arms are comparable by construction.
"""

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request

DEFAULT_G = 3328

FILLER_WORDS = (
    "measurement discipline block cache prefix drafter acceptance latency "
    "throughput scheduler coordinator alignment granularity boundary token "
    "sequence attention indexer pooling window residual routing expert kernel "
    "profile roofline sidecar checkpoint quantisation fabric collective "
).split()


def make_filler(n_words: int, seed: int) -> str:
    rnd = random.Random(seed)
    return " ".join(rnd.choice(FILLER_WORDS) for _ in range(n_words))


def scrape(api: str, timeout: int = 20) -> dict:
    with urllib.request.urlopen(api + "/metrics", timeout=timeout) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name, _, rest = line.partition("{")
        if not rest:
            name, _, val = line.partition(" ")
            try:
                out[name] = out.get(name, 0.0) + float(val)
            except ValueError:
                pass
            continue
        _labels, _, val = rest.partition("} ")
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


KEYS = (
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
)


def wait_settled(api: str, before: dict, want_prompt: int, budget: float = 25.0) -> dict:
    """Poll until the prompt-token counter has moved by the whole request."""
    deadline = time.monotonic() + budget
    last = before
    while time.monotonic() < deadline:
        cur = scrape(api)
        moved = cur.get("vllm:prompt_tokens_total", 0) - before.get(
            "vllm:prompt_tokens_total", 0
        )
        if moved >= want_prompt * 0.98:
            return cur
        last = cur
        time.sleep(0.4)
    return last


def chat(api: str, messages: list, max_tokens: int, timeout: int = 900) -> dict:
    body = json.dumps(
        {
            "model": "glm-5.3-flash",
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        api + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    ttft = None
    usage = None
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.monotonic() - t0
                    text.append(piece)
    return {
        "ttft_s": ttft,
        "total_s": time.monotonic() - t0,
        "usage": usage or {},
        "text": "".join(text)[:400],
    }


def one(api: str, messages: list, max_tokens: int, g: int) -> dict:
    before = scrape(api)
    res = chat(api, messages, max_tokens)
    prompt_tokens = int(res["usage"].get("prompt_tokens") or 0)
    after = wait_settled(api, before, prompt_tokens or 1)
    d = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in KEYS}
    q = d["vllm:prefix_cache_queries_total"]
    h = d["vllm:prefix_cache_hits_total"]
    drafts = d["vllm:spec_decode_num_draft_tokens_total"]
    acc = d["vllm:spec_decode_num_accepted_tokens_total"]
    n = prompt_tokens or int(q)
    ceiling = ((n - 1) // g) * g / n if n else 0.0
    return {
        "prompt_tokens": n,
        "queries": q,
        "hits": h,
        "hit_ratio": (h / q) if q else None,
        "ceiling": ceiling,
        "of_ceiling": ((h / q) / ceiling) if (q and ceiling) else None,
        "ttft_s": res["ttft_s"],
        "total_s": res["total_s"],
        "draft_tokens": drafts,
        "accepted_tokens": acc,
        "acceptance": (acc / drafts) if drafts else None,
        "preemptions": d["vllm:num_preemptions_total"],
        "completion_tokens": int(res["usage"].get("completion_tokens") or 0),
        "sample": res["text"][:120],
    }


def scenario_repeat(api: str, words: int, repeats: int, seed: int, g: int, tag: str):
    body = make_filler(words, seed)
    prompt = (
        "Below is a reference document. Read it, then answer the question at the end.\n\n"
        "=== DOCUMENT ===\n" + body + "\n=== END DOCUMENT ===\n\n"
        "Question: reply with exactly the word ACK and nothing else."
    )
    msgs = [{"role": "user", "content": prompt}]
    rows = []
    for i in range(repeats):
        r = one(api, msgs, 8, g)
        r["round"] = i
        r["scenario"] = tag
        rows.append(r)
        print(
            f"  [{tag}] round {i}: n={r['prompt_tokens']:,} hit={r['hits']:,.0f}/"
            f"{r['queries']:,.0f} = "
            f"{(r['hit_ratio'] or 0) * 100:.1f}% (ceiling {r['ceiling'] * 100:.1f}%, "
            f"{(r['of_ceiling'] or 0) * 100:.1f}% of it) TTFT={r['ttft_s']}",
            flush=True,
        )
    return rows


def scenario_agent(api: str, words: int, turns: int, seed: int, g: int, tag: str):
    """Agent-style: the whole history is re-sent every turn, so turn k should hit
    everything turn k-1 computed."""
    body = make_filler(words, seed)
    msgs = [
        {
            "role": "system",
            "content": "You are a terse assistant. Answer in at most five words.",
        },
        {
            "role": "user",
            "content": "Reference material follows; keep it in mind.\n\n"
            + body
            + "\n\nSay READY.",
        },
    ]
    rows = []
    for t in range(turns):
        r = one(api, msgs, 24, g)
        r["round"] = t
        r["scenario"] = tag
        rows.append(r)
        print(
            f"  [{tag}] turn {t}: n={r['prompt_tokens']:,} hit="
            f"{(r['hit_ratio'] or 0) * 100:.1f}% (ceiling {r['ceiling'] * 100:.1f}%) "
            f"TTFT={r['ttft_s']}",
            flush=True,
        )
        msgs = msgs + [
            {"role": "assistant", "content": r["sample"] or "ok"},
            {
                "role": "user",
                "content": f"Follow-up number {t + 1}: name one word from the "
                f"reference material and stop.",
            },
        ]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://192.0.2.10:8001")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--granularity", type=int, default=DEFAULT_G)
    ap.add_argument("--skip", default="", help="comma list: r8k,r60k,agent")
    a = ap.parse_args()
    skip = {s for s in a.skip.split(",") if s}

    print(f"#### prefix-hit probe [{a.label}] {time.strftime('%H:%M:%S')}")
    rows = []
    # ~8k and ~60k tokens: this filler runs about 1.35 tokens per word.
    if "r8k" not in skip:
        rows += scenario_repeat(a.api, 7070, 4, 11, a.granularity, "repeat8k")
    if "r60k" not in skip:
        rows += scenario_repeat(a.api, 53050, 3, 22, a.granularity, "repeat60k")
    if "agent" not in skip:
        rows += scenario_agent(a.api, 7070, 4, 33, a.granularity, "agent4turn")

    summary = {}
    for tag in sorted({r["scenario"] for r in rows}):
        sel = [r for r in rows if r["scenario"] == tag and r["round"] > 0]
        if not sel:
            continue
        summary[tag] = {
            "repeat_hit_pct": round(
                100 * statistics.mean(r["hit_ratio"] or 0 for r in sel), 2
            ),
            "repeat_of_ceiling_pct": round(
                100 * statistics.mean(r["of_ceiling"] or 0 for r in sel), 2
            ),
            "repeat_ttft_s": round(
                statistics.mean(r["ttft_s"] or 0 for r in sel), 3
            ),
            "first_repeat_hit_pct": round(100 * (sel[0]["hit_ratio"] or 0), 2),
            "n_tokens": sel[0]["prompt_tokens"],
        }
    tot_d = sum(r["draft_tokens"] for r in rows)
    tot_a = sum(r["accepted_tokens"] for r in rows)
    summary["acceptance_pct"] = round(100 * tot_a / tot_d, 2) if tot_d else None
    summary["preemptions"] = sum(r["preemptions"] for r in rows)

    doc = {
        "label": a.label,
        "api": a.api,
        "granularity": a.granularity,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rows": rows,
        "summary": summary,
    }
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"#### written {a.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
