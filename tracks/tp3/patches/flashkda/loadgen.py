#!/usr/bin/env python3
"""The prefill stopwatch: fresh nonce-prefixed prompts, 24 in flight, a 60 s window.

Written for the FlashKDA A/B of 10 September 2026 and used for every prefill
number in results/gates/flashkda-ab-10sep.md.  It is a THROUGHPUT instrument and
a different measurement from bench/prefill-fresh.py, which times one request at a
time: this one reads the engine's own counters under sustained load, so the number
it returns is what the cluster actually sustains with 24 prompts queued.  Either
way the prompt is fresh -- a prefill figure from a repeated prompt is not a
prefill figure (docs/09 section 3), and this script proves freshness from the
prefix-cache counters rather than asserting it.

Keeps N (default 24) requests in flight against the production engine, every
one of them a FRESH, unique ~7,000-token English prompt, so nothing can be read
out of the prefix cache.  Prefill throughput is the delta of
``vllm:prompt_tokens_total`` over a WINDOW-second window that opens SETTLE
seconds after the load began; the prefix-cache counters are sampled over the
same window and reported, so a hit cannot hide inside the number.

Python stdlib only.  Each request asks for max_tokens=8 at temperature 0, so
decode work is negligible and the engine is prefill-bound.

    loadgen.py stopwatch  [API] [--conc 24] [--settle 20] [--window 60]
                          [--runtime 100] [--json out.json] [--tag name]
    loadgen.py ttft       [API] [--reps 5]        single-stream TTFT, median
    loadgen.py decode     [API] [--max-tokens 512]  single-stream decode tok/s

Writes nothing except the file named by --json.
"""

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.request

# Varied English prose vocabulary.  Deliberately ordinary words: the point is a
# long, unique, low-entropy-per-word document, not a tokenizer stress test.
WORDS = """
harbour lantern gravel meadow cinder ledger anchor thistle furnace pebble willow
cavern marble drifting quarry beacon thicket ember lattice hollow shutter timber
kettle cobble lichen bramble rafter trellis mortar pewter sable orchard plinth
gable vellum cistern conduit furrow granary heather ingot juniper keystone
limestone millrace nettle oakum parapet quayside reed sandstone tannery urn
vestibule wainscot yardarm zenith almanac bellows coracle dovetail escarpment
fathom gantry halyard inkwell jetty kiln loam mainsay nocturne oilskin pantile
quern rigging scupper tiller undertow valance windlass arbour brazier cordage
dormer earthworks flagstone girder hawser iron joists kelp lodestone mizzen
nightjar osier purlin quoin ratchet spandrel transom upland vane weir
""".split()

SENTENCE_HEADS = [
    "The surveyor noted that",
    "By the second week",
    "Records from the adjacent parish show",
    "A later correction explains",
    "In practice the crew found",
    "The ledger entry reads",
    "Observers downstream reported",
    "It was agreed at the meeting",
    "According to the shipping list",
    "The foreman's own account says",
    "Measurements taken at dawn suggest",
    "One inspector disagreed, arguing",
]
SENTENCE_TAILS = [
    "and the figure was never revised.",
    "though the margin of error was wide.",
    "which settled the matter for that season.",
    "before the weather closed in again.",
    "so the work carried on without pause.",
    "and a second reading confirmed it.",
    "despite the missing page in the register.",
    "and the cost was charged to the estate.",
]


def make_prompt(rnd: random.Random, words: int = 4460) -> str:
    """A unique pseudo-random English document of roughly `words` words.

    The first thing in the prompt is a 24-hex-digit nonce, so even the FIRST
    prefix-cache block differs between requests and no block can ever be shared.
    """
    nonce = "".join(rnd.choice("0123456789abcdef") for _ in range(24))
    out = [
        f"Document reference {nonce}. Read the survey notes below and then "
        "answer with a single word.\n\n"
    ]
    n = 0
    para = []
    while n < words:
        head = rnd.choice(SENTENCE_HEADS)
        body = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(9, 19)))
        tail = rnd.choice(SENTENCE_TAILS)
        s = f"{head} {body} {tail}"
        n += s.count(" ") + 1
        para.append(s)
        if len(para) >= 7:
            out.append(" ".join(para) + "\n\n")
            para = []
    if para:
        out.append(" ".join(para) + "\n\n")
    out.append("Question: in one word, what kind of notes were these? Answer:")
    return "".join(out)


def post(api, body, timeout=900, stream=False):
    req = urllib.request.Request(
        api + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    if stream:
        return urllib.request.urlopen(req, timeout=timeout)
    with urllib.request.urlopen(req, timeout=timeout) as h:
        return json.load(h)


METRIC_KEYS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:request_success_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
)


def metrics(api):
    with urllib.request.urlopen(api + "/metrics", timeout=30) as h:
        txt = h.read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or "{" not in line:
            continue
        name = line.split("{", 1)[0]
        if name in METRIC_KEYS:
            try:
                val = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            out[name] = out.get(name, 0.0) + val
    out["_t"] = time.monotonic()
    return out


def chat_body(prompt, max_tokens, stream=False):
    b = {
        "model": "glm-5.3-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
    }
    if stream:
        b["stream"] = True
    return b


# ---------------------------------------------------------------- stopwatch
def stopwatch(a):
    api = a.api
    stop = threading.Event()
    lock = threading.Lock()
    sent = {"ok": 0, "err": 0, "ptok": [], "errs": []}

    def worker(wid):
        rnd = random.Random((os.getpid() * 7919) ^ (wid * 104729) ^ time.time_ns())
        while not stop.is_set():
            try:
                d = post(api, chat_body(make_prompt(rnd, a.words), 8), timeout=600)
                pt = d.get("usage", {}).get("prompt_tokens", 0)
                with lock:
                    sent["ok"] += 1
                    sent["ptok"].append(pt)
            except Exception as e:  # noqa: BLE001 -- record, keep the load on
                with lock:
                    sent["err"] += 1
                    if len(sent["errs"]) < 5:
                        sent["errs"].append(repr(e)[:200])

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(a.conc)]
    for t in threads:
        t.start()

    time.sleep(max(0.0, a.settle - (time.monotonic() - t0)))
    m0 = metrics(api)
    time.sleep(a.window)
    m1 = metrics(api)
    dt = m1["_t"] - m0["_t"]

    stop.set()
    # let the in-flight requests drain rather than abandoning sockets
    for t in threads:
        t.join(timeout=180)

    d_prompt = m1["vllm:prompt_tokens_total"] - m0["vllm:prompt_tokens_total"]
    d_gen = m1["vllm:generation_tokens_total"] - m0["vllm:generation_tokens_total"]
    d_q = m1["vllm:prefix_cache_queries_total"] - m0["vllm:prefix_cache_queries_total"]
    d_h = m1["vllm:prefix_cache_hits_total"] - m0["vllm:prefix_cache_hits_total"]
    res = {
        "tag": a.tag,
        "mode": "stopwatch",
        "conc": a.conc,
        "settle_s": a.settle,
        "window_s": round(dt, 3),
        "prefill_tok_per_s": round(d_prompt / dt, 1),
        "decode_tok_per_s_during": round(d_gen / dt, 1),
        "prompt_tokens_in_window": int(d_prompt),
        "prefix_cache_queries_in_window": int(d_q),
        "prefix_cache_hits_in_window": int(d_h),
        "requests_ok": sent["ok"],
        "requests_err": sent["err"],
        "errors": sent["errs"],
        "prompt_tokens_mean": round(statistics.fmean(sent["ptok"]), 1) if sent["ptok"] else None,
        "prompt_tokens_min": min(sent["ptok"]) if sent["ptok"] else None,
        "prompt_tokens_max": max(sent["ptok"]) if sent["ptok"] else None,
        "running_at_window_open": m0.get("vllm:num_requests_running"),
        "waiting_at_window_open": m0.get("vllm:num_requests_waiting"),
    }
    return res


# ---------------------------------------------------------------------- ttft
def ttft(a):
    api = a.api
    rnd = random.Random(time.time_ns())
    vals, ptoks = [], []
    for _ in range(a.reps):
        prompt = make_prompt(rnd, a.words)
        t = time.monotonic()
        first = None
        h = post(api, chat_body(prompt, 8, stream=True), timeout=900, stream=True)
        try:
            for raw in h:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    ch = json.loads(payload)["choices"][0]["delta"]
                except Exception:  # noqa: BLE001
                    continue
                if ch.get("content") or ch.get("reasoning") or ch.get("reasoning_content"):
                    first = time.monotonic() - t
                    break
        finally:
            h.close()
        if first is None:
            first = time.monotonic() - t
        vals.append(first)
        # token count of the same prompt, cheaply, via /tokenize
        try:
            req = urllib.request.Request(
                api + "/tokenize",
                json.dumps({"model": "glm-5.3-flash", "prompt": prompt}).encode(),
                {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                ptoks.append(json.load(r).get("count"))
        except Exception:  # noqa: BLE001
            pass
    vals.sort()
    return {
        "tag": a.tag,
        "mode": "ttft",
        "reps": a.reps,
        "ttft_s": [round(v, 3) for v in vals],
        "ttft_median_s": round(statistics.median(vals), 3),
        "ttft_mean_s": round(statistics.fmean(vals), 3),
        "prompt_tokens": ptoks,
    }


# -------------------------------------------------------------------- decode
DECODE_PROMPT = (
    "Write a single self-contained Python module that implements a small "
    "in-memory key-value store with TTL expiry, an LRU eviction policy, and a "
    "thread-safe API (get, set, delete, stats). Include docstrings and type "
    "hints. Output only code."
)


def decode(a):
    api = a.api
    out = []
    for _ in range(a.reps):
        t = time.monotonic()
        d = post(api, chat_body(DECODE_PROMPT, a.max_tokens), timeout=900)
        dt = time.monotonic() - t
        u = d.get("usage", {})
        ct = u.get("completion_tokens", 0)
        out.append({
            "wall_s": round(dt, 3),
            "completion_tokens": ct,
            "prompt_tokens": u.get("prompt_tokens"),
            "tok_per_s": round(ct / dt, 2),
        })
    rates = sorted(r["tok_per_s"] for r in out)
    return {
        "tag": a.tag,
        "mode": "decode",
        "reps": a.reps,
        "max_tokens": a.max_tokens,
        "runs": out,
        "tok_per_s_median": rates[len(rates) // 2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stopwatch", "ttft", "decode"])
    ap.add_argument("api", nargs="?", default="http://192.0.2.10:8001")
    ap.add_argument("--conc", type=int, default=24)
    ap.add_argument("--settle", type=float, default=20.0)
    ap.add_argument("--window", type=float, default=60.0)
    ap.add_argument("--words", type=int, default=4460)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--json", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    res = {"stopwatch": stopwatch, "ttft": ttft, "decode": decode}[a.mode](a)
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
