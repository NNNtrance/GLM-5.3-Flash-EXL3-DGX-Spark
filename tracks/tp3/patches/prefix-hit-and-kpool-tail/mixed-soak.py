#!/usr/bin/env python3
"""Fifteen-minute mixed soak: eight concurrent streams, code and prose, plus
two long generations.

kpool-soak.py stresses generation length on one kind of prompt.  This one
stresses the shape production actually sees: eight streams in flight the whole
time -- half of them code, half of them prose -- with two 4,096-token
generations riding alongside, so the scheduler is queueing as well as batching
(`max_num_seqs` is 8 and there are 10 in flight).

The point is not throughput.  It is to churn the KV pool, the prefix cache and
the K-pool tail for a quarter of an hour and then re-run the quality battery on
the engine in that state.

Coherence is the same cheap, honest check kpool-soak.py settled on: non-empty,
no single word dominating, printable.  The budget rule kpool-soak.py uses is
kept only for the two long generations, whose prompts are deliberately close to
unsatisfiable; a short code answer that finishes early is a correct answer, not
an incoherent one.  The documented empty-`content` phenomenon is reported
separately rather than counted as incoherent -- thinking is on, and this model
sometimes leaves `content` empty.

    mixed-soak.py --api http://HOST:8001 --minutes 15 --out soak.json
"""

import argparse
import collections
import concurrent.futures as cf
import json
import time
import urllib.request

CODE_PROMPTS = [
    "Write a Python class implementing an LRU cache with O(1) get and put, then "
    "a second implementation using only a dict and explain the trade-off. "
    "Include doctests for both.",
    "Implement Dijkstra's algorithm in Python with a binary heap, then write a "
    "property-based test that checks it against a brute-force shortest path on "
    "random small graphs.",
    "Write a small recursive-descent parser in Python for arithmetic with "
    "parentheses, unary minus and right-associative exponentiation. Include a "
    "tokenizer and ten unit tests.",
    "Write a Python context manager that retries a callable with exponential "
    "backoff and jitter, is safe under threads, and has tests using a fake clock.",
    "Implement a bitset-backed sparse matrix multiply in Python, with a "
    "correctness test against a dense reference and a short complexity note.",
    "Write a Python function that merges overlapping intervals, one that "
    "subtracts one interval list from another, and exhaustive tests for both.",
]

PROSE_PROMPTS = [
    "Write a numbered list of 200 distinct one-sentence rules for a fictional "
    "city's public transport system. Every rule must mention a different street.",
    "Explain, in careful plain English and about 1,200 words, how a "
    "write-back cache differs from a write-through cache, with three worked "
    "examples and a discussion of failure modes.",
    "Enumerate, one per line, 200 different imaginary chemical compounds with a "
    "name, a colour and a melting point. No repetitions.",
    "Write a 1,200-word essay on why measurement instruments need their own "
    "error bars before they can adjudicate anything, with three historical "
    "examples.",
    "Produce 200 numbered haiku about different tools in a workshop. Each haiku "
    "must name a tool no earlier haiku named.",
    "Describe, in about 1,000 words, the trade-offs between speculative "
    "decoding and larger batch sizes for a latency-sensitive service.",
]

LONG_PROMPTS = [
    "List 400 numbered fictional ship names with their home port and cargo. "
    "Every port must be different. Do not stop early.",
    "Write 400 numbered short definitions for invented words in a constructed "
    "language, each with a part of speech and an example sentence.",
]


def gen(api: str, prompt: str, max_tokens: int, kind: str, idx: int) -> dict:
    body = json.dumps(
        {
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
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
    text = []
    usage = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
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
                        text.append(piece)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    out = "".join(text)
    completion = int((usage or {}).get("completion_tokens") or 0)
    words = [w for w in out.split() if len(w) > 2 and any(c.isalnum() for c in w)]
    top = collections.Counter(words).most_common(1)
    dominant = (top[0][1] / len(words)) if words else 1.0
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in out) / max(len(out), 1)
    content_empty = len(out) == 0 and completion > 0
    # The budget rule applies only to the two long generations, whose prompts are
    # deliberately close to unsatisfiable.  A short code answer that finishes
    # early is a correct answer, not an incoherent one -- judging it by budget
    # would fail the soak on the model being concise.
    budget_ok = completion >= 0.6 * max_tokens if kind == "long" else completion > 0
    ok = (
        err is None
        and budget_ok
        and (content_empty or (len(out) > 300 and dominant < 0.35 and printable > 0.99))
    )
    return {
        "idx": idx,
        "kind": kind,
        "max_tokens": max_tokens,
        "completion_tokens": completion,
        "seconds": round(time.monotonic() - t0, 1),
        "chars": len(out),
        "dominant_word_share": round(dominant, 4),
        "error": err,
        "content_empty": bool(content_empty),
        "coherent": bool(ok),
        # kept so a row the heuristic flags can be read rather than guessed at
        "head": out[:400],
        "tail": out[-400:],
        "top_word": top[0] if top else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://192.0.2.10:8001")
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--label", default="mixed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream-tokens", type=int, default=1024)
    ap.add_argument("--long-tokens", type=int, default=4096)
    a = ap.parse_args()

    deadline = time.monotonic() + a.minutes * 60.0
    rows = []

    def worker(w: int) -> list:
        mine = []
        n = 0
        while time.monotonic() < deadline:
            if w % 2 == 0:
                prompt = CODE_PROMPTS[(w // 2 + n) % len(CODE_PROMPTS)]
                kind = "code"
            else:
                prompt = PROSE_PROMPTS[(w // 2 + n) % len(PROSE_PROMPTS)]
                kind = "prose"
            r = gen(a.api, prompt, a.stream_tokens, kind, w * 100 + n)
            mine.append(r)
            print(
                f"    w{w} {kind:5} #{n}: {r['completion_tokens']}/{r['max_tokens']} tok "
                f"in {r['seconds']}s coherent={r['coherent']} err={r['error']}",
                flush=True,
            )
            n += 1
        return mine

    def longgen(i: int) -> dict:
        r = gen(a.api, LONG_PROMPTS[i], a.long_tokens, "long", 9000 + i)
        print(
            f"    long#{i}: {r['completion_tokens']}/{r['max_tokens']} tok in "
            f"{r['seconds']}s coherent={r['coherent']} err={r['error']}",
            flush=True,
        )
        return r

    print(f"#### MIXED SOAK [{a.label}] {a.minutes} min, 8 streams + 2 long "
          f"{time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        futs = [ex.submit(worker, w) for w in range(8)]
        longs = [ex.submit(longgen, i) for i in range(2)]
        for f in futs:
            rows.extend(f.result())
        rows.extend(f.result() for f in longs)

    alive = False
    try:
        with urllib.request.urlopen(a.api + "/health", timeout=20) as r:
            alive = r.status == 200
    except Exception:  # noqa: BLE001
        alive = False

    total = sum(r["completion_tokens"] for r in rows)
    bad = [r for r in rows if not r["coherent"]]
    empt = [r for r in rows if r["content_empty"]]
    errs = [r for r in rows if r["error"]]
    mins = (time.monotonic() - t0) / 60.0
    print()
    print(f"requests {len(rows)}  tokens generated {total:,}  in {mins:.1f} min")
    print(f"engine alive after: {alive}")
    print(f"incoherent {len(bad)}   empty-content {len(empt)}   errors {len(errs)}")
    for r in bad:
        print(f"  INCOHERENT idx={r['idx']} kind={r['kind']} "
              f"{r['completion_tokens']}/{r['max_tokens']} err={r['error']}")
    verdict = alive and not bad and not errs
    print(f"MIXED SOAK: {'PASS' if verdict else 'FAIL'}")
    with open(a.out, "w") as f:
        json.dump(
            {"label": a.label, "minutes": round(mins, 2), "rows": rows,
             "tokens": total, "alive": alive, "incoherent": len(bad),
             "empty_content": len(empt), "errors": len(errs), "pass": verdict},
            f, indent=1,
        )
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
