#!/usr/bin/env python3
"""Why one exact repeat reaches the ceiling and another does not: the offset.

The alignment granularity is 3,328 tokens. A request of `n` tokens sits
`n mod 3328` tokens past its last aligned boundary, and that offset decides
whether the drafter's sliding-window group has a cached block *beyond* the
boundary to give back when it takes the EAGLE drop. Its block is 256 tokens; if
the offset is smaller than that, the drop has nothing to eat and the
re-alignment pop costs a whole 3,328-token block instead.

This sweep holds everything else fixed and varies only the offset.

    offset-sweep.py --api http://HOST:8001 --label ab --out offsets-ab.json
"""

import argparse
import json
import random
import time
import urllib.request

G = 3328
WORDS = (
    "measurement discipline block cache prefix drafter acceptance latency "
    "throughput scheduler coordinator alignment granularity boundary token "
    "sequence attention indexer pooling window residual routing expert kernel "
).split()


def filler(n_words: int, seed: int) -> str:
    rnd = random.Random(seed)
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))


def scrape(api):
    with urllib.request.urlopen(api + "/metrics", timeout=20) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        name, _, rest = line.partition("{")
        val = rest.partition("} ")[2] if rest else line.partition(" ")[2]
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def one(api, prompt):
    body = json.dumps(
        {
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        api + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    before = scrape(api)
    t0 = time.monotonic()
    ttft = None
    usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content")) and ttft is None:
                    ttft = time.monotonic() - t0
    n = int((usage or {}).get("prompt_tokens") or 0)
    deadline = time.monotonic() + 25
    after = before
    while time.monotonic() < deadline:
        after = scrape(api)
        if after.get("vllm:prompt_tokens_total", 0) - before.get(
            "vllm:prompt_tokens_total", 0
        ) >= n * 0.98:
            break
        time.sleep(0.4)
    q = after.get("vllm:prefix_cache_queries_total", 0) - before.get(
        "vllm:prefix_cache_queries_total", 0
    )
    h = after.get("vllm:prefix_cache_hits_total", 0) - before.get(
        "vllm:prefix_cache_hits_total", 0
    )
    return n, q, h, ttft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://192.0.2.10:8001")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--words", default="52930,53600,54300,55000", help="comma list of word counts"
    )
    a = ap.parse_args()
    rows = []
    print(f"#### offset sweep [{a.label}] {time.strftime('%H:%M:%S')}")
    for i, w in enumerate(int(x) for x in a.words.split(",")):
        body = filler(w, 700 + i)
        prompt = (
            "Reference document follows. Answer the question at the end.\n\n"
            "=== DOCUMENT ===\n" + body + "\n=== END DOCUMENT ===\n\n"
            "Question: reply with exactly the word ACK and nothing else."
        )
        n0, _, _, _ = one(a.api, prompt)          # cold
        n, q, h, ttft = one(a.api, prompt)        # exact repeat
        ceiling = ((n - 1) // G) * G / n if n else 0
        row = {
            "words": w,
            "n_tokens": n,
            "offset_past_boundary": n % G,
            "hit_tokens": h,
            "hit_pct": round(100 * h / q, 2) if q else None,
            "ceiling_pct": round(100 * ceiling, 2),
            "of_ceiling_pct": round(100 * (h / q) / ceiling, 2) if q and ceiling else None,
            "blocks_hit": round(h / G, 2),
            "blocks_ceiling": (n - 1) // G,
            "ttft_s": round(ttft, 3) if ttft else None,
        }
        rows.append(row)
        print(
            f"  n={n:,} offset={row['offset_past_boundary']:,} "
            f"hit={row['hit_pct']}% of ceiling {row['ceiling_pct']}% "
            f"-> {row['of_ceiling_pct']}%  blocks {row['blocks_hit']}/"
            f"{row['blocks_ceiling']}  TTFT={row['ttft_s']}s",
            flush=True,
        )
    with open(a.out, "w") as f:
        json.dump({"label": a.label, "granularity": G, "rows": rows}, f, indent=2)
    print(f"#### written {a.out}")


if __name__ == "__main__":
    main()
