#!/usr/bin/env python3
"""Erken bitiş A/B — uzun biçimli üretimde spec açık/kapalı kıyası (HAREM, 10 Eyl 2026).
Her istem en az N kelime ister; 'erken bitiş' = finish_reason=stop VE içerik kelimesi < 0,4·N.
kullanım: erken-bitis.py --tag specon --top-p 0.95 [--temperature 1.0] [--concurrency 4] [--max-tokens 8192]"""
import argparse, json, time, urllib.request, concurrent.futures as cf, statistics, pathlib, sys
P = []
def add(min_words, text): P.append((min_words, text))
ART = "Write a complete, self-contained technical article of at least {w} words about {t}. Use section headings, worked examples and a closing summary. Do not stop early; the article must be at least {w} words."
for t in ["TCP congestion control from Tahoe to BBR", "tracing garbage collectors and generational heaps", "how DNS resolution and caching actually work end to end",
          "CRDTs and why they converge", "Kalman filters for a self-driving toy car", "the GPU memory hierarchy and why bandwidth dominates inference",
          "how a compiler turns C into machine code, pass by pass", "RAID levels, failure math and rebuild risk"]:
    add(2500, ART.format(w=2500, t=t))
STORY = "Write a short story of at least {w} words. {t} Give it a beginning, a middle with rising tension, and a proper ending. Do not stop before {w} words."
for t in ["A lighthouse keeper discovers the light has started sending messages.", "Two rival street food cooks are forced to share one cart for a month.",
          "A retired astronaut teaches a village to read the night sky.", "A translator finds a word that does not exist in any language she knows.",
          "A city where every citizen must forget one memory a year.", "A chess prodigy loses to a child and has to understand why."]:
    add(2500, STORY.format(w=2500, t=t))
CODE = "Write a complete, well-documented Python module implementing {t}. Include docstrings, type hints, a thorough unit test suite using unittest, and a usage example. The file must be at least 400 lines long; do not abbreviate or truncate."
for t in ["an LRU cache with TTL support and statistics", "a tiny Markdown-to-HTML converter", "a rate limiter with token bucket and sliding window strategies",
          "a priority job scheduler with dependencies and retries", "an in-memory key-value store with transactions and snapshots", "a CSV parser that handles quoting, escaping and streaming"]:
    add(1800, CODE.format(t=t))
DOC = "Write a complete document of at least {w} words: {t}. Use headings and numbered sections; do not stop before {w} words."
for t in ["a game design document for a co-op roguelike about beekeeping", "a product specification for a family calendar app with offline sync",
          "a six-lesson plan teaching teenagers basic statistics", "a detailed travel guide for four days in Istanbul on a modest budget"]:
    add(2500, DOC.format(w=2500, t=t))

def one(i, url, args):
    mw, prompt = P[i]
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens}
    t0 = time.time()
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=2400) as r: d = json.load(r)
    except Exception as e:
        return {"i": i, "error": f"{type(e).__name__}: {e}"[:160], "secs": round(time.time() - t0, 1)}
    c = d["choices"][0]; content = c["message"].get("content") or ""; reasoning = c["message"].get("reasoning_content") or ""
    words = len(content.split())
    return {"i": i, "finish": c["finish_reason"], "completion_tokens": d["usage"]["completion_tokens"], "words": words, "min_words": mw,
            "reasoning_chars": len(reasoning), "tail": content[-90:].replace("\n", " "), "secs": round(time.time() - t0, 1),
            "early": c["finish_reason"] == "stop" and words < 0.4 * mw}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--url", default="http://192.168.1.103:8001")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    out = pathlib.Path(args.out) / f"{args.tag}-p{args.top_p}.jsonl"
    rows = []
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        for r in ex.map(lambda i: one(i, args.url, args), range(len(P))):
            rows.append(r); out.open("a").write(json.dumps(r, ensure_ascii=False) + "\n"); print(json.dumps(r, ensure_ascii=False)[:150], flush=True)
    ok = [r for r in rows if "error" not in r]
    summ = {"tag": args.tag, "top_p": args.top_p, "temperature": args.temperature, "n": len(rows), "errors": len(rows) - len(ok),
            "early_stops": sum(1 for r in ok if r["early"]), "finish": {k: sum(1 for r in ok if r["finish"] == k) for k in {r["finish"] for r in ok}},
            "median_words": statistics.median(r["words"] for r in ok) if ok else 0, "median_tokens": statistics.median(r["completion_tokens"] for r in ok) if ok else 0,
            "min_words_seen": min((r["words"] for r in ok), default=0), "total_secs": round(sum(r["secs"] for r in rows), 0)}
    (pathlib.Path(args.out) / f"{args.tag}-p{args.top_p}-ozet.json").write_text(json.dumps(summ, indent=1))
    print("ÖZET", json.dumps(summ))
