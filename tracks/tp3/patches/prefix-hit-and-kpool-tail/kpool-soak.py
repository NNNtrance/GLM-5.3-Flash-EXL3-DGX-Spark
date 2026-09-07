#!/usr/bin/env python3
"""Long-generation soak for the K-pool tail arm.

Position is what drives the tail bug: ``pos // block_size`` walks further past
the tail's one-column block-table row the longer a sequence gets, so the thing
to stress is GENERATION length, not prompt length.  Two phases:

  phase 1   eight concurrent generations of 4,096 tokens   (= max_num_seqs)
  phase 2   two concurrent generations of 8,192 tokens

The prompts are deliberately close to unsatisfiable so the model keeps working
to its budget instead of stopping early (the trick is vcruz305's).  Thinking is
left at the production setting; ``enable_thinking=false`` is never sent.

Coherence is checked cheaply and honestly: the output must be non-empty, must
reach at least 60 % of its token budget, must not degenerate into one repeated
token, and must stay in a sane character set.  This is a "did the engine stay
sane" test, not a benchmark.

    kpool-soak.py --api http://HOST:8001 --label b --out soak-b.json
"""

import argparse
import collections
import concurrent.futures as cf
import json
import time
import urllib.request

PROMPTS = [
    "Write a numbered list of 400 distinct one-sentence rules for a fictional "
    "city's public transport system. Every rule must be different from every "
    "other rule and must mention a different street name. Do not stop early.",
    "Enumerate, one per line, 400 different imaginary chemical compounds with a "
    "name, a colour and a melting point. No repetitions. Keep going until you "
    "have 400.",
    "Produce 400 numbered haiku about different tools in a workshop. Each haiku "
    "must name a tool no earlier haiku named.",
    "List 400 numbered fictional ship names with their home port and cargo. "
    "Every port must be different.",
    "Write 400 numbered short definitions for invented words in a constructed "
    "language, each with a part of speech and an example sentence.",
    "Give 400 numbered one-line descriptions of imaginary board games, each "
    "with a different number of players between 2 and 9.",
    "Write 400 numbered lines describing a different star in a fictional "
    "catalogue: designation, colour, distance.",
    "Produce 400 numbered fictional recipes, one line each: dish, one unusual "
    "ingredient, one cooking time.",
    "Write 400 numbered rules for an invented card game, each rule referring to "
    "a different card.",
    "List 400 numbered imaginary mountain peaks with height and first-ascent "
    "year, all different.",
]


def gen(api: str, prompt: str, max_tokens: int, idx: int) -> dict:
    body = json.dumps(
        {
            "model": "glm-5.3-flash",
            "messages": [{"role": "user", "content": prompt}],
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
    # coherence: not empty, reached most of the budget, not one token on repeat,
    # and mostly printable.
    # Degeneration check: the share of the most common *word*.  Separators and
    # one- or two-character tokens are excluded -- a numbered list of compound
    # names legitimately repeats "—" in every line, and counting that as
    # degeneration failed a perfectly good 1,024-token generation on the first
    # arm (8 September).
    words = [w for w in out.split() if len(w) > 2 and any(c.isalnum() for c in w)]
    top = collections.Counter(words).most_common(1)
    dominant = (top[0][1] / len(words)) if words else 1.0
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in out) / max(len(out), 1)
    # This model sometimes returns its whole answer in `reasoning_content` and
    # leaves `content` empty (the documented empty-content phenomenon).  A
    # request that spent its whole budget and streamed nothing is reported as
    # `content_empty` rather than incoherent: it is a client-visibility
    # question, not an engine-sanity one, and it is the same on both arms.
    content_empty = len(out) == 0 and completion >= 0.6 * max_tokens
    ok = (
        err is None
        and completion >= 0.6 * max_tokens
        and (content_empty or (len(out) > 500 and dominant < 0.35 and printable > 0.99))
    )
    return {
        "idx": idx,
        "max_tokens": max_tokens,
        "completion_tokens": completion,
        "seconds": round(time.monotonic() - t0, 1),
        "chars": len(out),
        "dominant_word_share": round(dominant, 4),
        "printable_share": round(printable, 5),
        "error": err,
        "content_empty": bool(content_empty),
        "coherent": bool(ok),
        "head": out[:150],
        "tail": out[-150:],
    }


def phase(api: str, n: int, max_tokens: int, offset: int, tag: str) -> list:
    print(f"  [{tag}] {n} concurrent x {max_tokens} tokens ...", flush=True)
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [
            ex.submit(gen, api, PROMPTS[(offset + i) % len(PROMPTS)], max_tokens, i)
            for i in range(n)
        ]
        rows = [f.result() for f in futs]
    for r in rows:
        print(
            f"    #{r['idx']}: {r['completion_tokens']}/{r['max_tokens']} tok in "
            f"{r['seconds']}s coherent={r['coherent']} err={r['error']}",
            flush=True,
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://192.0.2.10:8001")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p1", default="8x4096", help="phase 1 as <n>x<tokens>")
    ap.add_argument("--p2", default="2x8192", help="phase 2 as <n>x<tokens>; 'none' to skip")
    a = ap.parse_args()

    print(f"#### kpool soak [{a.label}] {time.strftime('%H:%M:%S')}")
    t0 = time.monotonic()
    n1, t1 = (int(x) for x in a.p1.split("x"))
    rows = phase(a.api, n1, t1, 0, "phase1")
    if a.p2 != "none":
        n2, t2 = (int(x) for x in a.p2.split("x"))
        rows += phase(a.api, n2, t2, n1, "phase2")
    alive = False
    try:
        with urllib.request.urlopen(a.api + "/health", timeout=15) as r:
            alive = r.status == 200
    except Exception:  # noqa: BLE001
        alive = False
    doc = {
        "label": a.label,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "minutes": round((time.monotonic() - t0) / 60, 1),
        "engine_alive_after": alive,
        "all_coherent": all(r["coherent"] for r in rows),
        "errors": [r["error"] for r in rows if r["error"]],
        "total_generated_tokens": sum(r["completion_tokens"] for r in rows),
        "rows": rows,
    }
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(
        f"#### SOAK [{a.label}] alive={alive} coherent={doc['all_coherent']} "
        f"tokens={doc['total_generated_tokens']:,} in {doc['minutes']} min"
    )


if __name__ == "__main__":
    main()
