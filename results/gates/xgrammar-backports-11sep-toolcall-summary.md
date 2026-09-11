# issue #7 — tool-call corruption measurement (2026-09-11 19:34)

- API: `http://127.0.0.1:8011` · model `glm-5.3-flash` · effort `low` · clear_thinking `True` · tool count 30
- sessions 4 · turns 120 · errors 0 · highest prompt_tokens 97319
- **sessions with first corruption: 0/4** · corrupt turn ratio 0.0

| context_bucket | turns | tool_calls | wellformed | rejected(b) | schema_violation(c) | json_error | empty_turn(d) | repeat(e) | length(f) |
|---|---|---|---|---|---|---|---|---|---|
| 0–10k | 13 | 15 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10–20k | 35 | 40 | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20–30k | 51 | 56 | 56 | 0 | 0 | 0 | 0 | 0 | 0 |
| 40–50k | 12 | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50–60k | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60–70k | 5 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 70–80k | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 90–100k | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| TOTAL | 120 | 132 | 132 | 0 | 0 | 0 | 0 | 0 | 0 |

## First corruption (per session)

- session 0: no corruption
- session 1: no corruption
- session 2: no corruption
- session 3: no corruption
