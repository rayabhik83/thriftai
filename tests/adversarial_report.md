# Adversarial cache pairs

This file is regenerated when `tests/test_adversarial.py` runs against a real embedding model. To populate it:

```
OPENAI_API_KEY=… THRIFTAI_LIVE_TEST=1 \
  pytest tests/test_adversarial.py -m live
```

(Override the model with `THRIFTAI_EMBED_MODEL=…`. Default is `text-embedding-3-small`.)

The result table has one row per `(category, query_a, query_b)` pair from `tests/adversarial_pairs.py`, with cosine similarity and HIT/MISS verdicts at thresholds 0.85, 0.92, 0.95.

## Reading the table

- **`HIT ✓`** — the cache correctly served a paraphrase response (positive control).
- **`HIT ✗`** — the cache **would have served the wrong response** (wrong-hit hazard). Anywhere this appears, the cache is a real production hazard at that threshold.
- **`MISS ✓`** — the threshold correctly rejected a distinct query.
- **`MISS ✗`** — the threshold rejected a legitimate paraphrase (false miss; you're paying for an LLM call you didn't need to make).

Tune `Session(semantic_threshold=…)` based on which threshold column matches your tolerance. Higher threshold → fewer wrong-hits, more false misses.

## Categories under test

See `tests/adversarial_pairs.py` for the full list and rationale per pair. Headline categories:

- **Antonym** — opposite-valence answers, single-word flip in a long template.
- **Negation** — `is X?` vs `is X not?` — embeddings are notoriously bad at this.
- **Numeric drift** — different numbers, same surrounding text (`2+2` vs `2+3`, `top 3` vs `top 30`).
- **Entity swap** — different proper nouns (`CEO of Apple` vs `CEO of Google`).
- **Date drift** — time-bearing queries that ask about a different period.
- **Boilerplate dilution** — operative one-word flip drowned out by ~200 tokens of shared preamble.
- **Code op flip** — opposite arithmetic operator in otherwise identical code.
- **Crosslingual paraphrase** — same answer, different language (gray area; documented, not asserted).
- **English paraphrase** — positive control; the case the cache exists for.
