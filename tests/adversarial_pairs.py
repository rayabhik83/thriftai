"""
Adversarial pairs for the semantic cache.

Each entry is a `(query_a, query_b)` pair the cache might be asked to match.
`should_match` says whether they SHOULD be served the same response:

- `True`  → positive control. The cache is *supposed* to hit. Asserted in the test.
- `False` → wrong-hit hazard. Distinct meaning; the cache must NOT serve A's
            response for B. We **measure** but don't assert — embeddings will
            fail some of these and the report exists to document which.
- `None`  → gray area. Whether they should match depends on policy
            (e.g. cross-lingual). Documented, not asserted.

`notes` shows up in the generated report so users can see *why* each pair
matters.

The list is data only — `tests/test_adversarial.py` consumes it.
"""

# ~200-token enterprise-procurement-ish boilerplate. Used to drown out the
# operative token in the boilerplate-dilution attack: A and B differ only in
# the final word, but the long shared preamble dominates the embedding.
LONG_PREAMBLE = (
    "You are an enterprise procurement assistant operating under our published "
    "vendor risk and contract review policy. Carefully evaluate the following "
    "request against our standard checklist: contract value, data residency, "
    "subprocessor exposure, indemnification scope, termination rights, "
    "renewal cadence, and applicable regulatory frameworks (SOC 2 Type II, "
    "ISO 27001, GDPR, CCPA, HIPAA where relevant). Cross-reference the "
    "vendor's published security documentation against our internal risk "
    "register. Note any deviation from our standard contract template. "
    "Surface red flags in the order they appear, with remediation suggestions "
    "where appropriate. Conclude with a single-line decision in the form of "
    "either APPROVE or REJECT, followed by a one-sentence justification."
)


ADVERSARIAL_PAIRS = [
    # ─────────────────────────────────────────────────────────────────────
    # WRONG-HIT HAZARDS (should_match=False)
    # ─────────────────────────────────────────────────────────────────────
    (
        "antonym_attribute",
        "List five reasons coffee is healthy.",
        "List five reasons coffee is unhealthy.",
        False,
        "opposite-valence answers; long shared template, single-word flip",
    ),
    (
        "negation",
        "Is Python a statically typed language?",
        "Is Python not a statically typed language?",
        False,
        "single-word negation flips the answer; embeddings notoriously miss this",
    ),
    (
        "numeric_drift_arithmetic",
        "Compute 2 + 2 step by step.",
        "Compute 2 + 3 step by step.",
        False,
        "different correct answer (4 vs 5); lexically near-identical",
    ),
    (
        "numeric_drift_quantifier",
        "What are the top 3 risks of agentic AI?",
        "What are the top 30 risks of agentic AI?",
        False,
        "answer set differs by an order of magnitude (3 items vs 30)",
    ),
    (
        "entity_swap",
        "Who is the CEO of Apple?",
        "Who is the CEO of Google?",
        False,
        "single proper-noun swap; common false-positive on entity-bearing queries",
    ),
    (
        "date_drift",
        "What were Apple's earnings in Q1 2024?",
        "What were Apple's earnings in Q1 2025?",
        False,
        "time-bearing query; serving the wrong year is a real prod hazard",
    ),
    (
        "boilerplate_dilution",
        LONG_PREAMBLE + "\n\nVendor: ACME. Decision: APPROVE.",
        LONG_PREAMBLE + "\n\nVendor: ACME. Decision: REJECT.",
        False,
        "200-token shared preamble swamps the operative one-word flip",
    ),
    (
        "code_op_flip",
        "Refactor for clarity:\n\ndef f(x):\n    return x + 1",
        "Refactor for clarity:\n\ndef f(x):\n    return x - 1",
        False,
        "opposite arithmetic; near-identical surface form",
    ),

    # ─────────────────────────────────────────────────────────────────────
    # GRAY AREAS (should_match=None)  — document, do not assert
    # ─────────────────────────────────────────────────────────────────────
    (
        "crosslingual_paraphrase",
        "What's the capital of France?",
        "Quelle est la capitale de la France?",
        None,
        "answer is the same but the cache stores the EN response — policy call",
    ),

    # ─────────────────────────────────────────────────────────────────────
    # POSITIVE CONTROLS (should_match=True)  — asserted
    # ─────────────────────────────────────────────────────────────────────
    (
        "english_paraphrase",
        "What's the capital of France?",
        "Tell me the capital city of France.",
        True,
        "the case the semantic cache exists for; expected HIT",
    ),
]
