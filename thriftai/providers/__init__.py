"""
Provider abstraction via LiteLLM passthrough.

LiteLLM gives us multi-provider support (Anthropic, OpenAI, Google, etc.)
out of the box. We don't need separate provider adapters — LiteLLM IS the
adapter layer.

The provider module normalizes responses into a CompletionResult dataclass
that carries response text, token counts, and cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompletionResult:
    """Normalized result from any LLM provider."""
    response_text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw_response: Any = None


def call_litellm(messages: list[dict], model: str, **kwargs: Any) -> CompletionResult:
    """
    Make a completion call via LiteLLM and return a normalized result.

    This is the single live-call path. The broker calls this when
    a request falls through cache and replay.
    """
    import litellm

    response = litellm.completion(model=model, messages=messages, **kwargs)

    return CompletionResult(
        response_text=response.choices[0].message.content,
        model=model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cost_usd=litellm.completion_cost(completion_response=response),
        raw_response=response,
    )
