"""
Broker — the decision cascade that routes each LLM call.

The broker is the brain of ThriftAI. For each completion() call, it decides:

    1. REPLAY CHECK: Is this a replay run and is this agent NOT in the live list?
       → Return exact output from trace. Done.

    2. EXACT CACHE CHECK: Is there an exact-match cache hit for
       (agent_name, prompt_template_hash, content_hash)?
       → Return cached response. Done.

    3. SEMANTIC CACHE CHECK (opt-in): Is there an embedding-similar
       cached response for (agent_name, prompt_template_hash)?
       → Return cached response. Done.

    4. LIVE CALL: Route to the LLM provider.
       → Record response in exact cache (and semantic cache if enabled).
       → Track cost, including the embedding API cost when semantic
         cache is active.

    5. DOWNSTREAM INVALIDATION (replay only): Compare live agent's output
       against what's in the trace. If different, mark all dependents
       as invalidated (they can't use replay stubs anymore).

Design notes:
- The broker does NOT know about specific providers. It delegates
  live calls to `thriftai.providers.call_litellm`.
- Semantic cache is opt-in; when disabled the cascade shortcuts to
  exact cache → live, with no embedding overhead.
- The embedding for a query is computed at most once per call: the
  broker caches it across the semantic get() and the subsequent put().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from thriftai.cache import compute_content_hash, compute_prompt_hash
from thriftai.cost import estimate_cost
from thriftai.providers import call_litellm
from thriftai.trace import Trace

log = logging.getLogger(__name__)


class CallResolution(Enum):
    """How a completion call was resolved."""
    REPLAY = "replay"             # served from trace
    CACHE_HIT = "cache_hit"       # served from exact cache
    SEMANTIC_HIT = "semantic_hit"  # served from semantic (embedding) cache
    LIVE = "live"                 # sent to LLM provider


@dataclass
class BrokerResult:
    """Result of a brokered completion call."""
    resolution: CallResolution
    response_text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached_cost_usd: float = 0.0  # what it WOULD have cost if live
    embedding_cost_usd: float = 0.0  # spend on semantic-cache embedding calls
    similarity_score: float | None = None  # set on SEMANTIC_HIT


class Broker:
    """Routes LLM calls through the replay → exact → semantic → live cascade."""

    def __init__(self, cache, trace_store, cost_tracker, semantic_cache=None):
        self.cache = cache
        self.trace_store = trace_store
        self.cost_tracker = cost_tracker
        self.semantic_cache = semantic_cache

    def route(
        self,
        messages: list[dict],
        model: str,
        agent_name: str | None,
        replay_trace: Trace | None = None,
        live_agents: list[str] | None = None,
        invalidated_agents: set[str] | None = None,
        **kwargs: Any,
    ) -> BrokerResult:
        """Route a single LLM call through the decision cascade."""
        live_agents = live_agents or []
        invalidated_agents = invalidated_agents or set()

        prompt_hash = compute_prompt_hash(messages)
        content_hash = compute_content_hash(messages)

        # 1. REPLAY CHECK
        if (
            replay_trace is not None
            and agent_name is not None
            and agent_name not in live_agents
            and agent_name not in invalidated_agents
        ):
            entry = self.trace_store.get_agent_output(replay_trace, agent_name)
            if entry is not None:
                log.debug("broker: REPLAY agent=%s", agent_name)
                would_have = estimate_cost(
                    entry.model, entry.input_tokens, entry.output_tokens
                )
                return BrokerResult(
                    resolution=CallResolution.REPLAY,
                    response_text=entry.response_text,
                    model=entry.model,
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    cost_usd=0.0,
                    cached_cost_usd=would_have,
                )

        # 2. EXACT CACHE CHECK
        if agent_name is not None:
            hit = self.cache.get(agent_name, prompt_hash, content_hash)
            if hit is not None:
                log.debug("broker: CACHE_HIT agent=%s", agent_name)
                would_have = estimate_cost(
                    hit["model"], hit["input_tokens"], hit["output_tokens"]
                )
                return BrokerResult(
                    resolution=CallResolution.CACHE_HIT,
                    response_text=hit["response_text"],
                    model=hit["model"],
                    input_tokens=hit["input_tokens"],
                    output_tokens=hit["output_tokens"],
                    cost_usd=0.0,
                    cached_cost_usd=would_have,
                )

        # 3. SEMANTIC CACHE CHECK (opt-in)
        query_embedding = None
        if self.semantic_cache is not None and agent_name is not None:
            query_embedding = self.semantic_cache.embed(messages)
            sem_hit = self.semantic_cache.get(
                agent_name,
                prompt_hash,
                messages,
                precomputed_embedding=query_embedding,
            )
            embed_cost = self.semantic_cache.take_pending_embed_cost()
            if sem_hit is not None:
                log.debug(
                    "broker: SEMANTIC_HIT agent=%s similarity=%.4f",
                    agent_name,
                    sem_hit["similarity_score"],
                )
                would_have = estimate_cost(
                    sem_hit["model"],
                    sem_hit["input_tokens"],
                    sem_hit["output_tokens"],
                )
                return BrokerResult(
                    resolution=CallResolution.SEMANTIC_HIT,
                    response_text=sem_hit["response_text"],
                    model=sem_hit["model"],
                    input_tokens=sem_hit["input_tokens"],
                    output_tokens=sem_hit["output_tokens"],
                    cost_usd=0.0,
                    cached_cost_usd=would_have,
                    embedding_cost_usd=embed_cost,
                    similarity_score=sem_hit["similarity_score"],
                )
            # Semantic miss: keep the embed cost for the live-call path to attribute.
            pending_embed_cost = embed_cost
        else:
            pending_embed_cost = 0.0

        # 4. LIVE CALL
        log.debug("broker: LIVE agent=%s model=%s", agent_name, model)
        completion = call_litellm(messages=messages, model=model, **kwargs)

        if agent_name is not None:
            self.cache.put(
                agent_name=agent_name,
                prompt_hash=prompt_hash,
                content_hash=content_hash,
                model=model,
                response_text=completion.response_text,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )
            if self.semantic_cache is not None:
                self.semantic_cache.put(
                    agent_name=agent_name,
                    prompt_hash=prompt_hash,
                    content_hash=content_hash,
                    messages=messages,
                    model=model,
                    response_text=completion.response_text,
                    input_tokens=completion.input_tokens,
                    output_tokens=completion.output_tokens,
                    precomputed_embedding=query_embedding,
                )
                # put() may have embedded (e.g. if query_embedding was None),
                # so drain any extra pending cost into the same bucket.
                pending_embed_cost += self.semantic_cache.take_pending_embed_cost()

        return BrokerResult(
            resolution=CallResolution.LIVE,
            response_text=completion.response_text,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=completion.cost_usd,
            cached_cost_usd=completion.cost_usd,
            embedding_cost_usd=pending_embed_cost,
        )
