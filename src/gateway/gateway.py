"""
src/gateway/gateway.py

The main LLM Gateway — orchestrates routing, caching, cost tracking,
rate limiting and retries into a single entry point.

This is the "front door" every request goes through.

Usage:
    from src.gateway.gateway import LLMGateway
    gateway  = LLMGateway()
    response = gateway.process_request(
        query="What is RAG?",
        client_id="user_123",
    )
"""

import time
from dataclasses import dataclass, field
from src.routing.router import ModelRouter
from src.cache.semantic_cache import SemanticCache
from src.cost.cost_tracker import CostTracker
from src.gateway.rate_limiter import TokenBucketRateLimiter
from src.gateway.retry_handler import RetryHandler, RetryExhaustedError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GatewayResponse:
    """The complete response from a gateway request, with full transparency."""
    answer:          str
    model_used:      str
    was_cache_hit:   bool
    was_rate_limited: bool
    complexity_score: float
    latency_ms:      float
    cost_usd:        float
    metadata:        dict = field(default_factory=dict)


class LLMGateway:
    """
    The orchestration layer that every request flows through.

    Request flow:
        1. Rate limit check       — reject if client over quota
        2. Cache lookup           — return cached answer if similar enough
        3. Route decision         — pick simple or complex model
        4. Call LLM with retry    — exponential backoff on failure
        5. Cache the new answer
        6. Record cost
        7. Return full response with transparency metadata
    """

    def __init__(self):
        self.router        = ModelRouter()
        self.cache          = SemanticCache()
        self.cost_tracker   = CostTracker()
        self.rate_limiter   = TokenBucketRateLimiter()
        self.retry_handler  = RetryHandler()

        logger.info(f"LLM Gateway initialized")

    def process_request(
        self,
        query:     str,
        client_id: str = "default",
        system:    str = "You are a helpful assistant.",
    ) -> GatewayResponse:
        """
        Process a single request through the full gateway pipeline.
        """
        start_time = time.time()

        # ── Step 1: Rate limiting ──
        allowed, rate_info = self.rate_limiter.allow_request(client_id)

        if not allowed:
            return GatewayResponse(
                answer="Rate limit exceeded. Please try again shortly.",
                model_used="none",
                was_cache_hit=False,
                was_rate_limited=True,
                complexity_score=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0.0,
                metadata={"rate_limit_info": rate_info},
            )

        # ── Step 2: Routing decision (needed for cache lookup too) ──
        decision = self.router.route(query)

        # ── Step 3: Cache lookup ──
        cached = self.cache.get(query, model=decision.model)

        if cached is not None:
            answer, similarity = cached

            # Record as a cache hit — zero actual cost, but we log
            # what it WOULD have cost for savings reporting
            estimated_tokens = len(query.split()) + len(answer.split())
            self.cost_tracker.record_request(
                model=decision.model,
                input_tokens=len(query.split()) * 2,   # rough estimate
                output_tokens=len(answer.split()) * 2,
                was_cache_hit=True,
            )

            return GatewayResponse(
                answer=answer,
                model_used=decision.model,
                was_cache_hit=True,
                was_rate_limited=False,
                complexity_score=decision.complexity.score,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0.0,
                metadata={"cache_similarity": round(similarity, 4)},
            )

        # ── Step 4: Call LLM with retry ──
        try:
            response = self.retry_handler.execute_with_retry(
                self._call_llm,
                query, system, decision.model,
            )
        except RetryExhaustedError as e:
            logger.error(f"Gateway request failed after retries", extra={"error": str(e)})
            return GatewayResponse(
                answer="Service temporarily unavailable. Please try again.",
                model_used=decision.model,
                was_cache_hit=False,
                was_rate_limited=False,
                complexity_score=decision.complexity.score,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0.0,
                metadata={"error": str(e)},
            )

        # ── Step 5: Cache the new answer ──
        self.cache.set(query, response.content, model=decision.model)

        # ── Step 6: Record actual cost ──
        cost_record = self.cost_tracker.record_request(
            model=decision.model,
            input_tokens=response.prompt_tokens,
            output_tokens=response.output_tokens,
            was_cache_hit=False,
        )

        return GatewayResponse(
            answer=response.content,
            model_used=decision.model,
            was_cache_hit=False,
            was_rate_limited=False,
            complexity_score=decision.complexity.score,
            latency_ms=(time.time() - start_time) * 1000,
            cost_usd=cost_record.total_cost,
            metadata={
                "routing_reason": decision.reason,
                "prompt_tokens":  response.prompt_tokens,
                "output_tokens":  response.output_tokens,
            },
        )

    def _call_llm(self, query: str, system: str, model: str):
        """
        Internal LLM call — wrapped by retry_handler.
        Uses HuggingFace Inference API with the routed model.
        """
        import requests
        from src.utils.config import settings as global_settings

        headers = {"Authorization": f"Bearer {global_settings.env.hf_token}"}
        api_url = f"https://api-inference.huggingface.co/models/{model}"

        prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{query} [/INST]"

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.3,
                "max_new_tokens": 512,
                "return_full_text": False,
            }
        }

        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            content = data[0].get("generated_text", "")
        else:
            content = data.get("generated_text", "")

        class SimpleResponse:
            pass

        result = SimpleResponse()
        result.content = content
        result.prompt_tokens = len(prompt.split())
        result.output_tokens = len(content.split())
        return result

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.3,
            system=system,
            messages=[{"role": "user", "content": query}],
        )

        class SimpleResponse:
            content        = response.content[0].text
            prompt_tokens  = response.usage.input_tokens
            output_tokens  = response.usage.output_tokens

        return SimpleResponse()

    def get_stats(self) -> dict:
        """Return combined statistics from cache and cost tracker."""
        return {
            "cache":  self.cache.get_stats(),
            "cost":   self.cost_tracker.get_summary(),
        }