"""
src/api/routes.py

FastAPI route handlers for the LLM Gateway.
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.api.schemas import (
    GatewayRequest, GatewayResponseSchema,
    StatsResponse, HealthResponse,
)
from src.gateway.gateway import LLMGateway
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Single gateway instance shared across requests
_gateway = LLMGateway()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        components={
            "api":      "up",
            "cache":    "up",
            "router":   "up",
        }
    )


@router.post("/chat", response_model=GatewayResponseSchema)
async def chat(request: GatewayRequest):
    """
    Standard (blocking) chat endpoint.

    Goes through the full gateway pipeline:
    rate limit -> cache -> route -> call LLM -> cache -> cost track
    """
    logger.info(f"Chat request", extra={
        "query":     request.query[:60],
        "client_id": request.client_id,
    })

    try:
        response = _gateway.process_request(
            query=request.query,
            client_id=request.client_id,
            system=request.system,
        )

        return GatewayResponseSchema(
            answer=response.answer,
            model_used=response.model_used,
            was_cache_hit=response.was_cache_hit,
            was_rate_limited=response.was_rate_limited,
            complexity_score=response.complexity_score,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            metadata=response.metadata,
        )

    except Exception as e:
        logger.error(f"Chat request failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: GatewayRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Note: caching and routing decisions happen BEFORE streaming
    starts (we need to know which model to call). The streaming
    only applies to the actual token generation from the LLM.

    If the answer is a cache hit, we simulate a fast "stream"
    by sending the full cached answer as a single chunk —
    there's no generation to stream since we already have it.
    """
    async def event_generator():
        # Routing + cache check happen synchronously first
        decision = _gateway.router.route(request.query)
        cached   = _gateway.cache.get(request.query, model=decision.model)

        if cached is not None:
            answer, similarity = cached
            # Cache hit — send the whole answer as one SSE event
            yield f"data: {json.dumps({'chunk': answer, 'done': False})}\n\n"
            yield f"data: {json.dumps({'done': True, 'cache_hit': True, 'similarity': similarity})}\n\n"
            return

        # Cache miss — stream from the LLM directly
        import anthropic
        import os

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

        full_text = ""
        with client.messages.stream(
            model=decision.model,
            max_tokens=1024,
            temperature=0.3,
            system=request.system,
            messages=[{"role": "user", "content": request.query}],
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                yield f"data: {json.dumps({'chunk': text, 'done': False})}\n\n"

        # Cache the complete answer after streaming finishes
        _gateway.cache.set(request.query, full_text, model=decision.model)

        yield f"data: {json.dumps({'done': True, 'cache_hit': False, 'model': decision.model})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Return cache and cost statistics."""
    stats = _gateway.get_stats()
    return StatsResponse(cache=stats["cache"], cost=stats["cost"])