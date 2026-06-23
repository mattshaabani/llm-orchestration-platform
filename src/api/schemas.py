"""
src/api/schemas.py

Pydantic schemas for the LLM Gateway API.
"""

from pydantic import BaseModel, Field
from typing import Optional


class GatewayRequest(BaseModel):
    """Request body for POST /v1/chat"""
    query:     str = Field(..., description="The user's question or prompt")
    client_id: str = Field(default="default", description="Client identifier for rate limiting")
    system:    str = Field(
        default="You are a helpful assistant.",
        description="System prompt"
    )


class GatewayResponseSchema(BaseModel):
    """Response body for POST /v1/chat"""
    answer:           str
    model_used:       str
    was_cache_hit:    bool
    was_rate_limited: bool
    complexity_score: float
    latency_ms:       float
    cost_usd:         float
    metadata:         dict


class StatsResponse(BaseModel):
    """Response body for GET /v1/stats"""
    cache: dict
    cost:  dict


class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status:     str
    version:    str = "1.0.0"
    components: dict