"""
src/api/main.py

FastAPI application entry point for the LLM Gateway.

Run locally:
    uvicorn src.api.main:app --reload --port 8081

Docs:
    http://localhost:8081/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.utils.logger import get_logger
from src.utils.config import settings

logger = get_logger(__name__)

app = FastAPI(
    title="LLM Orchestration Gateway",
    description="Production-grade LLM gateway with routing, caching, cost tracking, and rate limiting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router, prefix="/v1")


@app.on_event("startup")
async def startup_event():
    logger.info(f"LLM Gateway starting up", extra={
        "environment": settings.env.app_env,
    })