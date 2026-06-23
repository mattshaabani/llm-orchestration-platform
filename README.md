# LLM Orchestration Platform

A production-grade LLM gateway with intelligent routing, semantic caching,
cost tracking, rate limiting, and retry logic — the infrastructure layer
that makes LLMs production-safe and cost-efficient.

---

## Overview

Every company deploying LLMs at scale hits the same problems: costs spiral,
latency is unpredictable, and a single API failure breaks the whole app.
This project builds the operational layer that solves all of these problems.

**Core insight:** not every question needs your most expensive model, and
many questions have already been answered before. Routing + caching together
can reduce LLM costs by 85-90% on realistic traffic patterns.

---

## Architecture

    Incoming Request
           |
    [1] Token Bucket Rate Limiter     ← reject abuse, protect budget
           |
    [2] Complexity Scorer             ← classify query difficulty (0-1)
           |
    [3] Model Router                  ← cheap model vs powerful model
           |
    [4] Semantic Cache (Redis)        ← return cached answer if similar enough
           |
    [5] LLM Client + Retry Logic      ← exponential backoff on failure
           |
    [6] Cache Store                   ← save for future similar queries
           |
    [7] Cost Tracker                  ← record spend and savings
           |
    Final Response + Full Metadata

---

## Tech Stack

| Component | Technology |
|---|---|
| Gateway API | FastAPI + Uvicorn |
| Semantic cache | Redis + sentence-transformers |
| Rate limiting | Token bucket algorithm on Redis |
| LLM backend | HuggingFace Inference API / Anthropic |
| Cost tracking | In-memory tracker with per-model pricing |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Compose |
| Environment | Conda + ipykernel |

---

## Project Structure

    llm-orchestration-platform/
    |-- src/
    |   |-- gateway/
    |   |   |-- gateway.py           # main orchestration — ties everything together
    |   |   |-- rate_limiter.py      # token bucket rate limiting via Redis
    |   |   └-- retry_handler.py     # exponential backoff retry logic
    |   |-- cache/
    |   |   └-- semantic_cache.py    # vector similarity cache on Redis
    |   |-- routing/
    |   |   |-- complexity_scorer.py # heuristic complexity scoring
    |   |   └-- router.py            # model selection based on score
    |   |-- cost/
    |   |   └-- cost_tracker.py      # per-request cost calculation and reporting
    |   |-- api/
    |   |   |-- main.py              # FastAPI app entry point
    |   |   |-- routes.py            # /chat, /chat/stream, /stats, /health
    |   |   └-- schemas.py           # Pydantic request/response models
    |   └-- utils/
    |       |-- config.py            # centralized config from YAML + .env
    |       └-- logger.py            # structured colored logging
    |-- notebooks/
    |   |-- 01_caching_demo.ipynb
    |   |-- 02_routing_analysis.ipynb
    |   |-- 03_cost_analysis.ipynb
    |   └-- 04_gateway_demo.ipynb
    |-- configs/
    |   |-- gateway_config.yaml      # routing, caching, rate limit, cost settings
    |   └-- db_config.yaml           # Redis and PostgreSQL connection settings
    |-- docker/
    |   └-- Dockerfile
    |-- monitoring/
    |   └-- prometheus.yml
    |-- docker-compose.yml
    |-- environment.yml
    └-- .env.example

---

## Quick Start

**1. Clone and set up environment**

    git clone https://github.com/MattShaabani/llm-orchestration-platform.git
    cd llm-orchestration-platform

    conda env create -f environment.yml
    conda activate llm-orchestration
    pip install -e .

**2. Configure environment variables**

    cp .env.example .env

Edit .env and add your HuggingFace token or Anthropic API key.

**3. Start Redis and monitoring stack**

    docker-compose up redis prometheus grafana -d

**4. Run the gateway API**

    uvicorn src.api.main:app --reload --port 8081

Open http://localhost:8081/docs for interactive API documentation.

**5. Run the full stack with Docker**

    docker-compose up --build

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /v1/health | System health check |
| POST | /v1/chat | Standard blocking chat |
| POST | /v1/chat/stream | Streaming chat (SSE) |
| GET | /v1/stats | Cache and cost statistics |
| GET | /metrics | Prometheus metrics |
| GET | /docs | Swagger UI |

---

## Key Algorithms

### Token Bucket Rate Limiting

Each client gets a bucket with capacity = burst_size tokens.
Tokens refill at requests_per_minute / 60 per second.

    On each request:
        elapsed  = now - last_refill
        tokens   = min(capacity, tokens + elapsed × refill_rate)
        if tokens >= 1: ALLOW and consume 1 token
        else:           REJECT

Advantage over naive counters: allows controlled bursts while
maintaining a smooth average rate over time.

### Exponential Backoff

    delay = min(base_delay × (exponential_base ^ attempt), max_delay)
    delay_with_jitter = delay × (1 + uniform(-0.1, 0.1))

With config (base=1s, exp=2, max=30s):
- Attempt 1 fails → wait ~1s
- Attempt 2 fails → wait ~2s
- Attempt 3 fails → wait ~4s

Jitter prevents thundering herd when many clients retry simultaneously.

### Semantic Similarity Caching

    cache_hit = cosine_similarity(embed(query), embed(cached_query)) >= threshold

Threshold = 0.92 — calibrated to catch paraphrases (0.92-0.99)
without false positives (unrelated questions typically score < 0.80).

### Complexity Scoring

    score = 0.50 × keyword_score + 0.25 × length_score + 0.25 × structure_score

Keyword score is weighted most heavily because keyword presence
is the strongest predictor of reasoning depth required.

---

## Cost Analysis Results

Simulated on 1000 requests with 30% complex / 70% simple traffic
and 35% cache hit rate:

| Scenario | Cost | Savings |
|---|---|---|
| No routing, no caching | $0.1367 | baseline |
| Routing only | ~$0.040 | ~71% |
| Routing + caching (35%) | ~$0.019 | ~86% |

At production scale (1M requests/month), this represents thousands
of dollars in monthly savings from infrastructure that costs
essentially nothing to operate.

---

## Routing Decisions

The complexity scorer correctly routes:

| Query type | Score | Model |
|---|---|---|
| "What is Python?" | 0.20 | Simple (cheap) |
| "Define machine learning." | 0.18 | Simple (cheap) |
| "Compare transformer vs LSTM architectures" | 0.73 | Complex (powerful) |
| "Design a distributed system for 1M events/sec" | 0.64 | Complex (powerful) |

---

## Semantic Cache Performance

Tested with threshold = 0.92:

| Query | Type | Similarity | Result |
|---|---|---|---|
| "What is RAG?" (exact) | exact | 1.0000 | HIT |
| "Can you explain what RAG is?" | paraphrase | 0.9456 | HIT |
| "Tell me about RAG" | rewording | 0.9234 | HIT |
| "What is machine learning?" | related topic | 0.7523 | MISS |
| "What is the weather today?" | unrelated | 0.1823 | MISS |

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| HF_TOKEN | HuggingFace API token | yes |
| ANTHROPIC_API_KEY | Anthropic Claude API key | optional |
| APP_ENV | development or production | no |
| LOG_LEVEL | INFO, DEBUG, WARNING | no |

---

## License

MIT