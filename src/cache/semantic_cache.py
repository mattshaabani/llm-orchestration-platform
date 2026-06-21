"""
src/cache/semantic_cache.py

Semantic caching layer using vector similarity.
Reduces LLM API costs by reusing answers to semantically
similar questions instead of calling the LLM every time.

Storage: Redis (fast in-memory key-value store)
Similarity: cosine similarity between question embeddings

Usage:
    from src.cache.semantic_cache import SemanticCache
    cache = SemanticCache()
    cached = cache.get("What is RAG?")
    if cached is None:
        answer = call_llm(...)
        cache.set("What is RAG?", answer)
"""

import json
import time
import hashlib
import numpy as np
from dataclasses import dataclass
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A single cached question-answer pair with its embedding."""
    question:  str
    answer:    str
    embedding: list[float]
    model:     str
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    """
    Semantic cache backed by Redis.

    Architecture:
        - Embeddings stored in Redis as JSON (question, answer, vector)
        - On lookup: embed query, compare against all cached vectors
        - On hit: return cached answer, increment hit counter
        - On miss: caller computes new answer, we store it

    For simplicity we keep all embeddings in a Python list loaded
    from Redis rather than using Redis's vector search module
    (which would require RediSearch — an extra dependency).
    This works fine up to a few thousand cached entries.
    """

    CACHE_KEY_PREFIX = "llm_cache:"

    def __init__(self):
        self.similarity_threshold = settings.cache.similarity_threshold
        self.ttl_seconds          = settings.cache.ttl_seconds
        self.enabled              = settings.cache.enabled
        self._embedder            = None
        self._redis               = None

        logger.info(f"Initialized SemanticCache", extra={
            "threshold": self.similarity_threshold,
            "enabled":   self.enabled,
        })

    @property
    def embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(settings.cache.embedding_model)
        return self._embedder

    @property
    def redis(self):
        """Lazy load Redis connection."""
        if self._redis is None:
            import redis
            self._redis = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.cache_db,
                decode_responses=True,
            )
        return self._redis

    def _make_key(self, question: str, model: str) -> str:
        """
        Generate a unique Redis key for a cache entry.
        Hash of question+model so each (question, model) pair
        gets its own storage slot.
        """
        raw = f"{question}::{model}"
        hash_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"{self.CACHE_KEY_PREFIX}{hash_id}"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Standard cosine similarity — same math as Project 1."""
        dot   = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def _get_all_entries(self, model: str) -> list[CacheEntry]:
        """
        Fetch all cached entries for a given model from Redis.
        Used to compare the incoming query against everything cached.
        """
        keys = self.redis.keys(f"{self.CACHE_KEY_PREFIX}*")
        entries = []

        for key in keys:
            raw = self.redis.get(key)
            if raw is None:
                continue
            try:
                data = json.loads(raw)
                if data.get("model") != model:
                    continue
                entries.append(CacheEntry(**data))
            except (json.JSONDecodeError, TypeError):
                continue

        return entries

    def get(self, question: str, model: str) -> tuple[str, float] | None:
        """
        Look up a semantically similar cached answer.

        Returns:
            (cached_answer, similarity_score) if a hit above threshold found
            None if no sufficiently similar entry exists

        Algorithm:
            1. Embed the incoming question
            2. Fetch all cached entries for this model
            3. Compute cosine similarity against each
            4. If max similarity > threshold, return that entry's answer
        """
        if not self.enabled:
            return None

        query_vector = self.embedder.encode(question, normalize_embeddings=True)
        entries      = self._get_all_entries(model)

        if not entries:
            logger.debug(f"Cache empty for model", extra={"model": model})
            return None

        best_entry = None
        best_score = -1.0

        for entry in entries:
            entry_vector = np.array(entry.embedding)
            score = self._cosine_similarity(query_vector, entry_vector)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.similarity_threshold:
            logger.info(f"Cache HIT", extra={
                "question":        question[:50],
                "matched_question": best_entry.question[:50],
                "similarity":       round(best_score, 4),
            })
            self._increment_hit_count(best_entry, model)
            return best_entry.answer, best_score

        logger.debug(f"Cache MISS", extra={
            "question":    question[:50],
            "best_score":  round(best_score, 4),
            "threshold":   self.similarity_threshold,
        })
        return None

    def set(self, question: str, answer: str, model: str) -> None:
        """
        Store a new question-answer pair in the cache.

        Embeds the question and saves it to Redis with a TTL
        so stale entries automatically expire.
        """
        if not self.enabled:
            return

        vector = self.embedder.encode(question, normalize_embeddings=True)

        entry = CacheEntry(
            question=question,
            answer=answer,
            embedding=vector.tolist(),
            model=model,
            timestamp=time.time(),
            hit_count=0,
        )

        key = self._make_key(question, model)
        self.redis.setex(
            key,
            self.ttl_seconds,
            json.dumps(entry.__dict__),
        )

        logger.debug(f"Cached new entry", extra={
            "question": question[:50],
            "model":    model,
        })

    def _increment_hit_count(self, entry: CacheEntry, model: str) -> None:
        """Track how many times each cache entry has been reused."""
        key = self._make_key(entry.question, model)
        entry.hit_count += 1
        self.redis.setex(
            key,
            self.ttl_seconds,
            json.dumps(entry.__dict__),
        )

    def get_stats(self) -> dict:
        """Return cache statistics — size, total hits, hit distribution."""
        keys = self.redis.keys(f"{self.CACHE_KEY_PREFIX}*")
        total_hits = 0
        entries_data = []

        for key in keys:
            raw = self.redis.get(key)
            if raw:
                data = json.loads(raw)
                total_hits += data.get("hit_count", 0)
                entries_data.append(data)

        return {
            "total_entries": len(keys),
            "total_hits":    total_hits,
            "entries":       entries_data,
        }

    def clear(self) -> int:
        """Clear all cache entries. Returns count of entries removed."""
        keys = self.redis.keys(f"{self.CACHE_KEY_PREFIX}*")
        if keys:
            self.redis.delete(*keys)
        logger.info(f"Cache cleared", extra={"count": len(keys)})
        return len(keys)