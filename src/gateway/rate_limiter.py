"""
src/gateway/rate_limiter.py

Token bucket rate limiter backed by Redis.
Redis is used so the rate limit is shared correctly even if
multiple gateway instances are running (horizontal scaling).

Usage:
    from src.gateway.rate_limiter import TokenBucketRateLimiter
    limiter = TokenBucketRateLimiter()
    allowed = limiter.allow_request(client_id="user_123")
"""

import time
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TokenBucketRateLimiter:
    """
    Token bucket algorithm implemented atomically in Redis.

    Each client gets their own bucket, identified by client_id.
    Bucket state (current tokens, last refill time) is stored
    in Redis so it persists across requests and gateway restarts.

    Math:
        capacity    = burst_size           (max tokens in bucket)
        refill_rate = requests_per_minute / 60   (tokens added per second)

        On each request:
            elapsed_seconds = now - last_refill
            new_tokens      = min(capacity, current_tokens + elapsed_seconds × refill_rate)
            if new_tokens >= 1:
                consume 1 token, ALLOW
            else:
                REJECT
    """

    def __init__(self):
        self.capacity    = settings.rate_limit.burst_size
        self.refill_rate = settings.rate_limit.requests_per_minute / 60.0
        self._redis      = None

        logger.info(f"Initialized TokenBucketRateLimiter", extra={
            "capacity":    self.capacity,
            "refill_rate": round(self.refill_rate, 4),
        })

    @property
    def redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                db=settings.redis.db,
                decode_responses=True,
            )
        return self._redis

    def allow_request(self, client_id: str) -> tuple[bool, dict]:
        """
        Check if a request from this client should be allowed.

        Returns:
            (allowed: bool, info: dict with current bucket state)
        """
        key = f"rate_limit:{client_id}"
        now = time.time()

        bucket_data = self.redis.hgetall(key)

        if bucket_data:
            tokens       = float(bucket_data.get("tokens", self.capacity))
            last_refill  = float(bucket_data.get("last_refill", now))
        else:
            tokens      = float(self.capacity)
            last_refill = now

        # Refill based on elapsed time
        elapsed     = now - last_refill
        tokens      = min(self.capacity, tokens + elapsed * self.refill_rate)

        if tokens >= 1:
            tokens -= 1
            allowed = True
        else:
            allowed = False

        # Persist updated bucket state
        self.redis.hset(key, mapping={
            "tokens":      tokens,
            "last_refill": now,
        })
        self.redis.expire(key, 120)   # auto-cleanup idle buckets

        info = {
            "allowed":          allowed,
            "remaining_tokens": round(tokens, 2),
            "capacity":         self.capacity,
        }

        if not allowed:
            logger.warning(f"Rate limit exceeded", extra={
                "client_id": client_id,
            })

        return allowed, info