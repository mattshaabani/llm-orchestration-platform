"""
src/gateway/retry_handler.py

Exponential backoff retry logic for LLM API calls.

Usage:
    from src.gateway.retry_handler import RetryHandler
    handler = RetryHandler()
    result  = handler.execute_with_retry(some_function, arg1, arg2)
"""

import time
import random
from typing import Callable, TypeVar
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have failed."""
    pass


class RetryHandler:
    """
    Executes a function with exponential backoff retry on failure.

    Backoff formula:
        delay = min(base_delay × (exponential_base ^ attempt), max_delay)
        delay_with_jitter = delay × (1 + uniform(-0.1, 0.1))

    Jitter prevents synchronized retry storms when many clients
    fail at the same time (e.g. during an LLM provider outage).
    """

    def __init__(self):
        self.max_retries       = settings.retry.max_retries
        self.base_delay        = settings.retry.base_delay_seconds
        self.max_delay         = settings.retry.max_delay_seconds
        self.exponential_base  = settings.retry.exponential_base

    def _compute_delay(self, attempt: int) -> float:
        """
        Compute the delay before the next retry attempt.

        attempt=0 → first retry delay
        attempt=1 → second retry delay
        etc.
        """
        raw_delay = self.base_delay * (self.exponential_base ** attempt)
        capped_delay = min(raw_delay, self.max_delay)

        jitter_factor = 1 + random.uniform(-0.1, 0.1)
        return capped_delay * jitter_factor

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args,
        retryable_exceptions: tuple = (Exception,),
        **kwargs,
    ) -> T:
        """
        Call func(*args, **kwargs), retrying on failure with
        exponential backoff.

        Args:
            func:                  The function to call.
            retryable_exceptions:  Exception types that should trigger a retry.
                                   Other exceptions propagate immediately.

        Raises:
            RetryExhaustedError: if all attempts fail.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Retry succeeded", extra={"attempt": attempt})
                return result

            except retryable_exceptions as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(f"Attempt failed, retrying", extra={
                        "attempt":     attempt + 1,
                        "max_retries": self.max_retries,
                        "delay_sec":   round(delay, 2),
                        "error":       str(e),
                    })
                    time.sleep(delay)
                else:
                    logger.error(f"All retry attempts exhausted", extra={
                        "attempts": self.max_retries + 1,
                        "error":    str(e),
                    })

        raise RetryExhaustedError(
            f"Failed after {self.max_retries + 1} attempts: {last_exception}"
        )