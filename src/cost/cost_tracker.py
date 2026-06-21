"""
src/cost/cost_tracker.py

Tracks the cost of every LLM request and computes savings
from caching and routing.

Storage: PostgreSQL — durable, queryable, supports analytics.

Usage:
    from src.cost.cost_tracker import CostTracker
    tracker = CostTracker()
    tracker.record_request(
        model="claude-haiku-4-5-20251001",
        input_tokens=150,
        output_tokens=80,
        was_cache_hit=False,
    )
    report = tracker.get_summary()
"""

import time
from dataclasses import dataclass
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RequestCost:
    """Cost breakdown for a single LLM request."""
    model:          str
    input_tokens:   int
    output_tokens:  int
    input_cost:     float
    output_cost:    float
    total_cost:     float
    was_cache_hit:  bool
    timestamp:      float


class CostCalculator:
    """
    Pure cost math — no storage, just calculation.
    Kept separate from CostTracker so it's easily unit-testable.
    """

    def __init__(self):
        self.pricing = {
            settings.routing.simple_model: {
                "input":  settings.cost.simple_model_input_cost,
                "output": settings.cost.simple_model_output_cost,
            },
            settings.routing.complex_model: {
                "input":  settings.cost.complex_model_input_cost,
                "output": settings.cost.complex_model_output_cost,
            },
        }

    def calculate(
        self,
        model:         str,
        input_tokens:  int,
        output_tokens: int,
    ) -> tuple[float, float, float]:
        """
        Compute (input_cost, output_cost, total_cost) in USD.

        Formula:
            cost = (tokens / 1_000_000) × price_per_million
        """
        prices = self.pricing.get(model)

        if prices is None:
            logger.warning(f"No pricing found for model, using complex model pricing", extra={
                "model": model
            })
            prices = self.pricing[settings.routing.complex_model]

        input_cost  = (input_tokens  / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]
        total_cost  = input_cost + output_cost

        return round(input_cost, 6), round(output_cost, 6), round(total_cost, 6)


class CostTracker:
    """
    Records every request's cost and computes aggregate savings
    from caching and routing.

    In-memory storage by default (list of RequestCost). For
    production you'd swap this to PostgreSQL — we'll add that
    in the database layer, but the in-memory version lets us
    test and reason about the math first.
    """

    def __init__(self):
        self.calculator = CostCalculator()
        self.requests: list[RequestCost] = []

    def record_request(
        self,
        model:          str,
        input_tokens:   int,
        output_tokens:  int,
        was_cache_hit:  bool = False,
    ) -> RequestCost:
        """
        Record a single request and its cost.

        Note: if was_cache_hit=True, the ACTUAL cost incurred is $0
        (we didn't call the LLM) — but we still compute what it
        WOULD have cost, stored as total_cost, so we can later sum
        up "money saved by cache" separately from "money spent".
        """
        input_cost, output_cost, total_cost = self.calculator.calculate(
            model, input_tokens, output_tokens
        )

        record = RequestCost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            was_cache_hit=was_cache_hit,
            timestamp=time.time(),
        )

        self.requests.append(record)

        logger.debug(f"Recorded request cost", extra={
            "model":      model,
            "total_cost": total_cost,
            "cache_hit":  was_cache_hit,
        })

        return record

    def get_summary(self) -> dict:
        """
        Compute aggregate cost statistics.

        Key metrics:
            actual_spend       — money actually spent (cache misses only)
            saved_by_cache      — money NOT spent because of cache hits
            total_requests      — count of all requests
            cache_hit_rate      — % of requests served from cache
            savings_percentage  — % reduction in total cost due to cache
        """
        if not self.requests:
            return {
                "total_requests":      0,
                "actual_spend":         0.0,
                "saved_by_cache":       0.0,
                "cache_hit_rate":       0.0,
                "savings_percentage":   0.0,
            }

        cache_hits  = [r for r in self.requests if r.was_cache_hit]
        cache_misses = [r for r in self.requests if not r.was_cache_hit]

        actual_spend   = sum(r.total_cost for r in cache_misses)
        saved_by_cache = sum(r.total_cost for r in cache_hits)

        # What would total cost have been with NO caching at all?
        hypothetical_total_cost = actual_spend + saved_by_cache

        savings_percentage = (
            (saved_by_cache / hypothetical_total_cost * 100)
            if hypothetical_total_cost > 0 else 0.0
        )

        cache_hit_rate = len(cache_hits) / len(self.requests) * 100

        # Breakdown by model
        by_model = {}
        for r in self.requests:
            if r.model not in by_model:
                by_model[r.model] = {"count": 0, "cost": 0.0}
            by_model[r.model]["count"] += 1
            if not r.was_cache_hit:
                by_model[r.model]["cost"] += r.total_cost

        return {
            "total_requests":      len(self.requests),
            "cache_hits":           len(cache_hits),
            "cache_misses":         len(cache_misses),
            "cache_hit_rate":       round(cache_hit_rate, 2),
            "actual_spend":         round(actual_spend, 6),
            "saved_by_cache":       round(saved_by_cache, 6),
            "hypothetical_no_cache_cost": round(hypothetical_total_cost, 6),
            "savings_percentage":   round(savings_percentage, 2),
            "by_model":             by_model,
        }

    def print_report(self) -> None:
        """Print a formatted cost report to terminal."""
        summary = self.get_summary()

        print("\n" + "="*55)
        print("COST TRACKING REPORT")
        print("="*55)
        print(f"  Total requests:        {summary['total_requests']}")
        print(f"  Cache hits:             {summary.get('cache_hits', 0)}")
        print(f"  Cache hit rate:         {summary['cache_hit_rate']}%")
        print("-"*55)
        print(f"  Actual spend:           ${summary['actual_spend']:.6f}")
        print(f"  Saved by cache:         ${summary['saved_by_cache']:.6f}")
        print(f"  Hypothetical (no cache):${summary.get('hypothetical_no_cache_cost', 0):.6f}")
        print(f"  Savings:                {summary['savings_percentage']}%")
        print("-"*55)
        print(f"  Breakdown by model:")
        for model, data in summary.get("by_model", {}).items():
            print(f"    {model}: {data['count']} requests, ${data['cost']:.6f} spent")
        print("="*55)