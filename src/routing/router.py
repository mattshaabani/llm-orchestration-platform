"""
src/routing/router.py

Routes incoming queries to the appropriate LLM based on complexity.

Usage:
    from src.routing.router import ModelRouter
    router = ModelRouter()
    decision = router.route("What is the capital of France?")
    print(decision.model)  # claude-haiku-4-5-20251001
"""

from dataclasses import dataclass
from src.routing.complexity_scorer import ComplexityScorer, ComplexityResult
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingDecision:
    """The result of a routing decision, with full transparency."""
    model:           str
    complexity:      ComplexityResult
    reason:          str


class ModelRouter:
    """
    Routes queries to simple or complex models based on
    heuristic complexity scoring.

    Why route at all instead of always using the best model?
        Cost. The complex model can cost 10-20x more per token
        than the simple model. If 60-70% of real-world traffic
        is simple factual questions, routing saves enormous
        amounts of money at scale with minimal quality loss.
    """

    def __init__(self):
        self.scorer    = ComplexityScorer()
        self.threshold = settings.routing.complexity_threshold
        self.simple_model  = settings.routing.simple_model
        self.complex_model = settings.routing.complex_model

        logger.info(f"Initialized ModelRouter", extra={
            "threshold":     self.threshold,
            "simple_model":  self.simple_model,
            "complex_model": self.complex_model,
        })

    def route(self, query: str) -> RoutingDecision:
        """
        Decide which model should handle this query.

        Returns:
            RoutingDecision with the chosen model and reasoning.
        """
        complexity = self.scorer.score_detailed(query)

        if complexity.score >= self.threshold:
            model  = self.complex_model
            reason = (
                f"complexity {complexity.score} >= threshold {self.threshold}"
            )
        else:
            model  = self.simple_model
            reason = (
                f"complexity {complexity.score} < threshold {self.threshold}"
            )

        decision = RoutingDecision(
            model=model,
            complexity=complexity,
            reason=reason,
        )

        logger.info(f"Routing decision", extra={
            "query":      query[:50],
            "model":      model,
            "complexity": complexity.score,
        })

        return decision