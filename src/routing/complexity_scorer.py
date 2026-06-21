"""
src/routing/complexity_scorer.py

Heuristic complexity scoring for incoming queries.
Determines whether a query is "simple" (route to cheap/fast model)
or "complex" (route to powerful model) WITHOUT calling an LLM.

This must be fast and free — it runs before every single request.

Usage:
    from src.routing.complexity_scorer import ComplexityScorer
    scorer = ComplexityScorer()
    score  = scorer.score("What is the capital of France?")
    # score ≈ 0.1 (simple)
"""

from email.mime import text
import re
from dataclasses import dataclass
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Keyword signals
# ─────────────────────────────────────────────

# Keywords that suggest the question needs deeper reasoning
COMPLEX_KEYWORDS = [
    "compare", "analyze", "design", "architect", "explain why",
    "evaluate", "optimize", "trade-off", "tradeoff", "implement",
    "debug", "refactor", "prove", "derive", "synthesize",
    "step by step", "pros and cons", "what if", "strategy",
]

# Keywords that suggest a simple factual lookup
SIMPLE_KEYWORDS = [
    "what is", "who is", "when did", "where is", "define",
    "spell", "translate", "what time", "how many",
]


@dataclass
class ComplexityResult:
    """Breakdown of the complexity score for transparency/debugging."""
    score:           float
    length_score:    float
    keyword_score:   float
    structure_score: float
    matched_complex_keywords: list[str]
    matched_simple_keywords:  list[str]


class ComplexityScorer:
    """
    Scores query complexity on a 0-1 scale using fast heuristics.

    Weighted combination:
        complexity = 0.3×length + 0.4×keywords + 0.3×structure

    These weights were chosen because keyword signals are the
    strongest indicator of reasoning depth required, while length
    alone is a weaker signal (a long simple question still exists).
    """

    WEIGHT_LENGTH    = 0.25
    WEIGHT_KEYWORDS  = 0.50
    WEIGHT_STRUCTURE = 0.25

    # Calibration constants
    LENGTH_SATURATION_WORDS = 60   # word count at which length_score = 1.0

    def _length_score(self, text: str) -> float:
        """
        Longer queries tend to carry more complexity, but with
        diminishing returns. We saturate at LENGTH_SATURATION_WORDS.

        score = min(word_count / saturation, 1.0)
        """
        word_count = len(text.split())
        return min(word_count / self.LENGTH_SATURATION_WORDS, 1.0)

    def _keyword_score(self, text: str) -> tuple[float, list[str], list[str]]:
        text_lower = text.lower()

        matched_complex = [kw for kw in COMPLEX_KEYWORDS if kw in text_lower]
        matched_simple  = [kw for kw in SIMPLE_KEYWORDS if kw in text_lower]

        score = 0.5
        score += 0.25 * len(matched_complex)   # increased from 0.15
        score -= 0.15 * len(matched_simple)
        score = max(0.0, min(1.0, score))

        return score, matched_complex, matched_simple

    def _structure_score(self, text: str) -> float:
        """
        Structural complexity: multiple clauses, conjunctions,
        multi-part questions tend to need more reasoning.

        Signals:
            - number of commas
            - "and"/"or" conjunctions
            - multiple question marks (multi-part question)
            - code blocks or technical syntax (suggests code reasoning)
        """
        comma_count      = text.count(",")
        conjunction_count = len(re.findall(r'\b(and|or)\b', text.lower()))
        question_count    = text.count("?")
        has_code           = bool(re.search(r'[{}\[\]();]|def |class |function ', text))

        raw_score = (
            comma_count * 0.1 +
            conjunction_count * 0.15 +
            max(0, question_count - 1) * 0.2 +
            (0.3 if has_code else 0)
        )

        return min(raw_score, 1.0)

    def score(self, query: str) -> float:
        """
        Compute the overall complexity score for a query.

        Returns:
            Float between 0 (very simple) and 1 (very complex).
        """
        result = self.score_detailed(query)
        return result.score

    def score_detailed(self, query: str) -> ComplexityResult:
        """
        Compute complexity score with full breakdown — useful for
        debugging routing decisions and for the evaluation notebook.
        """
        length_score = self._length_score(query)
        keyword_score, matched_complex, matched_simple = self._keyword_score(query)
        structure_score = self._structure_score(query)

        total = (
            self.WEIGHT_LENGTH * length_score +
            self.WEIGHT_KEYWORDS * keyword_score +
            self.WEIGHT_STRUCTURE * structure_score
        )
        total = round(min(1.0, max(0.0, total)), 4)

        logger.debug(f"Complexity scored", extra={
            "query": query[:50],
            "score": total,
        })

        return ComplexityResult(
            score=total,
            length_score=round(length_score, 4),
            keyword_score=round(keyword_score, 4),
            structure_score=round(structure_score, 4),
            matched_complex_keywords=matched_complex,
            matched_simple_keywords=matched_simple,
        )