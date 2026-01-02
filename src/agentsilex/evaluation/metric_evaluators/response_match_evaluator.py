from enum import Enum
from typing import List

from agentsilex.evaluation.data_types import (
    MetricResult,
    MetricTurnResult,
    DialogTurn,
)
from agentsilex.evaluation.metric_evaluators.base_metric_evaluator import (
    BaseMetricEvaluator,
)


class SimilarityMetric(Enum):
    """The type of similarity metric to use."""

    ROUGE_1 = "rouge1"
    """ROUGE-1 (unigram) F-measure."""

    ROUGE_2 = "rouge2"
    """ROUGE-2 (bigram) F-measure."""

    ROUGE_L = "rougeL"
    """ROUGE-L (longest common subsequence) F-measure."""

    EXACT = "exact"
    """Exact string match (case-insensitive, stripped)."""


class ResponseMatchEvaluator(BaseMetricEvaluator):
    """
    Evaluates if agent's final response matches expected response using text similarity.

    This evaluator compares the agent's response against a golden/expected response
    using various similarity metrics (ROUGE or exact match).

    Value range for this metric is [0, 1], with values closer to 1 more desirable.
    """

    name = "response_match"
    description = (
        "Evaluates if agent's response matches expected response using text similarity. "
        "Value range is [0, 1], with values closer to 1 more desirable."
    )

    def __init__(
        self,
        threshold: float = 0.7,
        metric: SimilarityMetric = SimilarityMetric.ROUGE_1,
        use_stemmer: bool = True,
    ):
        """
        Initialize the ResponseMatchEvaluator.

        Args:
            threshold: The threshold for pass/fail determination. Default is 0.7.
            metric: The similarity metric to use. Default is ROUGE_1.
            use_stemmer: Whether to use stemming for ROUGE metrics. Default is True.
        """
        super().__init__(threshold)
        self.metric = metric
        self.use_stemmer = use_stemmer
        self._scorer = None

    def _get_scorer(self):
        """Lazy initialization of rouge scorer."""
        if self._scorer is None and self.metric != SimilarityMetric.EXACT:
            try:
                from rouge_score import rouge_scorer

                self._scorer = rouge_scorer.RougeScorer(
                    [self.metric.value], use_stemmer=self.use_stemmer
                )
            except ImportError:
                raise ImportError(
                    "rouge_score package is required for ROUGE metrics. "
                    "Install it with: pip install rouge-score"
                )
        return self._scorer

    def evaluate(
        self, actual: List[DialogTurn], expected: List[DialogTurn]
    ) -> MetricResult:
        """Evaluate response similarity for each turn."""
        turn_results = []

        for actual_turn, expected_turn in zip(actual, expected):
            score = self._calculate_similarity(
                actual_turn.agent_response, expected_turn.agent_response
            )
            turn_results.append(
                MetricTurnResult(score=score, passed=score >= self.threshold)
            )

        if not turn_results:
            return MetricResult(score=0.0, passed=False, turn_results=[])

        overall_score = sum(r.score for r in turn_results) / len(turn_results)
        return MetricResult(
            score=overall_score,
            passed=overall_score >= self.threshold,
            turn_results=turn_results,
        )

    def _calculate_similarity(self, actual: str, expected: str) -> float:
        """Calculate similarity between actual and expected response."""
        if self.metric == SimilarityMetric.EXACT:
            return self._exact_match(actual, expected)
        else:
            return self._rouge_score(actual, expected)

    def _exact_match(self, actual: str, expected: str) -> float:
        """Exact string match (case-insensitive, whitespace-normalized)."""
        actual_normalized = " ".join(actual.lower().split())
        expected_normalized = " ".join(expected.lower().split())
        return 1.0 if actual_normalized == expected_normalized else 0.0

    def _rouge_score(self, actual: str, expected: str) -> float:
        """Calculate ROUGE score between actual and expected."""
        scorer = self._get_scorer()

        # Handle empty strings
        if not actual.strip() or not expected.strip():
            return 0.0 if actual.strip() != expected.strip() else 1.0

        scores = scorer.score(expected, actual)
        return scores[self.metric.value].fmeasure
