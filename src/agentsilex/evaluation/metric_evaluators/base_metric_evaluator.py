from typing import List

from agentsilex.evaluation.data_types import MetricResult, DialogTurn


class BaseMetricEvaluator:
    name: str = ""
    description: str = ""
    min_value: float = 0.0
    max_value: float = 1.0

    def __init__(self, threshold: float = 0.8):
        """
        Initialize the evaluator with a threshold.

        Args:
            threshold: The threshold for pass/fail determination. Default is 0.8.
        """
        self.threshold = threshold

    def evaluate(
        self, actual: List[DialogTurn], expected: List[DialogTurn]
    ) -> MetricResult:
        """
        Evaluate the actual dialog case (multiple turns) against the expected dialog case.

        Args:
            actual: The actual dialog turns produced by the agent.
            expected: The expected dialog turns (golden reference).

        Returns:
            MetricResult containing score, pass/fail status, and per-turn results.
        """
        raise NotImplementedError
