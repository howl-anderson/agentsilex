import json
from typing import List

from agentsilex.evaluation.data_types import (
    MetricResult,
    MetricTurnResult,
    DialogTurn,
    ToolUsage,
)
from agentsilex.evaluation.metric_evaluators.base_metric_evaluator import (
    BaseMetricEvaluator,
)


class ToolTrajectoryEvaluator(BaseMetricEvaluator):
    """
    Evaluates tool use trajectories for accuracy.

    This evaluator compares the sequence of tools called by the agent against
    expected calls. It requires an exact match of tool names and arguments.

    For each turn, if the tool calls match exactly, a score of 1.0 is awarded,
    otherwise 0.0. The overall score is the average across all turns.
    """

    name = "tool_trajectory"
    description = (
        "Compares tool call trajectories (expected vs actual). "
        "Score of 1.0 indicates a perfect match, 0.0 indicates a mismatch."
    )

    def __init__(self, threshold: float = 0.8):
        """
        Initialize the ToolTrajectoryEvaluator.

        Args:
            threshold: The threshold for pass/fail determination. Default is 0.8.
        """
        super().__init__(threshold)

    def evaluate(
        self, actual: List[DialogTurn], expected: List[DialogTurn]
    ) -> MetricResult:
        """Evaluate tool trajectories for each turn."""
        turn_results = []

        for actual_turn, expected_turn in zip(actual, expected):
            score = self._calculate_turn_score(
                actual_turn.tools_used, expected_turn.tools_used
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

    def _calculate_turn_score(
        self, actual_tools: List[ToolUsage], expected_tools: List[ToolUsage]
    ) -> float:
        """Calculate the score for a single turn's tool usage (exact match)."""
        if len(actual_tools) != len(expected_tools):
            return 0.0

        for actual, expected in zip(actual_tools, expected_tools):
            if not self._tool_matches(actual, expected):
                return 0.0

        return 1.0

    def _tool_matches(self, actual: ToolUsage, expected: ToolUsage) -> bool:
        """Check if two tool usages match exactly."""
        if actual.name != expected.name:
            return False

        actual_args = self._normalize_arguments(actual.arguments)
        expected_args = self._normalize_arguments(expected.arguments)
        return actual_args == expected_args

    def _normalize_arguments(self, arguments: str) -> dict:
        """Normalize arguments string to dict for comparison."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                return {"raw": arguments}
        return {}
