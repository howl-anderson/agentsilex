"""
Evaluation demo with expected failures.

This demonstrates that the evaluators correctly detect mismatches between
actual agent behavior and expected behavior.
"""

from pathlib import Path

from agentsilex import Agent, tool
from agentsilex.evaluation import (
    AgentEvaluator,
    EvalSet,
    ToolTrajectoryEvaluator,
    ResponseMatchEvaluator,
    LLMJudgeEvaluator,
    SimilarityMetric,
)


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "SUNNY"


agent = Agent(
    name="weather_agent",
    model="gemini/gemini-2.0-flash",
    instructions="You are a helpful weather assistant. Use the get_weather tool to find weather information. Always respond with a clear statement about the weather.",
    tools=[get_weather],
)


def main():
    eval_file = (
        Path(__file__).parent / "eval_data" / "weather_agent_eval_with_failures.json"
    )
    eval_set = EvalSet.load(str(eval_file))

    print(f"Loaded {len(eval_set.cases)} evaluation cases (with expected failures)")
    print("=" * 60)

    tool_evaluator = ToolTrajectoryEvaluator(
        threshold=1.0,
    )

    response_evaluator = ResponseMatchEvaluator(
        threshold=0.6,
        metric=SimilarityMetric.ROUGE_1,
    )

    llm_judge = LLMJudgeEvaluator(
        threshold=0.5,
        model="gemini/gemini-2.0-flash",
        num_samples=1,
    )

    evaluator = AgentEvaluator(
        agent=agent,
        evaluators=[tool_evaluator, response_evaluator, llm_judge],
    )

    print("Running evaluation...")
    print("=" * 60)

    result = evaluator.evaluate(eval_set)

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Passed: {result.passed}")
    print(f"Pass Rate: {result.passed_cases}/{result.total_cases}")
    print()

    # Print metric summaries
    print("Metric Summaries:")
    for metric_name, summary in result.metric_summaries.items():
        print(
            f"  {metric_name}: "
            f"avg={summary.avg_score:.3f}, "
            f"min={summary.min_score:.3f}, "
            f"max={summary.max_score:.3f}, "
            f"pass_rate={summary.pass_rate:.1%} ({summary.passed_count}/{summary.total_count})"
        )
    print()

    print("Case Details:")
    for case_result in result.case_results:
        status = "PASS" if case_result.passed else "FAIL"
        print(f"[{status}] Case: {case_result.eval_id}")
        for metric_name, metric_result in case_result.metric_results.items():
            metric_status = "✓" if metric_result.passed else "✗"
            print(f"    {metric_status} {metric_name}: score={metric_result.score:.3f}")
        print()


if __name__ == "__main__":
    main()
