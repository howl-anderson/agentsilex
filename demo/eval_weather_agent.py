"""
End-to-end evaluation demo for the weather agent.

This script demonstrates how to:
1. Define an agent with tools
2. Load an evaluation set from a JSON file
3. Run evaluation using different metric evaluators
4. Inspect the results
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


# Define the weather tool (same as simple_agent.py)
@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return "SUNNY"


# Create the agent
agent = Agent(
    name="weather_agent",
    model="gemini/gemini-2.0-flash",
    instructions="You are a helpful weather assistant. Use the get_weather tool to find weather information. Always respond with a clear statement about the weather.",
    tools=[get_weather],
)


def main():
    # Load the evaluation set
    eval_file = Path(__file__).parent / "eval_data" / "weather_agent_eval.json"
    eval_set = EvalSet.load(str(eval_file))

    print(f"Loaded {len(eval_set.cases)} evaluation cases")
    print("=" * 60)

    # Create evaluators
    tool_evaluator = ToolTrajectoryEvaluator(threshold=1.0)

    response_evaluator = ResponseMatchEvaluator(
        threshold=0.5,
        metric=SimilarityMetric.ROUGE_1,
    )

    llm_judge = LLMJudgeEvaluator(
        threshold=0.5,
        model="gemini/gemini-2.0-flash",
        num_samples=1,
    )

    # Create the agent evaluator with all metrics
    evaluator = AgentEvaluator(
        agent=agent,
        evaluators=[tool_evaluator, response_evaluator, llm_judge],
    )

    # Run evaluation
    print("Running evaluation...")
    print("=" * 60)

    result = evaluator.evaluate(eval_set)

    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Passed: {result.passed}")
    print(f"Total Cases: {result.total_cases}")
    print(f"Passed Cases: {result.passed_cases}")
    print(f"Pass Rate: {result.passed_cases / result.total_cases * 100:.1f}%")
    print()

    # Print metric summaries
    print("Metric Summaries:")
    for metric_name, summary in result.metric_summaries.items():
        print(
            f"  {metric_name}: "
            f"avg={summary.avg_score:.3f}, "
            f"pass_rate={summary.pass_rate:.1%} ({summary.passed_count}/{summary.total_count})"
        )
    print()

    # Print detailed results for each case
    for case_result in result.case_results:
        print(f"Case: {case_result.eval_id}")
        print(f"  Passed: {case_result.passed}")
        for metric_name, metric_result in case_result.metric_results.items():
            print(f"  {metric_name}:")
            print(f"    Score: {metric_result.score:.3f}")
            print(f"    Passed: {metric_result.passed}")
            for i, turn_result in enumerate(metric_result.turn_results):
                print(
                    f"    Turn {i + 1}: score={turn_result.score:.3f}, passed={turn_result.passed}"
                )
        print()


if __name__ == "__main__":
    main()
