from collections import defaultdict
from typing import List
from agentsilex.evaluation.data_types import (
    EvalSet,
    EvalSetResult,
    EvalCaseResult,
    MetricSummary,
    DialogTurn,
    ToolUsage,
)
from agentsilex.evaluation.metric_evaluators.base_metric_evaluator import (
    BaseMetricEvaluator,
)
from agentsilex.agent import Agent
from agentsilex.runner import Runner
from agentsilex.session import Session


class AgentEvaluator:
    def __init__(self, agent: Agent, evaluators: List[BaseMetricEvaluator]):
        self.agent = agent
        self.evaluators = evaluators

    def evaluate(self, eval_set: EvalSet) -> EvalSetResult:
        eval_set_result = EvalSetResult(
            passed=True, total_cases=0, passed_cases=0, case_results=[]
        )
        for eval_case in eval_set.cases:
            session = Session()  # every case use new session
            runner = Runner(session)
            running_result = None
            actual_turns = []
            case_results = {}  # evaluutor instance -> MetricResult
            for expected_turn in eval_case.dialog_session:
                agent = self.agent if not running_result else running_result.last_agent
                running_result = runner.run(agent, expected_turn.user_utterance)

                actual_turns.append(
                    DialogTurn(
                        user_utterance=expected_turn.user_utterance,
                        agent_response=running_result.final_output,
                        tools_used=[
                            ToolUsage(name=name, arguments=arguments, response=response)
                            for name, arguments, response in running_result.function_calls
                        ],
                    )
                )

            for metric in self.evaluators:
                metric_result = metric.evaluate(actual_turns, eval_case.dialog_session)
                case_results[metric.name] = metric_result

            eval_set_result.total_cases += 1
            case_passed = all(result.passed for result in case_results.values())
            eval_set_result.passed_cases += 1 if case_passed else 0
            eval_set_result.passed = eval_set_result.passed and case_passed
            eval_set_result.case_results.append(
                EvalCaseResult(
                    eval_id=eval_case.eval_id,
                    passed=case_passed,
                    metric_results=case_results,
                )
            )

        # Compute metric summaries
        eval_set_result.metric_summaries = self._compute_metric_summaries(
            eval_set_result.case_results
        )

        return eval_set_result

    def _compute_metric_summaries(
        self, case_results: List[EvalCaseResult]
    ) -> dict[str, MetricSummary]:
        """Compute summary statistics for each metric across all cases."""
        if not case_results:
            return {}

        # Collect scores and pass status for each metric
        metric_scores: dict[str, list[float]] = defaultdict(list)
        metric_passed: dict[str, list[bool]] = defaultdict(list)

        for case_result in case_results:
            for metric_name, metric_result in case_result.metric_results.items():
                metric_scores[metric_name].append(metric_result.score)
                metric_passed[metric_name].append(metric_result.passed)

        # Compute summaries
        summaries = {}
        for metric_name in metric_scores:
            scores = metric_scores[metric_name]
            passed = metric_passed[metric_name]
            passed_count = sum(1 for p in passed if p)
            total_count = len(passed)

            summaries[metric_name] = MetricSummary(
                avg_score=sum(scores) / len(scores),
                min_score=min(scores),
                max_score=max(scores),
                passed_count=passed_count,
                total_count=total_count,
                pass_rate=passed_count / total_count if total_count > 0 else 0.0,
            )

        return summaries
