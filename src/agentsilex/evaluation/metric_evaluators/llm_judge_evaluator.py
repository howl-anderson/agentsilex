from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from litellm import completion

from agentsilex.evaluation.data_types import (
    MetricResult,
    MetricTurnResult,
    DialogTurn,
)
from agentsilex.evaluation.metric_evaluators.base_metric_evaluator import (
    BaseMetricEvaluator,
)


class JudgeVerdict(BaseModel):
    """Structured output for LLM judge verdict."""

    reasoning: str = Field(description="Step-by-step reasoning for the verdict")
    verdict: Literal["yes", "no"] = Field(
        description="'yes' if the response is valid, 'no' otherwise"
    )


DEFAULT_PROMPT_TEMPLATE = """You are an expert evaluator for AI agent responses.

Compare the actual response from the agent with the expected response.

User Query: {user_query}

Expected Response: {expected_response}

Actual Response: {actual_response}

Evaluate whether the actual response is valid and satisfies the user's intent compared to the expected response.
Be flexible about exact wording - focus on semantic equivalence and correctness."""


class LLMJudgeEvaluator(BaseMetricEvaluator):
    """
    Evaluates agent responses using an LLM as a judge with structured output.

    This evaluator uses another LLM to judge whether the agent's response
    is valid compared to an expected response. It uses structured output
    (Pydantic model) for reliable parsing and supports multiple sampling
    for more reliable results.

    Value range for this metric is [0, 1], with values closer to 1 more desirable.
    """

    name = "llm_judge"
    description = (
        "Uses an LLM to judge if agent's response matches expected response. "
        "Value range is [0, 1], with values closer to 1 more desirable."
    )

    def __init__(
        self,
        threshold: float = 0.5,
        model: str = "gpt-4o-mini",
        prompt_template: Optional[str] = None,
        num_samples: int = 3,
        temperature: float = 0.0,
    ):
        """
        Initialize the LLMJudgeEvaluator.

        Args:
            threshold: The threshold for pass/fail determination. Default is 0.5.
            model: The LLM model to use for judging. Default is "gpt-4o-mini".
            prompt_template: Custom prompt template. Must contain {user_query},
                {expected_response}, and {actual_response} placeholders.
            num_samples: Number of times to sample the LLM for each evaluation.
                Results are aggregated by majority vote. Default is 3.
            temperature: Temperature for LLM sampling. Default is 0.0.
        """
        super().__init__(threshold)
        self.model = model
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.num_samples = num_samples
        self.temperature = temperature

    def evaluate(
        self, actual: List[DialogTurn], expected: List[DialogTurn]
    ) -> MetricResult:
        """Evaluate responses using LLM as judge for each turn."""
        turn_results = []

        for actual_turn, expected_turn in zip(actual, expected):
            score = self._evaluate_turn(actual_turn, expected_turn)
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

    def _evaluate_turn(
        self, actual_turn: DialogTurn, expected_turn: DialogTurn
    ) -> float:
        """Evaluate a single turn using LLM judge with multiple samples."""
        prompt = self.prompt_template.format(
            user_query=actual_turn.user_utterance,
            expected_response=expected_turn.agent_response,
            actual_response=actual_turn.agent_response,
        )

        verdicts = []
        for _ in range(self.num_samples):
            verdict = self._call_llm(prompt)
            if verdict is not None:
                verdicts.append(verdict)

        if not verdicts:
            # If all samples failed, return 0
            return 0.0

        # Majority vote
        yes_count = sum(1 for v in verdicts if v)
        return 1.0 if yes_count > len(verdicts) / 2 else 0.0

    def _call_llm(self, prompt: str) -> Optional[bool]:
        """Call the LLM with structured output and return the verdict."""
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format=JudgeVerdict,
            )
            response_text = response.choices[0].message.content
            verdict = JudgeVerdict.model_validate_json(response_text)
            return verdict.verdict == "yes"
        except Exception:
            return None


class RubricBasedEvaluator(LLMJudgeEvaluator):
    """
    Evaluates agent responses against a set of rubrics using an LLM.

    Each rubric is a criterion that the response should satisfy.
    The score is the fraction of rubrics that are satisfied.
    Uses structured output for reliable parsing.
    """

    name = "rubric_based"
    description = (
        "Evaluates agent's response against a set of rubrics using LLM as judge. "
        "Score is the fraction of rubrics satisfied."
    )

    RUBRIC_PROMPT_TEMPLATE = """You are an expert evaluator for AI agent responses.

User Query: {user_query}

Agent's Response: {actual_response}

Evaluate the response against the following rubric:
{rubric}

Determine if the rubric criterion is satisfied by the agent's response."""

    def __init__(
        self,
        rubrics: List[str],
        threshold: float = 0.8,
        model: str = "gpt-4o-mini",
        num_samples: int = 3,
        temperature: float = 0.0,
    ):
        """
        Initialize the RubricBasedEvaluator.

        Args:
            rubrics: List of rubric strings to evaluate against.
            threshold: The threshold for pass/fail determination. Default is 0.8.
            model: The LLM model to use for judging. Default is "gpt-4o-mini".
            num_samples: Number of samples per rubric evaluation. Default is 3.
            temperature: Temperature for LLM sampling. Default is 0.0.
        """
        super().__init__(
            threshold=threshold,
            model=model,
            num_samples=num_samples,
            temperature=temperature,
        )
        self.rubrics = rubrics

    def _evaluate_turn(
        self, actual_turn: DialogTurn, expected_turn: DialogTurn
    ) -> float:
        """Evaluate a single turn against all rubrics."""
        if not self.rubrics:
            return 1.0

        rubric_scores = []
        for rubric in self.rubrics:
            prompt = self.RUBRIC_PROMPT_TEMPLATE.format(
                user_query=actual_turn.user_utterance,
                actual_response=actual_turn.agent_response,
                rubric=rubric,
            )

            verdicts = []
            for _ in range(self.num_samples):
                verdict = self._call_llm(prompt)
                if verdict is not None:
                    verdicts.append(verdict)

            if verdicts:
                # Majority vote for this rubric
                yes_count = sum(1 for v in verdicts if v)
                rubric_scores.append(1.0 if yes_count > len(verdicts) / 2 else 0.0)
            else:
                rubric_scores.append(0.0)

        # Overall score is the fraction of rubrics satisfied
        return sum(rubric_scores) / len(rubric_scores)
