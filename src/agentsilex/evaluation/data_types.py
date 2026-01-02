from typing import List, Dict, Any
from pydantic import BaseModel
from pydantic_core import from_json


class ToolUsage(BaseModel):
    name: str
    arguments: str
    response: Any


class DialogTurn(BaseModel):
    user_utterance: str
    agent_response: str
    tools_used: List[ToolUsage]


class EvalCase(BaseModel):
    eval_id: str
    dialog_session: List[DialogTurn]


class EvalSet(BaseModel):
    cases: List[EvalCase]

    @classmethod
    def load(cls, path) -> "EvalSet":
        with open(path, "r") as fd:
            json_data = fd.read()
            return cls.model_validate(from_json(json_data, allow_partial=False))

    def dump(self, path: str):
        with open(path, "w") as fd:
            fd.write(self.model_dump_json())


class MetricTurnResult(BaseModel):
    score: float
    passed: bool


class MetricResult(BaseModel):
    score: float
    passed: bool
    turn_results: List[MetricTurnResult]


class EvalCaseResult(BaseModel):
    eval_id: str
    passed: bool
    metric_results: Dict[str, MetricResult]


class MetricSummary(BaseModel):
    """Summary statistics for a single metric across all cases in an EvalSet."""

    avg_score: float
    min_score: float
    max_score: float
    passed_count: int
    total_count: int
    pass_rate: float


class EvalSetResult(BaseModel):
    passed: bool
    total_cases: int
    passed_cases: int
    case_results: List["EvalCaseResult"]
    metric_summaries: Dict[str, MetricSummary] = {}
