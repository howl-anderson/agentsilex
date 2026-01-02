from typing import Any, List, Tuple

from pydantic import BaseModel, ConfigDict

from agentsilex.agent import Agent


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    final_output: Any
    function_calls: List[Tuple[str, str, Any]]
    last_agent: Agent
