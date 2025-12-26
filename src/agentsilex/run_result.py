from pydantic import BaseModel
from typing import Any


class RunResult(BaseModel):
    final_output: Any
