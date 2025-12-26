from pydantic import BaseModel
from typing import Union, Any


class RunResult(BaseModel):
    final_output: Any
