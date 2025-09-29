from pydantic import BaseModel
from typing import Union, Any


class RunResult(BaseModel):
    output_type: Union[str, BaseModel] = ""  # NOT IMPLEMENTED YET
    final_output: Any
