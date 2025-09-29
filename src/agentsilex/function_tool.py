from dataclasses import dataclass
from typing import Callable, Any, Dict


@dataclass
class FunctionTool:
    name: str
    description: str
    function: Callable
    parameters_specification: Dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)
