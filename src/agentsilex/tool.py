from typing import Callable
from agentsilex.function_tool import FunctionTool
from agentsilex.extract_function_schema import (
    function_schema as extract_function_schema,
)


def tool(func: Callable) -> FunctionTool:
    schema = extract_function_schema(func)

    return FunctionTool(
        name=schema.name,
        description=schema.description or "",
        function=func,
        parameters_specification=schema.params_json_schema,
    )
