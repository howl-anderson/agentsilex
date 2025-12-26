from typing import Callable
from agentsilex.function_tool import FunctionTool
from agentsilex.extract_function_schema import extract_function_schema


def tool(func: Callable) -> FunctionTool:
    name, description, params_json_schema = extract_function_schema(func)

    return FunctionTool(
        name=name,
        description=description or "",
        function=func,
        parameters_specification=params_json_schema,
    )


def generate_tool(tool_name, tool_description):
    def wrapper(func: Callable):
        _, _, params_json_schema = extract_function_schema(func)

        return FunctionTool(
            name=tool_name,
            description=tool_description,
            function=func,
            parameters_specification=params_json_schema,
        )

    return wrapper
