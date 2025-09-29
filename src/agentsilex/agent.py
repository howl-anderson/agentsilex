from typing import Any, List
from agentsilex.function_tool import FunctionTool


class ToolsSet:
    def __init__(self, tools: List[FunctionTool]):
        self.tools = tools
        self.registry = {tool.name: tool for tool in tools}

    def get_specification(self):
        spec = []
        for tool in self.tools:
            spec.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_specification
                }
            })

        return spec

    def execute_function_call(self, call_spec):
        import json

        tool = self.registry.get(call_spec.function.name)

        if not tool:
            raise ValueError(f"Tool {call_spec.function.name} not found")

        args = json.loads(call_spec.function.arguments)

        result = tool(**args)

        return {
            "role": "tool",
            "tool_call_id": call_spec.id,
            "content": str(result)
        }


class Agent:
    def __init__(
        self,
        name: str,
        model: Any,
        instructions: str,
        tools: List[FunctionTool],
    ):
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools

        self.tools_set = ToolsSet(tools)
