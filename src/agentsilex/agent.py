from typing import Any, List, Tuple, Dict
from agentsilex.function_tool import FunctionTool
import json
import re
import inspect

HANDOFF_TOOL_PREFIX = "transfer_to_"


def has_context_param(func):
    sig = inspect.signature(func)
    return "context" in sig.parameters


def as_valid_tool_name(name: str, prefix: str | None = None) -> str:
    # Replace invalid chars with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_.:+-]", "_", name)

    # Ensure starts with letter or underscore
    if sanitized and not re.match(r"^[a-zA-Z_]", sanitized):
        sanitized = "_" + sanitized

    if prefix:
        sanitized = prefix + sanitized

    # Truncate to 64 chars
    return sanitized[:64] if sanitized else "_unnamed"


class ToolsSet:
    def __init__(self, tools: List[FunctionTool]):
        self.tools = tools
        self.registry = {tool.name: tool for tool in tools}

    def get_specification(self):
        spec = []
        for tool in self.tools:
            spec.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_specification,
                    },
                }
            )

        return spec

    def execute_function_call(self, context: dict, call_spec):
        tool = self.registry.get(call_spec.function.name)

        if not tool:
            raise ValueError(f"Tool {call_spec.function.name} not found")

        args = json.loads(call_spec.function.arguments)

        # inject context if the tool supports it
        if has_context_param(tool.function):
            args["context"] = context

        result = tool(**args)

        return {"role": "tool", "tool_call_id": call_spec.id, "content": str(result)}


class Handoff:
    def __init__(self, agent: "Agent"):
        self.agent = agent

    @property
    def name(self):
        return as_valid_tool_name(self.agent.name, prefix="transfer_to_")

    @property
    def description(self):
        return f"Handoff to the {self.agent.name} agent to handle the request. {self.agent.instructions}"

    @property
    def parameters_specification(self):
        return {}


class AgentHandoffs:
    def __init__(self, handoffs: List["Agent"]):
        self.handoffs: List[Handoff] = [Handoff(agent) for agent in handoffs]
        self.registry: Dict[str, Handoff] = {
            handoff.name: handoff for handoff in self.handoffs
        }

    def get_specification(self):
        spec = []
        for handoff in self.handoffs:
            spec.append(
                {
                    "type": "function",
                    "function": {
                        "name": handoff.name,
                        "description": handoff.description,
                        "parameters": handoff.parameters_specification,
                    },
                }
            )

        return spec

    def handoff_agent(self, call_spec) -> Tuple["Agent", Dict]:
        handoff = self.registry[call_spec.function.name]

        handoff_reponse = {
            "role": "tool",
            "tool_call_id": call_spec.id,
            "content": json.dumps({"assistant": handoff.agent.name}),
        }
        return handoff.agent, handoff_reponse


class Agent:
    def __init__(
        self,
        name: str,
        model: Any,
        instructions: str,
        tools: List[FunctionTool] | None = None,
        handoffs: List["Agent"] | None = None,
    ):
        self.name = name
        self.model = model
        self.instructions = instructions
        self.tools = tools or []
        self.tools_set = ToolsSet(self.tools)
        self.handoffs = AgentHandoffs(handoffs or [])

    def get_system_prompt(self):
        return {"role": "system", "content": self.instructions}

    def as_tool(self, tool_name: str, tool_description: str) -> FunctionTool:
        from agentsilex.agent_as_tool import agent_as_tool

        return agent_as_tool(self, tool_name, tool_description)
