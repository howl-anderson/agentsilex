from typing import Any


class Event:
    def __init__(self, type: str, data: Any = None):
        self.type = type
        self.data = data or {}

    def __repr__(self):
        return f"Event(type={self.type}, data={self.data})"


class RawChunkEvent(Event):
    def __init__(self, chunk: Any):
        super().__init__(type="raw_chunk", data={"chunk": chunk})


class PartialOutputEvent(Event):
    def __init__(self, content: str):
        super().__init__(type="partial_output", data={"content": content})


class AgentHandoffEvent(Event):
    def __init__(self, agent_name: str):
        super().__init__(type="agent_handoff", data={"agent_name": agent_name})


class ToolCallEvent(Event):
    def __init__(self, func_request: Any):
        super().__init__(
            type="tool_call",
            data={"func_request": func_request},
        )


class ToolResponseEvent(Event):
    def __init__(self, func_response: Any):
        super().__init__(
            type="tool_response",
            data={"func_response": func_response},
        )


class FinalResultEvent(Event):
    def __init__(self, final_output: Any):
        super().__init__(
            type="final_result",
            data={"final_output": final_output},
        )
