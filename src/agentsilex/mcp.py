import asyncio
import json
from collections.abc import Callable
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from agentsilex.function_tool import FunctionTool


def mcp_tool_as_function_tool(mcp_tool: mcp_types.Tool, func: Callable) -> FunctionTool:
    return FunctionTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        function=func,
        parameters_specification=mcp_tool.inputSchema,
    )


class MCPTools:
    def __init__(self, server_parameters: dict):
        self.http_params = None
        if "url" in server_parameters:
            if "command" in server_parameters:
                raise ValueError("Configure either a stdio command or an HTTP URL, not both")
            self.http_params = dict(server_parameters)
            headers = dict(self.http_params.get("headers") or {})
            if not any(key.lower() == "user-agent" for key in headers):
                headers["User-Agent"] = (
                    "AgentSilex (https://github.com/howl-anderson/agentsilex)"
                )
            self.http_params["headers"] = headers
            self.server_params = None
        else:
            self.server_params = StdioServerParameters(**server_parameters)
        self.tools = None

    @asynccontextmanager
    async def _connect(self):
        if self.http_params is not None:
            async with streamablehttp_client(**self.http_params) as (read, write, _):
                yield read, write
        else:
            async with stdio_client(self.server_params) as (read, write):
                yield read, write

    def get_tools(self) -> list[FunctionTool]:
        if not self.tools:
            mcp_tools = self._get_mcp_tools()
            self.tools = self._build_mcp_func(mcp_tools)

        return self.tools

    def _get_mcp_tools(self) -> list[mcp_types.Tool]:
        async def run():
            async with (
                self._connect() as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()

                tools = await session.list_tools()
                return tools.tools

        return asyncio.run(run())

    def _build_mcp_func(self, mcp_tools: list[mcp_types.Tool]) -> list[FunctionTool]:
        def func_maker(mcp_tool: mcp_types.Tool) -> FunctionTool:
            def wrapper(**kwargs):
                async def run():
                    async with (
                        self._connect() as (read, write),
                        ClientSession(read, write) as session,
                    ):
                        await session.initialize()
                        result = await session.call_tool(mcp_tool.name, kwargs)
                        return result

                return asyncio.run(run())

            return mcp_tool_as_function_tool(mcp_tool, wrapper)

        return [func_maker(name) for name in mcp_tools]


def mcp_tools(server_params) -> list[FunctionTool]:
    mcp_tools_instance = MCPTools(server_params)
    return mcp_tools_instance.get_tools()


def mcp_tool_from_config(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    # Support both direct format and Claude Desktop format
    if "mcpServers" in config:
        config = config["mcpServers"]

    return mcp_tools(config)
