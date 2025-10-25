import asyncio
from typing import Callable, List
import json

from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client

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
        self.server_params = StdioServerParameters(**server_parameters)
        self.tools = None

    def get_tools(self) -> List[FunctionTool]:
        if not self.tools:
            mcp_tools = self._get_mcp_tools()
            self.tools = self._build_mcp_func(mcp_tools)

        return self.tools

    def _get_mcp_tools(self) -> List[mcp_types.Tool]:
        async def run():
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = await session.list_tools()
                    return tools.tools

        return asyncio.run(run())

    def _build_mcp_func(self, mcp_tools: List[mcp_types.Tool]) -> List[FunctionTool]:
        def func_maker(mcp_tool: mcp_types.Tool) -> FunctionTool:
            def wrapper(**kwwargs):
                async def run():
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.call_tool(mcp_tool.name, kwwargs)
                            return result

                return asyncio.run(run())

            return mcp_tool_as_function_tool(mcp_tool, wrapper)

        return [func_maker(name) for name in mcp_tools]


def mcp_tools(server_params) -> List[FunctionTool]:
    mcp_tools_instance = MCPTools(server_params)
    return mcp_tools_instance.get_tools()


def mcp_tool_from_config(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    # Support both direct format and Claude Desktop format
    if "mcpServers" in config:
        config = config["mcpServers"]

    return mcp_tools(config)
