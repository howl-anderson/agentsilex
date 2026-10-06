"""
Simple unit tests for MCP module.
Tests basic functionality with mocked MCP server interactions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types as mcp_types

from agentsilex.mcp import MCPTools, mcp_tool_as_function_tool, mcp_tools


class TestMCPToolConversion:
    """Test MCP tool to FunctionTool conversion."""

    def test_mcp_tool_as_function_tool(self):
        """Test converting a single MCP tool to FunctionTool."""
        # Create a mock MCP tool
        mcp_tool = mcp_types.Tool(
            name="test_tool",
            description="A test tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"}
                },
                "required": ["param1"],
            },
        )

        # Create a simple callable
        def test_func(**kwargs):
            return kwargs

        # Convert to FunctionTool
        function_tool = mcp_tool_as_function_tool(mcp_tool, test_func)

        # Verify properties
        assert function_tool.name == "test_tool"
        assert function_tool.description == "A test tool"
        assert function_tool.parameters_specification["type"] == "object"
        assert "param1" in function_tool.parameters_specification["properties"]


class TestMCPTools:
    """Test MCPTools class functionality."""

    @patch("agentsilex.mcp.stdio_client")
    @patch("agentsilex.mcp.ClientSession")
    def test_get_tools_from_mcp_server(self, mock_session_class, mock_stdio):
        """Test retrieving tools from MCP server."""
        # Mock the MCP server response
        mock_tool = mcp_types.Tool(
            name="example_tool",
            description="Example tool from server",
            inputSchema={"type": "object", "properties": {"text": {"type": "string"}}},
        )

        # Setup mocks
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_tool]))
        mock_session_class.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_stdio.return_value.__aenter__ = AsyncMock(
            return_value=(MagicMock(), MagicMock())
        )
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        # Create MCPTools instance
        server_params = {"command": "test", "args": []}
        mcp_tools_instance = MCPTools(server_params)

        # Get tools
        tools = mcp_tools_instance.get_tools()

        # Verify results
        assert len(tools) == 1
        assert tools[0].name == "example_tool"
        assert tools[0].description == "Example tool from server"

    @patch("agentsilex.mcp.stdio_client")
    @patch("agentsilex.mcp.ClientSession")
    def test_tools_cached_after_first_call(self, mock_session_class, mock_stdio):
        """Test that tools are cached and not re-fetched on subsequent calls."""
        # Setup mocks with a non-empty tool
        mock_tool = mcp_types.Tool(
            name="cached_tool",
            description="Tool for cache test",
            inputSchema={"type": "object", "properties": {}},
        )

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[mock_tool]))
        mock_session_class.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_stdio.return_value.__aenter__ = AsyncMock(
            return_value=(MagicMock(), MagicMock())
        )
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        # Create instance and call get_tools twice
        mcp_tools_instance = MCPTools({"command": "test", "args": []})
        tools1 = mcp_tools_instance.get_tools()
        tools2 = mcp_tools_instance.get_tools()

        # Should return the same cached list
        assert tools1 is tools2
        assert len(tools1) == 1
        # list_tools should only be called once
        assert mock_session.list_tools.call_count == 1


class TestHTTPMCPTools:
    @patch("agentsilex.mcp.streamablehttp_client")
    @patch("agentsilex.mcp.ClientSession")
    def test_discovery_and_function_call(self, mock_session_class, mock_http):
        remote_tool = mcp_types.Tool(
            name="web_search",
            description="Search the web",
            inputSchema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
        result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text="A useful source")]
        )
        session = AsyncMock()
        session.list_tools.return_value = MagicMock(tools=[remote_tool])
        session.call_tool.return_value = result
        mock_session_class.return_value.__aenter__.return_value = session
        mock_http.return_value.__aenter__ = AsyncMock(
            return_value=(MagicMock(), MagicMock(), MagicMock())
        )
        mock_http.return_value.__aexit__ = AsyncMock(return_value=None)

        config = {"url": "https://example.com/mcp", "timeout": 10, "sse_read_timeout": 40}
        tools = mcp_tools(config)
        assert tools[0](query="MCP") is result
        session.call_tool.assert_awaited_once_with("web_search", {"query": "MCP"})
        assert tools[0].parameters_specification == remote_tool.inputSchema
        assert mock_http.call_count == 2
        for call in mock_http.call_args_list:
            assert call.kwargs == {
                **config,
                "headers": {
                    "User-Agent": "AgentSilex (https://github.com/howl-anderson/agentsilex)"
                },
            }
        assert session.initialize.await_count == 2
        assert mock_http.return_value.__aexit__.await_count == 2
        assert "headers" not in config

    def test_custom_headers_are_preserved(self):
        headers = {"user-agent": "MyApplication", "Authorization": "Bearer test"}
        instance = MCPTools({"url": "https://example.com/mcp", "headers": headers})
        assert instance.http_params["headers"] == headers
        assert instance.http_params["headers"] is not headers

    def test_ambiguous_transport_rejected(self):
        with pytest.raises(ValueError, match="either"):
            MCPTools({"url": "https://example.com/mcp", "command": "test"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
