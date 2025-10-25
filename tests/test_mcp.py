"""
Simple unit tests for MCP module.
Tests basic functionality with mocked MCP server interactions.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from mcp import types as mcp_types

from agentsilex.mcp import mcp_tool_as_function_tool, MCPTools, mcp_tools


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
                "required": ["param1"]
            }
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

    @patch('agentsilex.mcp.stdio_client')
    @patch('agentsilex.mcp.ClientSession')
    def test_get_tools_from_mcp_server(self, mock_session_class, mock_stdio):
        """Test retrieving tools from MCP server."""
        # Mock the MCP server response
        mock_tool = mcp_types.Tool(
            name="example_tool",
            description="Example tool from server",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                }
            }
        )

        # Setup mocks
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=MagicMock(tools=[mock_tool])
        )
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)

        # Create MCPTools instance
        server_params = {
            "command": "test",
            "args": []
        }
        mcp_tools_instance = MCPTools(server_params)

        # Get tools
        tools = mcp_tools_instance.get_tools()

        # Verify results
        assert len(tools) == 1
        assert tools[0].name == "example_tool"
        assert tools[0].description == "Example tool from server"

    @patch('agentsilex.mcp.stdio_client')
    @patch('agentsilex.mcp.ClientSession')
    def test_tools_cached_after_first_call(self, mock_session_class, mock_stdio):
        """Test that tools are cached and not re-fetched on subsequent calls."""
        # Setup mocks with a non-empty tool
        mock_tool = mcp_types.Tool(
            name="cached_tool",
            description="Tool for cache test",
            inputSchema={"type": "object", "properties": {}}
        )

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=MagicMock(tools=[mock_tool])
        )
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
