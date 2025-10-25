from agentsilex import Agent, Runner, Session
from agentsilex.mcp import mcp_tools
from agentsilex.utils import print_dialog_history

server_config = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-everything"],
}

tools = mcp_tools(server_config)

print(f"Loaded {len(tools)} tools from MCP server:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")
print()

agent = Agent(
    name="MCP Calculator Agent",
    model="gemini/gemini-2.0-flash-exp",
    instructions="You are a helpful assistant. Use the provided tools to help the user.",
    tools=tools,  # just like normal tools
)

session = Session()
result = Runner(session).run(agent, "Please add 123 and 456, then tell me the result.")

print("Final output:", result.final_output)

print_dialog_history(session)
