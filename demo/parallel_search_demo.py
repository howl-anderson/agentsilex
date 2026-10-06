"""Search and fetch through Parallel's anonymous Streamable HTTP MCP endpoint."""

import argparse
import uuid

from agentsilex import Agent, Runner, Session
from agentsilex.mcp import mcp_tools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="What is the Model Context Protocol?")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    args = parser.parse_args()

    tools = mcp_tools({"url": "https://search.parallel.ai/mcp"})
    agent = Agent(
        name="Web Research Assistant",
        model=args.model,
        instructions=(
            "Use web_search to find sources and web_fetch to read relevant pages. "
            "Answer the user's question with links to your sources. "
            f"For all search and fetch calls, use session_id={uuid.uuid4()} "
            f"and model_name={args.model}."
        ),
        tools=tools,
    )
    result = Runner(Session()).run(agent, args.query)
    print(result.final_output)


if __name__ == "__main__":
    main()
