from agentsilex import Runner, Session
from agentsilex.function_tool import FunctionTool
from agentsilex.tool import generate_tool


def agent_as_tool(agent, tool_name: str, tool_description: str) -> FunctionTool:

    @generate_tool(tool_name, tool_description)
    def agent_tool(input: str, context) -> str:
        # create an in-memory session, which will be temprary for this function call
        session = Session()

        # Run the agent with a passing query
        result = Runner(session, context).run(agent, input)

        return result.final_output

    return agent_tool
