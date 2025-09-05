from langchain_core.tools import tool

TOOLS_REGISTRY = []


def command(func):
    func = tool(func)

    TOOLS_REGISTRY.append(func)

    return func
