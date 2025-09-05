from agentsilex.commands.register import command
from agentsilex.commands.context import get_context


@command
def navigate(url: str):
    """
    navigate to specified URL.
    """
    ctx = get_context()
    ctx.page.goto(url)

    return f"Navigated to {url}"
