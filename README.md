# AgentSilex

A transparent, minimal, and hackable agent framework for developers who want full control.

## Why AgentSilex?

While large agent frameworks offer extensive features, they often become black boxes that are hard to understand, customize, or debug. AgentSilex takes a different approach:

- **Transparent**: Every line of code is readable and understandable. No magic, no hidden complexity.
- **Minimal**: Core implementation in ~300 lines. You can read the entire codebase in one sitting.
- **Hackable**: Designed for modification. Fork it, customize it, make it yours.
- **Educational**: Perfect for learning how agents actually work under the hood.

## Who is this for?

- **Companies** who need a customizable foundation for their agent systems
- **Developers** who want to understand agent internals, not just use them
- **Educators** teaching AI agent concepts
- **Researchers** prototyping new agent architectures

## Installation

```bash
pip install agentsilex
```

Or with uv:

```bash
uv add agentsilex
```

## Quick Start

```python
from agentsilex import Agent, Runner, Session, tool

# Define a simple tool
@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    # In production, this would call a real weather API
    return "SUNNY"

# Create an agent with the weather tool
agent = Agent(
    name="Weather Assistant",
    model="gemini/gemini-2.0-flash",  # or any LiteLLM supported model
    instructions="Help users find weather information using the available tools.",
    tools=[get_weather]
)

# Create a session to track conversation history
session = Session()

# Run the agent with a user query
runner = Runner(agent, session)
result = runner.run("What's the weather in Monte Cristo?")

# Output the result
print("Final output:", result.final_output)

# Access the conversation history
for message in session.get_dialogs():
    print(message)
```

