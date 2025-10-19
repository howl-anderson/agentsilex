# AgentSilex

A transparent, minimal, and hackable agent framework for developers who want full control.

## Why AgentSilex?

While large agent frameworks offer extensive features, they often become black boxes that are hard to understand, customize, or debug. AgentSilex takes a different approach:

- **Transparent**: Every line of code is readable and understandable. No magic, no hidden complexity.
- **Minimal**: Core implementation in ~300 lines. You can read the entire codebase in one sitting.
- **Hackable**: Designed for modification. Fork it, customize it, make it yours.
- **Universal LLM Support**: Built on LiteLLM, seamlessly switch between 100+ models - OpenAI, Anthropic, Google Gemini, DeepSeek, Azure, Mistral, local LLMs, and more. Change providers with one line of code.
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
    model="gemini/gemini-2.0-flash",  # Switch models: openai/gpt-4, anthropic/claude-3-5-sonnet, deepseek/deepseek-chat, et al.
    instructions="Help users find weather information using the available tools.",
    tools=[get_weather]
)

# Create a session to track conversation history
session = Session()

# Run the agent with a user query
runner = Runner(session)
result = runner.run(agent, "What's the weather in Monte Cristo?")

# Output the result
print("Final output:", result.final_output)

# Access the conversation history
for message in session.get_dialogs():
    print(message)
```

## Multi-Agent Example

AgentSilex supports intelligent agent handoffs, allowing a main agent to route requests to specialized sub-agents:

```python
from agentsilex import Agent, Runner, Session, tool

# Specialized weather agent with tools
@tool
def get_weather(city: str) -> str:
    """Get weather information for a city."""
    return "SUNNY"

weather_agent = Agent(
    name="Weather Agent",
    model="openai/gpt-4o-mini",
    instructions="Help users find weather information using tools",
    tools=[get_weather],
)

# Specialized FAQ agent
faq_agent = Agent(
    name="FAQ Agent",
    model="openai/gpt-4o-mini",
    instructions="Answer frequently asked questions about our products",
)

# Main orchestrator agent
main_agent = Agent(
    name="Main Agent",
    model="openai/gpt-4o-mini",
    instructions="Route user questions to the appropriate specialist agent",
    handoffs=[weather_agent, faq_agent],
)

# Execute multi-agent workflow
session = Session()
result = Runner(session).run(main_agent, "What's the weather in Paris?")
print(result.final_output)
```

## Features & Roadmap

### ✅ Implemented Features

#### Core Agent Capabilities
- **Single Agent Execution** - Create and run individual agents with custom instructions
- **Tool Calling** - Agents can call external tools/functions to perform actions
- **Tool Definition** - Simple `@tool` decorator to define callable functions with automatic schema extraction
- **Conversation Management** - Session-based dialog history tracking across multiple turns
- **Multi-Agent Handoff** - Main agent can intelligently route requests to specialized sub-agents
- **Agents as Tools** - Any agent can be converted into a tool for another agent with `agent.as_tool()`
- **Context Management** - Share mutable state across tools and conversation turns

#### Technical Features
- **Universal LLM Support** - Built on LiteLLM for seamless model switching (OpenAI, Anthropic, Google, DeepSeek, Azure, and 100+ models)
- **Type-Safe Tool Definitions** - Automatic parameter schema extraction from Python type hints
- **Transparent Architecture** - ~300 lines of readable, hackable code
- **Simple API** - Intuitive `Agent`, `Runner`, `Session`, and `@tool` abstractions

### 🚀 Roadmap

- [ ] **Streaming Support** - Real-time response streaming for better UX
- [ ] **Async Support** - Asynchronous execution for improved performance
- [ ] **Tool Call Error Handling** - Graceful handling of failed tool executions
- [ ] **Parallel Tool Execution** - Execute multiple tool calls concurrently
- [ ] **Agent Memory** - Built-in memory management for long conversations
- [ ] **State Persistence** - Save and restore agent sessions
- [ ] **Agent Observability** - Built-in logging, tracing, and debugging tools
- [ ] **Built-in Tools Library** - Common tools (web search, file operations, etc.)
- [ ] **Custom Agent Behaviors** - Pluggable behaviors (ReAct, Chain-of-Thought, etc.)
- [ ] **Human-in-the-Loop** - Built-in approval flows for sensitive operations
- [ ] **Agent Evaluation Framework** - Test and evaluate agent performance

