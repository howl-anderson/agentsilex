# AgentSilex

[![PyPI version](https://badge.fury.io/py/agentsilex.svg)](https://badge.fury.io/py/agentsilex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Downloads](https://pepy.tech/badge/agentsilex)](https://pepy.tech/project/agentsilex)

A transparent, minimal, and hackable agent framework for developers who want full control.

**Read the entire codebase in one sitting. Understand exactly how your agents work. No magic, no hidden complexity.**

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

## Demo

![AgentSilex Streaming Demo](media/agentsilex_streaming_demo.webp)

*Real-time streaming response demonstration - see how AgentSilex processes queries and streams results*

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

print(result.final_output)
# Output: "The weather in Monte Cristo is SUNNY."
```

**That's it!** In just 20 lines, you have a working agent that:
- ✅ Uses any LLM (OpenAI, Anthropic, Google, DeepSeek, etc.)
- ✅ Calls tools to get information
- ✅ Maintains conversation history
- ✅ Returns natural language responses

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

## Observability

AgentSilex includes built-in OpenTelemetry tracing to visualize agent execution, tool calls, and handoffs.

### Quick Setup with Phoenix

```bash
# Install and start Phoenix
pip install arize-phoenix
python -m phoenix.server.main serve

# Set the endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# Run your agent - view traces at http://localhost:6006
```

Phoenix will show a complete trace tree of your agent workflow, including all tool calls and agent handoffs.

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
- **Structured Output** - Define Pydantic models for type-safe, validated responses
- **Type-Safe Tool Definitions** - Automatic parameter schema extraction from Python type hints
- **Transparent Architecture** - ~300 lines of readable, hackable code
- **Simple API** - Intuitive `Agent`, `Runner`, `Session`, and `@tool` abstractions
- **OpenTelemetry Observability** - Built-in tracing compatible with Phoenix and other OTLP backends
- **Streaming Support** - Real-time response streaming with event-based architecture for better UX
- **Agent Memory** - Callback-based memory management for conversation history control
- **MCP Client Support** - Connect to Model Context Protocol servers to extend agent capabilities with external tools
- **Custom Agent Behaviors** - Pluggable callback system for implementing custom behaviors (ReAct, Chain-of-Thought, logging, etc.)

### 🚀 Roadmap

- [ ] **Async Support** - Asynchronous execution for improved performance
- [ ] **Tool Call Error Handling** - Graceful handling of failed tool executions
- [ ] **Parallel Tool Execution** - Execute multiple tool calls concurrently
- [ ] **State Persistence** - Save and restore agent sessions
- [ ] **Built-in Tools Library** - Common tools (web search, file operations, etc.)
- [ ] **Human-in-the-Loop** - Built-in approval flows for sensitive operations
- [ ] **Agent Evaluation Framework** - Test and evaluate agent performance

