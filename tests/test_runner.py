"""Core runner tests - minimal, focused on critical paths."""

from unittest.mock import MagicMock, patch
from pydantic import BaseModel

from agentsilex import Agent, Runner, Session, tool


def test_agent_creation():
    """Agent can be created with basic parameters."""
    agent = Agent(
        name="test",
        model="gpt-4o",
        instructions="You are a test agent.",
    )
    assert agent.name == "test"
    assert agent.model == "gpt-4o"
    assert agent.output_type is None


def test_agent_with_output_type():
    """Agent accepts output_type parameter."""

    class TestOutput(BaseModel):
        value: str

    agent = Agent(
        name="test",
        model="gpt-4o",
        instructions="Test",
        output_type=TestOutput,
    )
    assert agent.output_type == TestOutput


def test_tool_decorator():
    """@tool decorator creates FunctionTool."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add.name == "add"
    assert add.description == "Add two numbers."
    assert add(a=1, b=2) == 3


def test_runner_with_mocked_llm():
    """Runner executes agent and returns result."""
    agent = Agent(
        name="test",
        model="gpt-4o",
        instructions="Test",
    )
    session = Session()
    runner = Runner(session)

    # Mock LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello, world!"
    mock_response.choices[0].message.tool_calls = None

    with patch("agentsilex.runner.completion", return_value=mock_response):
        result = runner.run(agent, "Hi")

    assert result.final_output == "Hello, world!"


def test_runner_with_structured_output():
    """Runner validates structured output with Pydantic."""

    class Weather(BaseModel):
        city: str
        temp: float

    agent = Agent(
        name="weather",
        model="gpt-4o",
        instructions="Return weather",
        output_type=Weather,
    )
    session = Session()
    runner = Runner(session)

    # Mock LLM returning valid JSON
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"city": "Tokyo", "temp": 25.5}'
    mock_response.choices[0].message.tool_calls = None

    with patch("agentsilex.runner.completion", return_value=mock_response):
        result = runner.run(agent, "Weather in Tokyo?")

    assert isinstance(result.final_output, Weather)
    assert result.final_output.city == "Tokyo"
    assert result.final_output.temp == 25.5
