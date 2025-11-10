"""
AgentSilex Web Demo - FastAPI Backend with Server-Sent Events (SSE)
"""

import json
import asyncio
import time
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path

from agentsilex import Agent, Runner, Session, tool

# Get the directory where this file is located
BASE_DIR = Path(__file__).parent

# Load environment variables from parent directory (agentsilex root)
env_path = BASE_DIR.parent / ".env"
load_dotenv(dotenv_path=env_path)


# Define example tools
@tool
def get_current_date_and_location():
    """
    Get the current date and location information.

    Returns:
        A string containing the current date, location, and zip code.
    """
    # Simulate API call delay for demo video (longer for visibility)
    time.sleep(2.0)
    return "Current date: Nov 10 2025. Location: San Francisco, CA. Zip code: 94102."


@tool
def get_weather(date: str, zip_code: str):
    """
    Get the weather information for a given date and zip code.

    Args:
        date: A string representing the date for which to get the weather, in format 'YYYY-MM-DD'.
        zip_code: A string representing the zip code for which to get the weather.

    Returns:
        A string containing the weather information.
    """
    # Simulate weather API call delay (longer for demo visibility)
    time.sleep(2.5)
    return f"Weather on {date} for zip code {zip_code}: Sunny, 22°C, light breeze."


@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        The result of the calculation.
    """
    # Simulate computation delay (longer for demo visibility)
    time.sleep(1.5)
    try:
        # Simple and safe evaluation for basic math
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# Create the agent
demo_agent = Agent(
    name="DemoAssistant",
    model="gpt-4o-mini",  # Using a fast model for demo
    instructions="""
You are a helpful AI assistant with access to various tools.
You can help users with:
- Weather information (get current date/location and weather data)
- Mathematical calculations
- General questions and conversations

When using tools:
1. Explain what you're doing clearly
2. Use tools when they would be helpful
3. Provide friendly, concise responses

Be conversational and helpful!
    """.strip(),
    tools=[get_current_date_and_location, get_weather, calculate],
)


# Session storage (in production, use Redis or similar)
sessions = {}


def get_or_create_session(session_id: str) -> Session:
    """Get or create a session for the given ID."""
    if session_id not in sessions:
        sessions[session_id] = Session()
    return sessions[session_id]


# FastAPI app
app = FastAPI(
    title="AgentSilex Web Demo",
    description="Interactive demo showcasing AgentSilex capabilities",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


async def event_generator(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events from the agent's stream.
    Converts sync generator to async generator.
    """
    session = get_or_create_session(session_id)
    runner = Runner(session)

    try:
        # Run the agent stream in a thread pool to avoid blocking
        for event in runner.run_stream(demo_agent, message):
            # Convert event to JSON
            event_data = {
                "type": event.type,
                "data": {}
            }

            # Process different event types
            if event.type == "partial_output":
                event_data["data"]["content"] = event.data.get("content", "")
            elif event.type == "tool_call":
                func_request = event.data.get("func_request")
                event_data["data"]["tool_name"] = func_request.function.name
                event_data["data"]["tool_args"] = func_request.function.arguments
            elif event.type == "tool_response":
                func_response = event.data.get("func_response")
                # func_response is a dict: {"role": "tool", "tool_call_id": "...", "content": "..."}
                event_data["data"]["tool_result"] = func_response.get("content", str(func_response))
            elif event.type == "agent_handoff":
                event_data["data"]["agent_name"] = event.data.get("agent_name", "")
            elif event.type == "final_result":
                final_output = event.data.get("final_output")
                if hasattr(final_output, 'content'):
                    event_data["data"]["content"] = final_output.content
                else:
                    event_data["data"]["content"] = str(final_output)

            # Format as SSE
            yield f"data: {json.dumps(event_data)}\n\n"

            # Small delay to allow browser to process
            await asyncio.sleep(0.001)

    except Exception as e:
        error_data = {
            "type": "error",
            "data": {"message": str(e)}
        }
        yield f"data: {json.dumps(error_data)}\n\n"

    finally:
        # Send done event
        yield f"data: {json.dumps({'type': 'done', 'data': {}})}\n\n"


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream chat responses using Server-Sent Events.
    """
    return StreamingResponse(
        event_generator(request.message, request.session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.post("/reset")
async def reset_session(request: dict):
    """Reset a session."""
    session_id = request.get("session_id", "default")
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "ok", "message": f"Session {session_id} reset"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "agent": demo_agent.name, "model": demo_agent.model}


# Mount static files (must be after route definitions)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
