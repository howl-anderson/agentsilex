# AgentSilex Web Demo

An interactive web demo showcasing AgentSilex's streaming capabilities with real-time visualization of agent thinking, tool calls, and intermediate steps.

## Features

- **Streaming Responses**: Real-time token-by-token streaming using Server-Sent Events (SSE)
- **Tool Call Visualization**: See when the agent calls tools, what arguments it uses, and the results
- **Agent Handoff Display**: Visual notifications when the agent switches between sub-agents
- **Modern UI**: Beautiful, responsive interface built with Tailwind CSS
- **Markdown Support**: Formatted responses with code syntax highlighting
- **Session Management**: Persistent conversation history per session

## Quick Start

### 1. Set Up Environment

Create a `.env` file in the **parent directory** (agentsilex root, not web_demo) with your API keys:

```bash
cd /path/to/agentsilex  # Go to agentsilex root
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

Or add to existing `.env`:

```env
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Or for other providers (Anthropic, Google, etc.)
# ANTHROPIC_API_KEY=your_key
# GOOGLE_API_KEY=your_key
```

### 2. Install Dependencies

This project uses **uv** for package management (required for editable agentsilex install):

```bash
cd web_demo
uv pip install -r requirements.txt
```

### 3. Run the Server

**Option A: Use the convenience script (recommended)**

```bash
./run.sh
```

**Option B: Run directly with uv**

```bash
uv run python app.py
```

**Option C: Using uvicorn directly**

```bash
uv run uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

### 4. Open in Browser

Navigate to: http://localhost:8080

## Project Structure

```
web_demo/
├── app.py                 # FastAPI backend with SSE streaming
├── requirements.txt       # Python dependencies
├── run.sh                # Convenience startup script
├── README.md             # This file
└── static/
    ├── index.html        # Main HTML page (Tailwind CSS)
    └── chat.js           # Frontend JavaScript (SSE handling)

Note: .env file should be in the parent directory (agentsilex root)
```

## How It Works

### Backend (app.py)

1. **Agent Setup**: Creates an AgentSilex agent with multiple tools (weather, calculator, search)
2. **SSE Endpoint**: `/chat` endpoint streams events from `runner.run_stream()`
3. **Event Processing**: Converts AgentSilex events to JSON and sends via SSE

### Frontend (index.html + chat.js)

1. **User Input**: Captures messages and sends to `/chat` endpoint
2. **SSE Handling**: Listens to event stream and processes different event types
3. **UI Updates**: Dynamically renders messages, tool calls, and status updates

### Event Types

The demo handles these AgentSilex event types:

- `partial_output`: Token-by-token streaming text
- `tool_call`: When agent calls a tool (shows tool name and arguments)
- `tool_response`: Result from tool execution
- `agent_handoff`: When switching between agents
- `final_result`: Complete response
- `error`: Error messages

## Customization

### Adding New Tools

Edit `app.py` and add new tools using the `@tool` decorator:

```python
@tool
def my_custom_tool(param: str) -> str:
    """Description of what this tool does."""
    # Your implementation
    return "Result"

# Add to agent's tools list
demo_agent = Agent(
    name="DemoAssistant",
    tools=[..., my_custom_tool],
    ...
)
```

### Changing the Model

Modify the `model` parameter in `app.py`:

```python
demo_agent = Agent(
    model="gpt-4o",  # or "claude-3-5-sonnet-20241022", "gemini/gemini-2.0-flash", etc.
    ...
)
```

### Styling

The UI uses Tailwind CSS (CDN). To customize:

1. Edit classes in `static/index.html`
2. Modify custom CSS in the `<style>` tag
3. Or use a custom Tailwind configuration

## API Endpoints

- `GET /` - Main web interface
- `POST /chat` - SSE streaming chat endpoint
  - Body: `{"message": "user message", "session_id": "optional_session_id"}`
- `POST /reset` - Reset session
  - Body: `{"session_id": "session_id"}`
- `GET /health` - Health check

## Dependencies

- **FastAPI**: Modern web framework for Python
- **uvicorn**: ASGI server
- **agentsilex**: The agent framework
- **Tailwind CSS**: Utility-first CSS framework (CDN)
- **Marked.js**: Markdown parser (CDN)
- **Highlight.js**: Code syntax highlighting (CDN)

## Troubleshooting

### API Key Errors

Make sure your `.env` file in the **parent directory** contains the correct API key for your chosen model provider.

### AgentSilex Not Found

If you get `ModuleNotFoundError: No module named 'agentsilex'`, make sure you're using `uv run`:

```bash
uv run python app.py  # NOT just: python app.py
```

This ensures the editable install of agentsilex is available.

### Port Already in Use

Change the port in `app.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8001)  # Use a different port
```

Then run with: `uv run python app.py`

### Streaming Not Working

Some reverse proxies (like nginx) may buffer responses. The app includes the `X-Accel-Buffering: no` header to prevent this.

## License

This demo is part of the AgentSilex project and follows the same MIT license.

## Learn More

- [AgentSilex GitHub](https://github.com/howl-anderson/agentsilex)
- [AgentSilex Documentation](https://github.com/howl-anderson/agentsilex/blob/main/README.md)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
