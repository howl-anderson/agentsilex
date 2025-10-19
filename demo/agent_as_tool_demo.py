from agentsilex import Agent, Runner, Session


# Create a translator agent that will be used as a tool
translator_agent = Agent(
    name="Translator",
    model="openai/gpt-4o-mini",
    instructions="You are a professional translator. Translate the given text to Chinese",
)

# Create the main orchestrator agent with translator as a tool
orchestrator = Agent(
    name="Orchestrator",
    model="openai/gpt-4o-mini",
    instructions="You are a helpful assistant. Use the translate_text tool when users ask for Chinese translations.",
    tools=[
        # Convert translator agent to a tool
        translator_agent.as_tool(
            tool_name="translate_text",
            tool_description="Translate text to Chinese. Provide the text.",
        ),
    ],
)

# Create a session and run the orchestrator
session = Session()
result = Runner(session).run(
    orchestrator, "Please translate 'Hello, how are you today?' to Chinese"
)

print("Final output:", result.final_output)
print("\n=== Dialog History (showing tool call) ===")
for msg in session.get_dialogs():
    print(msg)

