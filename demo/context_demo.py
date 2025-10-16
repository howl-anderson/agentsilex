from agentsilex import Agent, Runner, Session, tool


@tool
def get_user_name(context: dict) -> str:
    """Read from context"""
    return f"User name: {context.get('user_name', 'Unknown')}"


@tool
def save_city(city: str, context: dict) -> str:
    """Write to context"""
    context["favorite_city"] = city
    return f"Saved {city} as favorite city"


agent = Agent(
    name="Assistant",
    model="gemini/gemini-2.0-flash-exp",
    instructions="Use tools to help user.",
    tools=[get_user_name, save_city],
)

session = Session()
context = {"user_name": "Alice"}
runner = Runner(session, context=context)

print("Initial context:", context)
print()

# Turn 1: Read
print("Turn 1: What's my name?")
result1 = runner.run(agent, "What's my name?")
print("Agent:", result1.final_output)
print("Context:", context)
print()

# Turn 2: Write
print("Turn 2: Save Paris as my favorite city")
result2 = runner.run(agent, "Save Paris as my favorite city")
print("Agent:", result2.final_output)
print("Context:", context)
