from agentsilex import Agent, Runner, Session, tool


# add a mock weather tool, which always returns "SUNNY"
@tool
def get_weather(city: str) -> str:
    return "SUNNY"


# create an agent with the weather tool
agent = Agent(
    name="A weather agent",
    model="openai/gpt-4o-mini",
    instructions="Help user to find the weather by using tools. You should think aloud, tell user your plan.",
    tools=[get_weather],
)

# create a session, which will keep track of the dialog history
session = Session()

# Run the agent with a user query
result = Runner(session).run_stream(agent, "What's the weather in Monte Cristo")

for event in result:
    if event.type == "raw_chunk":
        continue  # skip raw chunk events, too verbose
    if event.type == "final_result":
        continue  # skip final output event, already printed in the steam

    print(event)
