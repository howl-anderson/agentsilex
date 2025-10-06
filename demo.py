from agentsilex import Agent, Runner, Session, tool


# add a mock weather tool, which always returns "SUNNY"
@tool
def get_weather(city: str) -> str:
    return "SUNNY"


# create an agent with the weather tool
agent = Agent(
    name="A weather agent",
    model="gemini/gemini-2.5-flash",
    instructions="Help user to find the weather by using tools",
    tools=[get_weather],
)

# create a session, which will keep track of the dialog history
session = Session()

# Run the agent with a user query
result = Runner(session).run(agent, "What's the weather in Monte Cristo")

# output the result and the dialog history
print("Final output: ", result.final_output)
print("----")
print("Dialog history: ")
for i in session.get_dialogs():
    print(i)
