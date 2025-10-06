from agentsilex import Agent, Runner, Session, tool


# add a mock weather tool, which always returns "SUNNY"
@tool
def get_weather(city: str) -> str:
    return "SUNNY"


weather_agent = Agent(
    name="A weather agent",
    model="openai/gpt-4o-mini",
    instructions="Help user to find the weather by using tools",
    tools=[get_weather],
)

faq_agent = Agent(
    name="A FAQ agent",
    model="openai/gpt-4o-mini",
    instructions="Help user to answer FAQ questions",
)

main_agent = Agent(
    name="Main agent",
    model="openai/gpt-4o-mini",
    instructions="You will identify questions and pass it to other agents.",
    handoffs=[weather_agent, faq_agent],
)

# create a session, which will keep track of the dialog history
session = Session()

# Run the agent with a user query
result = Runner(session).run(main_agent, "What's the weather in Monte Cristo")

# output the result and the dialog history
print("Final output: ", result.final_output)
print("----")
print("Dialog history: ")
for i in session.get_dialogs():
    print(i)
