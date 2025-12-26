from pydantic import BaseModel
from agentsilex import Agent, Runner, Session


# Define the output structure
class WeatherReport(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int


# Create an agent with structured output
agent = Agent(
    name="Weather Reporter",
    model="gemini/gemini-2.0-flash",
    instructions="You are a weather reporter. When asked about weather, respond with realistic weather data.",
    output_type=WeatherReport,  # Specify the output type
)

# Run the agent
session = Session()
result = Runner(session).run(agent, "What's the weather in Tokyo?")

# result.final_output is now a WeatherReport instance, not a string
print(f"City: {result.final_output.city}")
print(f"Temperature: {result.final_output.temperature}C")
print(f"Condition: {result.final_output.condition}")
print(f"Humidity: {result.final_output.humidity}%")
print("----")
print(f"Type: {type(result.final_output)}")
