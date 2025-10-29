from agentsilex import Agent, Runner, Session
from agentsilex.callbacks import keep_most_recent_2_turns

agent = Agent(
    name="Assistant",
    model="gemini/gemini-2.0-flash-exp",
    instructions="You are a helpful assistant. Just echo my message",
)

# Demo 1: Without history management (history grows)
print("=== Without History Management ===")
session1 = Session()
runner1 = Runner(session1)

for i in range(5):
    result = runner1.run(agent, f"This is message {i}")

print(f"History length: {len(session1.dialogs)} messages")
print()

# Demo 2: With history management (keep only recent 2 turns)
print("=== With History Management (keep 2 turns) ===")
session2 = Session()
runner2 = Runner(session2, before_llm_call_callbacks=[keep_most_recent_2_turns])

for i in range(5):
    result = runner2.run(agent, f"This is message {i}")

print(f"History length: {len(session2.dialogs)} messages (limited to ~4)")
