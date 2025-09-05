from playwright.sync_api import sync_playwright

from agentsilex.commands.context import Context, set_context
from agentsilex.commands.register import TOOLS_REGISTRY
import agentsilex.commands.builtin_commands
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

load_dotenv()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    set_context(Context(page))

    llm = init_chat_model(
        "gemini-2.5-flash", temperature=0.0, model_provider="google_genai"
    )

    agent = create_react_agent(llm, TOOLS_REGISTRY)

    reponse = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "got to https://www.xiaoquankong.ai/",
                }
            ]
        }
    )

    print(page.title())

    print("done!")

    browser.close()
