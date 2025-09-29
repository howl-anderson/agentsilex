import json
from typing import Dict

from dotenv import load_dotenv
from litellm import completion

from agentsilex.agent import Agent
from agentsilex.run_result import RunResult
from agentsilex.session import Session

load_dotenv()


def user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def bot_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


class Runner:
    def __init__(self, agent: Agent, session: Session):
        self.agent = agent
        self.session = session

    def run(
        self,
        prompt: str,
    ) -> RunResult:
        should_stop = False

        msg = user_msg(prompt)
        self.session.add_new_messages([msg])

        loop_count = 0
        while loop_count < 10 and not should_stop:
            dialogs = self.session.get_dialogs()

            tools_spec = self.agent.tools_set.get_specification()

            response = completion(
                model=self.agent.model,
                messages=dialogs,
                tools=tools_spec if tools_spec else None,
            )

            response_message = response.choices[0].message

            self.session.add_new_messages([response_message])

            if not response_message.tool_calls:
                should_stop = True
                return RunResult(
                    final_output=response_message.content,
                )

            tools_response = [
                self.agent.tools_set.execute_function_call(call_spec)
                for call_spec in response_message.tool_calls
            ]

            self.session.add_new_messages(tools_response)

            loop_count += 1

        return RunResult(
            final_output="Error: Exceeded max iterations",
        )

    def convert_function_call_response_to_messages(
        self, function_call_response_list
    ) -> Dict[str, str]:
        return user_msg(json.dumps(function_call_response_list))
