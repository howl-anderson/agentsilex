import json
from typing import Dict

from dotenv import load_dotenv
from litellm import completion

from agentsilex.agent import Agent, HANDOFF_TOOL_PREFIX
from agentsilex.run_result import RunResult
from agentsilex.session import Session

load_dotenv()


def user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def bot_msg(content: str) -> dict:
    return {"role": "assistant", "content": content}


class Runner:
    def __init__(self, session: Session, context: dict | None = None):
        self.session = session

        # this is the content, a dict that will be passed to tools when executed, it can be read and written by tools
        self.context = context or {}

    def run(
        self,
        agent: Agent,
        prompt: str,
        context: dict | None = None,
    ) -> RunResult:
        current_agent = agent

        msg = user_msg(prompt)
        self.session.add_new_messages([msg])

        loop_count = 0
        should_stop = False
        while loop_count < 10 and not should_stop:
            dialogs = self.session.get_dialogs()

            tools_spec = (
                current_agent.tools_set.get_specification()
                + current_agent.handoffs.get_specification()
            )

            # because system prompt is depend on current agent,
            # so we get the full dialogs here, just before calling the model
            complete_dialogs = [current_agent.get_system_prompt()] + dialogs
            response = completion(
                model=current_agent.model,
                messages=complete_dialogs,
                tools=tools_spec if tools_spec else None,
            )

            response_message = response.choices[0].message

            self.session.add_new_messages([response_message])

            if not response_message.tool_calls:
                should_stop = True
                return RunResult(
                    final_output=response_message.content,
                )

            # deal with normal function calls firstly
            tools_response = [
                current_agent.tools_set.execute_function_call(self.context, call_spec)
                for call_spec in response_message.tool_calls
                if not call_spec.function.name.startswith(HANDOFF_TOOL_PREFIX)
            ]

            self.session.add_new_messages(tools_response)

            # then deal with agent handoff calls sencondly
            handoff_responses = [
                call_spec
                for call_spec in response_message.tool_calls
                if call_spec.function.name.startswith(HANDOFF_TOOL_PREFIX)
            ]
            if handoff_responses:
                # if there are multiple handoff, just pick the first one
                agent_spec = handoff_responses[0]
                current_agent, handoff_response = current_agent.handoffs.handoff_agent(
                    agent_spec
                )
                self.session.add_new_messages([handoff_response])

            loop_count += 1

        return RunResult(
            final_output="Error: Exceeded max iterations",
        )

    def convert_function_call_response_to_messages(
        self, function_call_response_list
    ) -> Dict[str, str]:
        return user_msg(json.dumps(function_call_response_list))
