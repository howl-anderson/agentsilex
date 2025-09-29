from typing import List


class Session:
    def __init__(self):
        self.dialogs = []

    def get_dialogs(self) -> List[dict]:
        return self.dialogs

    def add_new_messages(self, messages):
        self.dialogs.extend(messages)

    def pop_turn(self):
        pass

    def clean(self):
        self.dialogs = []
