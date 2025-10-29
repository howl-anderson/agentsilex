from agentsilex.session import Session


def keep_most_recent_2_turns(session: Session):
    MOST_RECENT = 2  # dialog turns to keep

    msg_count = 2 * MOST_RECENT  # include user and agent messages

    if msg_count < len(session.dialogs):
        session.dialogs = session.dialogs[-msg_count + 1 :]
