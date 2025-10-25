"""Utility functions for agentsilex."""


def print_dialog_history(session, max_content_len: int = 100):
    """
    Print the dialog history from a session in a compact format.

    Args:
        session: The Session object containing dialog history
        max_content_len: Maximum length for content display (default: 100)
    """
    dialogs = session.get_dialogs()

    print("\n=== Dialog History ===")
    for i, msg in enumerate(dialogs, 1):
        # Handle both dict and Message object types
        if isinstance(msg, dict):
            role = msg.get("role", "N/A")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")
        else:
            role = getattr(msg, "role", "N/A")
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)
            tool_call_id = getattr(msg, "tool_call_id", None)

        # Build compact one-line message
        parts = [f"[{i}] {role}:"]

        if tool_calls:
            # Extract tool call info
            tool_info = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    tool_args = func.get("arguments", "{}")
                else:
                    tool_name = getattr(tc, "name", "unknown")
                    tool_args = getattr(tc, "arguments", "{}")
                tool_info.append(f"{tool_name}({tool_args})")
            parts.append("CALL " + ", ".join(tool_info))
        elif tool_call_id:
            # Truncate tool response content
            content_str = str(content) if content else ""
            if len(content_str) > max_content_len:
                content_str = content_str[:max_content_len] + "..."
            parts.append(f"RESULT {content_str}")
        elif content:
            # Regular content
            content_str = str(content)
            if len(content_str) > max_content_len:
                content_str = content_str[:max_content_len] + "..."
            parts.append(content_str)

        print(" ".join(parts))

    print("=" * 80)
