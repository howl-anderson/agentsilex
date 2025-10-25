"""Test utilities module."""

from agentsilex.session import Session
from agentsilex.utils import print_dialog_history


def test_print_dialog_history():
    """Test the print_dialog_history function with sample messages."""
    # Create a session with mock dialog history
    session = Session()

    # Add various message types
    session.dialogs = [
        # User message
        {"role": "user", "content": "Please add 123 and 456"},
        # Assistant with tool call
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "add",
                        "arguments": '{"a": 123, "b": 456}',
                    }
                }
            ],
        },
        # Tool response
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "The sum of 123 and 456 is 579.",
        },
        # Final assistant response
        {"role": "assistant", "content": "The result is 579."},
    ]

    # Print the dialog history
    print("\n--- Testing print_dialog_history() ---")
    print_dialog_history(session)


def test_print_long_content():
    """Test printing with long content that should be truncated."""
    session = Session()

    session.dialogs = [
        {"role": "user", "content": "Short message"},
        {
            "role": "assistant",
            "content": "This is a very long message " * 20,  # Will be truncated
        },
    ]

    print("\n--- Testing with long content ---")
    print_dialog_history(session, max_content_len=50)


if __name__ == "__main__":
    test_print_dialog_history()
    test_print_long_content()
