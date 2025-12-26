"""
Simplified function schema extraction for LLM function calling.
Uses docstring_parser and pydantic for clean, maintainable implementation.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional, Union, get_args, get_origin, get_type_hints

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model


def extract_function_schema(
    func: Any,
    name_override: str | None = None,
    description_override: str | None = None,
) -> tuple[str, str | None, dict[str, Any]]:
    """
    Given a Python function, extracts schema information for LLM function calling.

    This simplified version:
    - Uses docstring_parser for automatic format detection (Google/Numpy/ReST)
    - Supports standard function parameters (no *args/**kwargs)
    - Handles Optional types and default values
    - Generates standard JSON Schema via Pydantic

    Args:
        func: The function to extract the schema from.
        name_override: If provided, use this name instead of the function's `__name__`.
        description_override: If provided, use this description instead of the one from docstring.

    Returns:
        A tuple of (name, description, params_json_schema).
    """
    # 1. Basic information
    func_name = name_override or func.__name__
    sig = inspect.signature(func)

    # 2. Parse docstring using docstring_parser
    docstring = parse(func.__doc__ or "")
    description = description_override or docstring.short_description

    # Create parameter description mapping
    param_docs = {param.arg_name: param.description for param in docstring.params}

    # 3. Get type hints
    type_hints = get_type_hints(func)

    # 4. Build Pydantic fields
    fields = {}
    for param_name, param in sig.parameters.items():
        # Skip self/cls parameters
        if param_name in ("self", "cls"):
            continue

        # Skip context parameter, which is a reserved name for context injection
        if param_name in ("context",):
            continue

        # Get type annotation (default to Any if not provided)
        param_type = type_hints.get(param_name, Any)

        # Get parameter description from docstring
        param_desc = param_docs.get(param_name)

        # Check if it's an Optional type
        is_optional, base_type = _unwrap_optional(param_type)

        # Determine if parameter is required
        if param.default == inspect.Parameter.empty and not is_optional:
            # Required parameter
            fields[param_name] = (base_type, Field(..., description=param_desc))
        else:
            # Optional parameter with default value
            default_value = (
                param.default if param.default != inspect.Parameter.empty else None
            )
            fields[param_name] = (
                base_type,
                Field(default=default_value, description=param_desc),
            )

    # 5. Create dynamic Pydantic model
    if fields:
        dynamic_model = create_model(f"{func_name}_params", **fields)
    else:
        # No parameters - create empty model
        dynamic_model = create_model(f"{func_name}_params", __base__=BaseModel)

    # 6. Generate JSON schema
    json_schema = dynamic_model.model_json_schema()

    return func_name, description, json_schema


def _unwrap_optional(type_hint: Any) -> tuple[bool, Any]:
    """
    Check if a type hint is Optional (Union[T, None]) and unwrap it.

    Args:
        type_hint: The type hint to check

    Returns:
        A tuple of (is_optional, base_type)
    """
    origin = get_origin(type_hint)

    # Check if it's a Union type
    if origin is Union:
        args = get_args(type_hint)
        # Optional[T] is equivalent to Union[T, None]
        if type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return True, non_none_args[0]
            # Union with multiple non-None types
            return True, Union[tuple(non_none_args)]

    return False, type_hint


# Simplified example usage
if __name__ == "__main__":
    from typing import List

    def example_function(
        name: str,
        age: Optional[int] = None,
        tags: List[str] = None,
        is_active: bool = True,
    ) -> dict:
        """
        Process user information.

        Args:
            name: The user's full name
            age: The user's age in years
            tags: List of tags associated with the user
            is_active: Whether the user account is active

        Returns:
            A dictionary containing processed user data
        """
        return {"name": name, "age": age, "tags": tags or [], "is_active": is_active}

    # Extract schema
    name, description, params_json_schema = extract_function_schema(example_function)
    print(f"Function: {name}")
    print(f"Description: {description}")
    print(f"Parameters JSON Schema: {params_json_schema}")
