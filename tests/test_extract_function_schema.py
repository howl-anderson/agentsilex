"""
Unit tests for extract_function_schema module.
Tests end-to-end functionality for common function patterns.
"""

import pytest
from typing import List, Optional, Union, Dict, Any
from pydantic import ValidationError

from agentsilex.extract_function_schema import function_schema, FuncSchema


class TestFunctionSchemaExtraction:
    """Test function schema extraction for various function patterns."""

    def test_simple_function_with_required_params(self):
        """Test extraction for a simple function with only required parameters."""

        def simple_func(name: str, age: int) -> str:
            """
            Create a greeting message.

            Args:
                name: Person's name
                age: Person's age
            """
            return f"Hello {name}, you are {age} years old"

        schema = function_schema(simple_func)

        # Verify basic properties
        assert schema.name == "simple_func"
        assert schema.description == "Create a greeting message."

        # Verify JSON schema structure
        json_schema = schema.params_json_schema
        assert json_schema["type"] == "object"
        assert "name" in json_schema["properties"]
        assert "age" in json_schema["properties"]
        assert json_schema["properties"]["name"]["type"] == "string"
        assert json_schema["properties"]["age"]["type"] == "integer"
        assert set(json_schema["required"]) == {"name", "age"}

        # Verify parameter descriptions
        assert json_schema["properties"]["name"]["description"] == "Person's name"
        assert json_schema["properties"]["age"]["description"] == "Person's age"

    def test_function_with_optional_params(self):
        """Test extraction for a function with optional parameters."""

        def func_with_optional(
            required_param: str,
            optional_param: Optional[int] = None,
            default_param: bool = True
        ) -> dict:
            """
            Function with mixed parameter types.

            Args:
                required_param: A required string parameter
                optional_param: An optional integer parameter
                default_param: A parameter with default value
            """
            return {}

        schema = function_schema(func_with_optional)

        json_schema = schema.params_json_schema

        # Check required parameters
        assert "required" in json_schema
        assert "required_param" in json_schema["required"]
        assert "optional_param" not in json_schema.get("required", [])
        assert "default_param" not in json_schema.get("required", [])

        # Check default values
        props = json_schema["properties"]
        assert "default" not in props["required_param"]  # No default for required
        assert props["optional_param"].get("default") is None
        assert props["default_param"].get("default") is True

    def test_function_with_complex_types(self):
        """Test extraction for a function with complex type annotations."""

        def complex_types_func(
            items: List[str],
            mapping: Dict[str, int],
            union_param: Union[str, int] = "default"
        ) -> List[Dict[str, Any]]:
            """
            Handle complex types.

            Args:
                items: List of string items
                mapping: Dictionary mapping strings to integers
                union_param: Can be string or integer
            """
            return []

        schema = function_schema(complex_types_func)

        json_schema = schema.params_json_schema
        props = json_schema["properties"]

        # Check list type
        assert props["items"]["type"] == "array"
        assert props["items"]["items"]["type"] == "string"

        # Check dict type
        assert props["mapping"]["type"] == "object"

        # Check union type (should be handled)
        assert "union_param" in props
        assert props["union_param"].get("default") == "default"

    def test_function_without_docstring(self):
        """Test extraction for a function without a docstring."""

        def no_docstring_func(param1: str, param2: int = 10):
            return param1 * param2

        schema = function_schema(no_docstring_func)

        assert schema.name == "no_docstring_func"
        assert schema.description is None

        json_schema = schema.params_json_schema
        assert "param1" in json_schema["properties"]
        assert "param2" in json_schema["properties"]
        assert json_schema["properties"]["param2"]["default"] == 10

    def test_function_without_type_hints(self):
        """Test extraction for a function without type hints."""

        def no_types_func(param1, param2=None):
            """
            Function without type annotations.

            Args:
                param1: First parameter
                param2: Second parameter
            """
            return str(param1) + str(param2)

        schema = function_schema(no_types_func)

        json_schema = schema.params_json_schema

        # Without type hints, parameters should still be present
        assert "param1" in json_schema["properties"]
        assert "param2" in json_schema["properties"]

        # Descriptions should still be extracted from docstring
        assert json_schema["properties"]["param1"]["description"] == "First parameter"
        assert json_schema["properties"]["param2"]["description"] == "Second parameter"

    def test_class_method_extraction(self):
        """Test extraction for class methods (should skip 'self' parameter)."""

        class MyClass:
            def method(self, param: str) -> str:
                """
                A class method.

                Args:
                    param: A string parameter
                """
                return param.upper()

        obj = MyClass()
        schema = function_schema(obj.method)

        json_schema = schema.params_json_schema

        # 'self' should not appear in the schema
        assert "self" not in json_schema["properties"]
        assert "param" in json_schema["properties"]
        assert len(json_schema["properties"]) == 1

    def test_function_with_name_and_description_override(self):
        """Test name and description override functionality."""

        def original_func(x: int) -> int:
            """Original description."""
            return x * 2

        schema = function_schema(
            original_func,
            name_override="custom_name",
            description_override="Custom description"
        )

        assert schema.name == "custom_name"
        assert schema.description == "Custom description"

    def test_to_call_args_conversion(self):
        """Test converting validated data back to function call arguments."""

        def test_func(pos_arg: str, keyword_arg: int = 5) -> str:
            """Test function for argument conversion."""
            return f"{pos_arg}-{keyword_arg}"

        schema = function_schema(test_func)

        # Create an instance of the Pydantic model with test data
        model_instance = schema.params_pydantic_model(
            pos_arg="test",
            keyword_arg=10
        )

        # Convert to call arguments
        args, kwargs = schema.to_call_args(model_instance)

        # For POSITIONAL_OR_KEYWORD parameters, they should be in positional args
        assert args == ["test", 10]
        assert kwargs == {}

        # Verify the function can be called with these arguments
        result = test_func(*args, **kwargs)
        assert result == "test-10"

    def test_function_with_no_parameters(self):
        """Test extraction for a function with no parameters."""

        def no_params_func() -> str:
            """Function with no parameters."""
            return "result"

        schema = function_schema(no_params_func)

        assert schema.name == "no_params_func"
        assert schema.description == "Function with no parameters."

        json_schema = schema.params_json_schema
        assert json_schema["type"] == "object"
        assert json_schema["properties"] == {}
        assert json_schema.get("required", []) == []

    def test_pydantic_model_validation(self):
        """Test that the generated Pydantic model performs validation correctly."""

        def validated_func(
            name: str,
            age: int,
            email: Optional[str] = None
        ) -> dict:
            """
            Function with validation requirements.

            Args:
                name: User's name (required)
                age: User's age (required)
                email: User's email (optional)
            """
            return {"name": name, "age": age, "email": email}

        schema = function_schema(validated_func)

        # Valid data should work
        valid_data = schema.params_pydantic_model(
            name="Alice",
            age=30,
            email="alice@example.com"
        )
        assert valid_data.name == "Alice"
        assert valid_data.age == 30
        assert valid_data.email == "alice@example.com"

        # Missing required field should raise ValidationError
        with pytest.raises(ValidationError):
            schema.params_pydantic_model(name="Bob")  # Missing 'age'

        # Wrong type should raise ValidationError
        with pytest.raises(ValidationError):
            schema.params_pydantic_model(name="Charlie", age="not_an_int")


class TestGoogleStyleDocstring:
    """Test Google-style docstring parsing."""

    def test_google_style_parsing(self):
        """Test that Google-style docstrings are parsed correctly."""

        def google_style_func(
            param1: str,
            param2: int = 10
        ) -> str:
            """
            This is a summary line.

            This is a longer description that provides more details
            about what the function does.

            Args:
                param1: Description of param1
                param2: Description of param2 with default

            Returns:
                A string result
            """
            return f"{param1}-{param2}"

        schema = function_schema(google_style_func)

        assert schema.description == "This is a summary line."

        json_schema = schema.params_json_schema
        assert json_schema["properties"]["param1"]["description"] == "Description of param1"
        assert json_schema["properties"]["param2"]["description"] == "Description of param2 with default"


class TestNumpyStyleDocstring:
    """Test Numpy-style docstring parsing."""

    def test_numpy_style_parsing(self):
        """Test that Numpy-style docstrings are parsed correctly."""

        def numpy_style_func(
            param1: str,
            param2: int = 10
        ) -> str:
            """
            This is a summary line.

            This is a longer description.

            Parameters
            ----------
            param1 : str
                Description of param1
            param2 : int
                Description of param2 with default

            Returns
            -------
            str
                A string result
            """
            return f"{param1}-{param2}"

        schema = function_schema(numpy_style_func)

        assert schema.description == "This is a summary line."

        json_schema = schema.params_json_schema
        assert json_schema["properties"]["param1"]["description"] == "Description of param1"
        assert json_schema["properties"]["param2"]["description"] == "Description of param2 with default"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])