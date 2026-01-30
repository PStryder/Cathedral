"""
Tests for SubAgentGate worker context formatting.
"""

import pytest
from cathedral.SubAgentGate.worker import (
    _format_context_value,
    _format_context,
    MAX_CONTEXT_VALUE_LEN,
    MAX_CONTEXT_TOTAL_LEN,
)


class TestFormatContextValue:
    """Tests for _format_context_value."""

    def test_short_string_unchanged(self):
        """Short strings pass through unchanged."""
        result = _format_context_value("hello")
        assert result == "hello"

    def test_long_string_truncated(self):
        """Long strings are truncated with ellipsis."""
        long_str = "x" * 1000
        result = _format_context_value(long_str)
        assert len(result) == MAX_CONTEXT_VALUE_LEN
        assert result.endswith("...")

    def test_none_returns_null(self):
        """None becomes 'null'."""
        result = _format_context_value(None)
        assert result == "null"

    def test_dict_serialized_as_json(self):
        """Dicts are serialized as compact JSON."""
        result = _format_context_value({"key": "value"})
        assert result == '{"key":"value"}'

    def test_large_dict_truncated(self):
        """Large dicts are truncated."""
        large_dict = {f"key{i}": f"value{i}" for i in range(100)}
        result = _format_context_value(large_dict)
        assert len(result) <= MAX_CONTEXT_VALUE_LEN
        assert result.endswith("...")

    def test_list_serialized_as_json(self):
        """Lists are serialized as compact JSON."""
        result = _format_context_value([1, 2, 3])
        assert result == "[1,2,3]"

    def test_nested_structure_serialized(self):
        """Nested structures are serialized."""
        nested = {"outer": {"inner": [1, 2, 3]}}
        result = _format_context_value(nested)
        assert result == '{"outer":{"inner":[1,2,3]}}'

    def test_number_converted_to_string(self):
        """Numbers are converted to string."""
        assert _format_context_value(42) == "42"
        assert _format_context_value(3.14) == "3.14"

    def test_boolean_converted_to_string(self):
        """Booleans are converted to string."""
        assert _format_context_value(True) == "True"
        assert _format_context_value(False) == "False"

    def test_custom_max_len(self):
        """Custom max_len is respected."""
        result = _format_context_value("hello world", max_len=8)
        assert result == "hello..."
        assert len(result) == 8


class TestFormatContext:
    """Tests for _format_context."""

    def test_empty_context_returns_empty(self):
        """Empty context returns empty string."""
        assert _format_context({}) == ""
        assert _format_context(None) == ""

    def test_simple_context_formatted(self):
        """Simple context is formatted as bullet list."""
        context = {"name": "test", "count": 42}
        result = _format_context(context)
        assert "- name: test" in result
        assert "- count: 42" in result

    def test_values_truncated(self):
        """Long values are truncated."""
        context = {"blob": "x" * 1000}
        result = _format_context(context)
        assert len(result) < 1000
        assert "..." in result

    def test_total_length_capped(self):
        """Total context length is capped."""
        context = {f"key{i}": f"value{i}" * 50 for i in range(100)}
        result = _format_context(context)
        assert len(result) <= MAX_CONTEXT_TOTAL_LEN + 50  # Some slack for truncation notice
        assert "[additional context truncated]" in result

    def test_nested_values_handled(self):
        """Nested values are serialized."""
        context = {"config": {"nested": {"deep": True}}}
        result = _format_context(context)
        assert "config:" in result
        assert "nested" in result
