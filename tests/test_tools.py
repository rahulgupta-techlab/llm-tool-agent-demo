"""Tests for tool execution correctness: calculator and knowledge lookup."""
import pytest

from src.tools import CalculatorTool, KnowledgeLookupTool, ToolError


class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    @pytest.mark.parametrize(
        "expression, expected",
        [
            ("2 + 2", "4"),
            ("2 + 3 * 4", "14"),
            ("(2 + 3) * 4", "20"),
            ("10 / 4", "2.5"),
            ("2 ** 8", "256"),
            ("-5 + 3", "-2"),
            ("10 % 3", "1"),
        ],
    )
    def test_correct_results(self, expression, expected):
        assert self.tool.run(expression) == expected

    def test_division_by_zero_is_rejected(self):
        with pytest.raises(ToolError):
            self.tool.run("1 / 0")

    @pytest.mark.parametrize(
        "unsafe_expression",
        [
            "__import__('os').system('echo hi')",
            "open('secret.txt').read()",
            "[x for x in range(10)]",
            "a + b",
            "1; 2",
        ],
    )
    def test_unsafe_input_is_rejected(self, unsafe_expression):
        with pytest.raises(ToolError):
            self.tool.run(unsafe_expression)


class TestKnowledgeLookupTool:
    def setup_method(self):
        self.tool = KnowledgeLookupTool()

    def test_returns_known_fact(self):
        result = self.tool.run("What is the capital of France?")
        assert "Paris" in result

    def test_returns_known_fact_case_insensitive(self):
        result = self.tool.run("what is the CAPITAL of japan")
        assert "Tokyo" in result

    def test_unknown_fact_raises_clear_error(self):
        with pytest.raises(ToolError):
            self.tool.run("What is the meaning of life?")
