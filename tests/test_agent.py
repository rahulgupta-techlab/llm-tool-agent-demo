"""Tests for tool-selection routing and the end-to-end ReAct agent loop.

All tests use the default `LocalStubLLMClient` (rule-based, offline), so
no API key or network access is required.
"""
from src.agent import ToolAgent
from src.llm_client import LocalStubLLMClient


class TestToolSelection:
    """The stub LLM should route each query to the right tool (or none)."""

    def setup_method(self):
        self.llm = LocalStubLLMClient()

    def test_math_query_selects_calculator(self):
        decision = self.llm.decide("What is 12 * (4 + 3)?", history=[])
        assert decision.tool_call is not None
        assert decision.tool_call.tool_name == "calculator"

    def test_bare_expression_selects_calculator(self):
        decision = self.llm.decide("2 + 2", history=[])
        assert decision.tool_call is not None
        assert decision.tool_call.tool_name == "calculator"

    def test_fact_query_selects_knowledge_lookup(self):
        decision = self.llm.decide("What is the capital of France?", history=[])
        assert decision.tool_call is not None
        assert decision.tool_call.tool_name == "knowledge_lookup"

    def test_chit_chat_query_answers_directly(self):
        decision = self.llm.decide("Tell me a joke.", history=[])
        assert decision.tool_call is None
        assert decision.final_answer is not None

    def test_history_present_yields_final_answer(self):
        decision = self.llm.decide("What is 2 + 2?", history=["4"])
        assert decision.tool_call is None
        assert decision.final_answer == "4"


class TestToolAgentEndToEnd:
    def setup_method(self):
        self.agent = ToolAgent()

    def test_math_query_produces_correct_answer(self):
        result = self.agent.run("What is 12 * (4 + 3)?")
        assert result.final_answer == "84"
        assert len(result.steps) == 1
        assert result.steps[0].tool_name == "calculator"

    def test_fact_query_produces_correct_answer(self):
        result = self.agent.run("What is the capital of France?")
        assert "Paris" in result.final_answer
        assert result.steps[0].tool_name == "knowledge_lookup"

    def test_direct_query_answers_without_tool(self):
        result = self.agent.run("Tell me something interesting about yourself.")
        assert result.steps[0].tool_name is None
        assert "interesting about yourself" in result.final_answer

    def test_unsafe_math_expression_is_handled_gracefully(self):
        # A crafted query that looks like a fact question but resolves to an
        # unknown fact should surface a clear "not found" style message
        # instead of raising out of the agent loop.
        result = self.agent.run("What is the largest unicorn population?")
        assert "Error" in result.final_answer or "No known fact" in result.final_answer

    def test_transcript_includes_thought_action_observation(self):
        result = self.agent.run("What is 3 + 4?")
        transcript = result.transcript()
        assert "Thought" in transcript
        assert "Action" in transcript
        assert "Observation" in transcript
        assert "Final Answer" in transcript
