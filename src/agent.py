"""A small ReAct-style ("Reason + Act") tool-using agent.

Given a user query, the agent loops:

    Thought  -> the LLMClient decides what to do next
    Action   -> if a tool was chosen, run it
    Observation -> the tool's result is recorded
    ... repeat, or ...
    Final Answer -> the LLMClient decides it's done and returns an answer

The "reasoning" (which tool to call, or whether to answer directly) is
delegated to an `LLMClient` (see `llm_client.py`). The default
`LocalStubLLMClient` is a deterministic, rule-based stand-in so the whole
agent runs with no API keys and no network calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.llm_client import LLMClient, LocalStubLLMClient
from src.tools import CalculatorTool, KnowledgeLookupTool, ToolError


@dataclass
class AgentStep:
    """One Thought/Action/Observation step recorded during the run."""

    thought: str
    tool_name: str | None = None
    tool_input: str | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    query: str
    final_answer: str
    steps: list[AgentStep] = field(default_factory=list)

    def transcript(self) -> str:
        """A human-readable rendering of the Thought/Action/Observation trace."""
        lines = [f"Query: {self.query}"]
        for i, step in enumerate(self.steps, start=1):
            lines.append(f"Thought {i}: {step.thought}")
            if step.tool_name:
                lines.append(f"Action {i}: {step.tool_name}[{step.tool_input}]")
                lines.append(f"Observation {i}: {step.observation}")
        lines.append(f"Final Answer: {self.final_answer}")
        return "\n".join(lines)


class ToolAgent:
    """A ReAct-style agent that plans and calls tools in a loop."""

    def __init__(self, llm_client: LLMClient | None = None, max_steps: int = 5):
        self.llm_client = llm_client or LocalStubLLMClient()
        self.max_steps = max_steps
        self.tools = {
            tool.name: tool
            for tool in (CalculatorTool(), KnowledgeLookupTool())
        }

    def run(self, query: str) -> AgentResult:
        history: list[str] = []
        steps: list[AgentStep] = []

        for _ in range(self.max_steps):
            decision = self.llm_client.decide(query, history)

            if decision.final_answer is not None:
                if steps:
                    # The final answer follows a tool observation; record the
                    # concluding thought against the last step for a clean
                    # transcript, then stop.
                    return AgentResult(query=query, final_answer=decision.final_answer, steps=steps)
                # No tools were called at all: answer directly.
                steps.append(AgentStep(thought="I can answer this directly without a tool."))
                return AgentResult(query=query, final_answer=decision.final_answer, steps=steps)

            tool_call = decision.tool_call
            assert tool_call is not None  # exactly one of the two is set
            tool = self.tools.get(tool_call.tool_name)

            if tool is None:
                final_answer = f"Error: unknown tool {tool_call.tool_name!r} requested."
                steps.append(AgentStep(thought=f"Wanted to call {tool_call.tool_name!r}, but it does not exist."))
                return AgentResult(query=query, final_answer=final_answer, steps=steps)

            thought = f"This looks like a job for the '{tool.name}' tool."
            try:
                observation = tool.run(tool_call.tool_input)
            except ToolError as exc:
                observation = f"Error: {exc}"

            steps.append(
                AgentStep(
                    thought=thought,
                    tool_name=tool.name,
                    tool_input=tool_call.tool_input,
                    observation=observation,
                )
            )
            history.append(observation)

        # Safety net: ran out of steps without a final answer.
        final_answer = "Could not produce a final answer within the step limit."
        return AgentResult(query=query, final_answer=final_answer, steps=steps)


def main() -> None:
    agent = ToolAgent()
    demo_queries = [
        "What is 12 * (4 + 3)?",
        "What is the capital of France?",
        "Tell me something interesting about yourself.",
    ]
    for query in demo_queries:
        result = agent.run(query)
        print(result.transcript())
        print("-" * 60)


if __name__ == "__main__":
    main()
