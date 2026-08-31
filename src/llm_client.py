"""The "brain" behind the agent's tool-routing decisions.

The decision-making call is behind a small `LLMClient` protocol so a real
provider (OpenAI, Anthropic, a local model, etc.) can be dropped in without
touching the agent loop. See the README for a sketch of what an
`OpenAIClient`/`AnthropicClient` implementation would look like.

The default, dependency-free implementation is `LocalStubLLMClient`: a
deterministic, rule-based ("regex/keyword matching") stand-in for a real
LLM. It looks at the query and decides:

- "this looks like a math expression" -> call the calculator tool
- "this looks like a fact lookup" -> call the knowledge lookup tool
- otherwise -> answer directly, no tool call

Because it's pure Python with no network calls or API keys, the whole demo
(including tests) runs completely offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ToolCall:
    """A decision to call a tool with a given input string."""

    tool_name: str
    tool_input: str


@dataclass
class Decision:
    """The result of asking the LLM what to do next.

    Exactly one of `tool_call` or `final_answer` is set:
    - `tool_call` set -> the agent should invoke that tool and continue the loop.
    - `final_answer` set -> the agent should stop and return this as the answer.
    """

    tool_call: ToolCall | None = None
    final_answer: str | None = None


class LLMClient(Protocol):
    def decide(self, query: str, history: list[str]) -> Decision:
        """Given the original query and a transcript of prior
        Thought/Action/Observation lines, decide the next step."""
        ...


# A query "looks like" a math expression if, after stripping a leading
# question phrase like "what is", it consists only of numbers, whitespace,
# and arithmetic operators/parentheses.
_MATH_EXPR_RE = re.compile(r"^[\s0-9+\-*/().%]+$")
_LEADING_PHRASES_RE = re.compile(
    r"^(what\s+is|what's|calculate|compute|evaluate)\s+", re.IGNORECASE
)

# Trigger words that suggest a factual lookup rather than direct chat.
_FACT_TRIGGERS = ("capital", "largest", "author", "wrote", "created", "creator",
                   "tallest", "mountain", "speed of light", "planet")


def _extract_math_expression(query: str) -> str | None:
    candidate = _LEADING_PHRASES_RE.sub("", query.strip())
    candidate = candidate.rstrip("?.! ")
    if candidate and _MATH_EXPR_RE.match(candidate) and any(c.isdigit() for c in candidate):
        return candidate
    return None


class LocalStubLLMClient:
    """Deterministic, rule-based local stub. Requires no API keys and makes
    no network calls, so the agent (and its tests) run entirely offline.

    This intentionally plays the role an LLM would play in a real ReAct
    agent: given the query (and so-far transcript), decide whether to call
    a tool, and if so, which one and with what input. Here that "reasoning"
    is simple keyword/regex matching instead of a model call.
    """

    def decide(self, query: str, history: list[str]) -> Decision:
        # Once a tool has already produced an observation, treat that as
        # the final answer instead of looping forever.
        if history:
            last_observation = history[-1]
            return Decision(final_answer=last_observation)

        math_expr = _extract_math_expression(query)
        if math_expr is not None:
            return Decision(tool_call=ToolCall(tool_name="calculator", tool_input=math_expr))

        lowered = query.lower()
        if any(trigger in lowered for trigger in _FACT_TRIGGERS):
            return Decision(tool_call=ToolCall(tool_name="knowledge_lookup", tool_input=query))

        return Decision(final_answer=f"I don't have a tool for that, but here's my best answer: {query}")


# ---------------------------------------------------------------------------
# Sketch of how a real LLM-backed client would plug in (NOT wired up, no
# extra dependencies required). See README.md "Swapping in a real LLM" for
# the full explanation.
# ---------------------------------------------------------------------------
#
# import json
# import openai  # pip install openai; not a dependency of this project
#
# class OpenAIClient:
#     """Example real LLMClient implementation using the OpenAI API."""
#
#     def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
#         self.client = openai.OpenAI(api_key=api_key)
#         self.model = model
#
#     def decide(self, query: str, history: list[str]) -> Decision:
#         prompt = build_react_prompt(query, history)  # your own prompt template
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         return parse_decision(response.choices[0].message.content)  # your own parser
#
#
# import anthropic  # pip install anthropic; not a dependency of this project
#
# class AnthropicClient:
#     """Example real LLMClient implementation using the Anthropic API."""
#
#     def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
#         self.client = anthropic.Anthropic(api_key=api_key)
#         self.model = model
#
#     def decide(self, query: str, history: list[str]) -> Decision:
#         prompt = build_react_prompt(query, history)
#         response = self.client.messages.create(
#             model=self.model,
#             max_tokens=512,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         return parse_decision(response.content[0].text)
