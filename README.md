# LLM Tool Agent Demo

A small ReAct-style ("Reason + Act") agent: given a query, it decides whether
to call a tool, executes the tool, observes the result, and produces a final
answer - Thought -> Action -> Observation -> ... -> Final Answer.

Built to demonstrate the core tool-using-agent pattern end-to-end without
external services - everything runs locally with a deterministic, rule-based
stand-in for the LLM. The decision-making step is pluggable: swap in a real
OpenAI or Anthropic client by implementing `LLMClient` in `src/llm_client.py`
(see "Swapping in a real LLM" below).

## Stack

- Python 3.11+
- Standard library only for the agent itself (`ast` for safe expression
  evaluation, `re` for the rule-based router)
- `pytest` for tests

## Project layout

```
src/
  llm_client.py   # LLMClient interface + LocalStubLLMClient (rule-based, no API keys)
  tools.py         # calculator tool (safe ast-based eval) + knowledge lookup tool
  agent.py         # the ReAct-style Thought/Action/Observation loop
tests/
  test_tools.py    # tool selection is NOT here; tool execution correctness is
  test_agent.py    # tool selection + end-to-end agent loop tests
```

## Running it

```bash
pip install -r requirements.txt
python -m src.agent
```

This runs the built-in demo (`main()` in `src/agent.py`), which sends three
example queries through the agent and prints the full Thought/Action/
Observation/Final Answer transcript for each, e.g.:

```
Query: What is 12 * (4 + 3)?
Thought 1: This looks like a job for the 'calculator' tool.
Action 1: calculator[12 * (4 + 3)]
Observation 1: 84
Final Answer: 84
------------------------------------------------------------
```

You can also use it directly in Python:

```python
from src.agent import ToolAgent

agent = ToolAgent()
result = agent.run("What is the capital of France?")
print(result.final_answer)   # "The capital of France is Paris."
print(result.transcript())   # full Thought/Action/Observation trace
```

## Tests

```bash
pytest
```

Covers: tool-selection routing (which tool a query should trigger), tool
execution correctness (calculator arithmetic and safe-input rejection,
knowledge lookup hits/misses), and the end-to-end agent loop for a few
example queries. Everything runs against the default `LocalStubLLMClient`,
so no API key or network access is needed for the suite to pass.

## How it works

- **`LLMClient`** (`src/llm_client.py`) is the interface between the agent
  loop and whatever is doing the "thinking". It exposes one method,
  `decide(query, history) -> Decision`, where a `Decision` is either a
  `ToolCall` (call this tool with this input) or a `final_answer` (stop and
  return this).
- **`LocalStubLLMClient`** is the default, zero-dependency implementation: it
  uses simple regex/keyword matching to decide whether a query looks like an
  arithmetic expression (-> `calculator`), mentions a known fact topic (->
  `knowledge_lookup`), or should just be answered directly. It's
  deterministic on purpose, so tests are reproducible.
- **`tools.py`** has two tools:
  - `CalculatorTool` parses the expression with `ast.parse` and walks the
    resulting tree itself, only permitting numeric constants and a small
    allow-list of arithmetic operators. It never calls `eval`/`exec`, so it
    can't run arbitrary code, unlike a naive `eval(expression)` approach.
  - `KnowledgeLookupTool` is an explicitly **mock/local** stand-in for a real
    search or vector-lookup tool: a hard-coded in-memory dict of a handful of
    facts, matched by keyword overlap. In a production system this would be
    a call to a search API or a vector database over a real corpus.
- **`ToolAgent`** (`src/agent.py`) runs the loop: ask the `LLMClient` what to
  do, run the chosen tool if any, record the Thought/Action/Observation step,
  and repeat (bounded by `max_steps`) until the `LLMClient` returns a final
  answer.

## Swapping in a real LLM

To use a real model instead of the rule-based stub, implement the
`LLMClient` protocol (one method: `decide(query, history) -> Decision`) with
a class that calls out to an LLM and parses its response into a `ToolCall`
or a final answer. Sketches for OpenAI and Anthropic (commented out at the
bottom of `src/llm_client.py`, not installed as dependencies):

```python
import openai  # pip install openai

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def decide(self, query: str, history: list[str]) -> Decision:
        prompt = build_react_prompt(query, history)  # your own prompt template
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_decision(response.choices[0].message.content)  # your own parser


import anthropic  # pip install anthropic

class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def decide(self, query: str, history: list[str]) -> Decision:
        prompt = build_react_prompt(query, history)
        response = self.client.messages.create(
            model=self.model, max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return parse_decision(response.content[0].text)
```

Then pass an instance in: `ToolAgent(llm_client=OpenAIClient(api_key="..."))`.
The tool layer, agent loop, and tests are unaffected - only `decide()`
changes.

## Notes

This is a portfolio/demo project - the tool set and rule-based stub LLM are
intentionally minimal so the ReAct mechanics (planning, tool execution,
observation, looping to a final answer) are easy to follow and extend.
