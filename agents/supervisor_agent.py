"""
Supervisor Agent — Milestone 21 (A2A)

LEARNING: WHAT IS A2A?
════════════════════════

A2A (Agent-to-Agent) means one AI agent communicates with and uses
another agent's capabilities through a defined interface.

In this project:

  PIPELINE AGENTS (M1–M9)            ← process individual calls
      ↓  store results in SQLite (M19)
  CALL CENTER DATA                   ← structured, queryable
      ↑  exposed as MCP tools (M20)
  SUPERVISOR AGENT (M21)             ← queries the pipeline's output
      ↑  natural language interface
  OPERATOR / USER

The SupervisorAgent doesn't know HOW calls are processed — it only
knows the tool interface (search, stats, details). This is A2A:
agents communicating through defined interfaces, not shared state.

─────────────────────────────────────────────────────────────────────
LEARNING: OPENAI FUNCTION CALLING (TOOL USE)
─────────────────────────────────────────────────────────────────────

OpenAI's function calling lets you define "tools" the LLM can choose
to call. The flow is a loop:

  1. Send messages + tool definitions to the LLM
  2. LLM responds with EITHER:
       a. A final text answer  → loop ends
       b. A tool_call request  → we execute the tool, add result, loop again

This is called the ReAct pattern:
  Reason → Act (call tool) → Observe (tool result) → Reason again → ...

  User: "Find all escalated calls this week"
  LLM:  [calls search_call_history("escalated")]
  Tool: [returns 3 matching calls]
  LLM:  [calls get_call_details("abc123")]
  Tool: [returns full details]
  LLM:  "Here are the 3 escalated calls: ..."

The LLM decides WHICH tool to call and WHEN to stop — we just execute
whatever it requests and feed the result back.

─────────────────────────────────────────────────────────────────────
LEARNING: TOOL DEFINITIONS (JSON SCHEMA)
─────────────────────────────────────────────────────────────────────

Each tool is described with a JSON Schema so the LLM knows:
  - What the tool does (description)
  - What arguments it takes (parameters + types)
  - Which arguments are required

The LLM uses these descriptions to decide which tool fits the query.
Good descriptions = better tool selection.
"""

import json
from typing import Any

from openai import OpenAI

from agents.base_agent import BaseAgent
from config.settings import settings
from utils.logger import logger

# Import the same functions the MCP server exposes — single source of truth
from mcp_server.server import (
    search_call_history,
    get_call_details,
    get_call_stats,
    check_outage,
)

# ── Tool definitions (JSON Schema) ────────────────────────────────────────────
# LEARNING: These mirror the MCP tool definitions exactly.
# The LLM reads these descriptions to decide which tool to call.
# Precise descriptions matter — vague descriptions lead to wrong tool choices.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_call_history",
            "description": (
                "Search stored call records by keyword. "
                "Searches across summaries and transcripts. "
                "Use for: finding calls about a topic, locating escalations, "
                "discovering billing complaints, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Word or phrase to search for (e.g. 'billing', 'outage', 'refund')",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_call_details",
            "description": (
                "Get full details for a specific call — summary, key points, "
                "action items, QA scores, and guardrail violations. "
                "Use when you have a call_id and need the complete record."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "call_id": {
                        "type": "string",
                        "description": "The call ID or its first 12 characters",
                    }
                },
                "required": ["call_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_call_stats",
            "description": (
                "Return aggregate statistics across all stored calls: "
                "total count, average QA score per dimension, cache hit rate, "
                "number of blocked/failed calls, guardrail warning count. "
                "Use for overview questions like 'how is the call center performing?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_outage",
            "description": (
                "Check whether there is a known service outage in a given area. "
                "Use when a customer or query mentions connectivity issues "
                "or service disruptions in a specific region."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "Geographic area to check (e.g. 'california', 'new york')",
                    }
                },
                "required": ["area"],
            },
        },
    },
]

# ── Tool dispatcher ────────────────────────────────────────────────────────────
# LEARNING: When the LLM requests a tool call, we receive the tool name and
# arguments as strings. This dispatcher maps name → function and executes it.

_TOOL_MAP = {
    "search_call_history": search_call_history,
    "get_call_details":    get_call_details,
    "get_call_stats":      get_call_stats,
    "check_outage":        check_outage,
}


def _execute_tool(name: str, arguments: str) -> str:
    """
    Execute a tool by name with JSON-encoded arguments.

    LEARNING: The LLM returns arguments as a JSON string.
    We parse it, call the function with **kwargs, and return
    the result as a string to feed back into the conversation.
    """
    func = _TOOL_MAP.get(name)
    if not func:
        return f"Error: unknown tool '{name}'"

    try:
        args = json.loads(arguments) if arguments else {}
        result = func(**args)
        return str(result)
    except Exception as exc:
        logger.warning(f"SupervisorAgent: tool '{name}' failed — {exc}")
        return f"Error executing {name}: {exc}"


# ── Supervisor Agent ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a call center supervisor AI assistant.

You have access to tools that let you query the call center's database:
- Search call history by topic or keyword
- Get detailed records for specific calls
- Retrieve aggregate performance statistics
- Check for known service outages in an area

Always use tools to get real data before answering. Be concise and specific.
Format your final answer clearly with sections if multiple topics are covered.
If asked about a specific call, always retrieve its full details."""


class SupervisorAgent(BaseAgent):
    """
    A2A Supervisor: an LLM agent that queries the call center pipeline
    via tool calling.

    LEARNING: THIS IS THE A2A PATTERN
    The pipeline agents (IntakeAgent, SummarizationAgent, etc.) produce data.
    The SupervisorAgent CONSUMES that data through a defined interface (tools).
    Neither side knows the other's internal implementation.

    The SupervisorAgent uses a tool-calling loop:
      1. Send query + available tools to LLM
      2. LLM decides which tool to call
      3. We execute the tool, append result to conversation
      4. Repeat until LLM produces a final text answer

    MAX_ITERATIONS prevents infinite loops if the LLM keeps calling tools.
    """

    MAX_ITERATIONS = 5

    def __init__(self):
        super().__init__(name="SupervisorAgent")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def handle_error(self, error: Exception, input_data=None) -> str:
        """Return a plain string on failure — SupervisorAgent never returns a CallRecord."""
        logger.error(f"SupervisorAgent error: {error}")
        return f"I encountered an error while analysing your request: {error}"

    def process(self, input_data: Any) -> str:
        """
        Answer a natural language query about the call center.

        Args:
            input_data: str — the supervisor query
                        (e.g. "How many calls were escalated this week?")

        Returns:
            str — the supervisor's natural language answer
        """
        if not isinstance(input_data, str) or not input_data.strip():
            return "Please provide a question about the call center."

        query = input_data.strip()
        logger.info(f"SupervisorAgent: query = '{query[:80]}'")

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": query},
        ]

        # ── ReAct tool-calling loop ────────────────────────────────────────
        for iteration in range(self.MAX_ITERATIONS):
            logger.debug(f"SupervisorAgent: iteration {iteration + 1}")

            response = self.client.chat.completions.create(
                model=settings.QA_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",   # LLM decides whether to call a tool
                temperature=0.2,
            )

            message = response.choices[0].message

            # ── Case 1: LLM wants to call a tool ──────────────────────────
            if message.tool_calls:
                # Add the assistant's tool-call request to conversation history
                messages.append(message)

                # Execute each requested tool and append results
                for tool_call in message.tool_calls:
                    name      = tool_call.function.name
                    arguments = tool_call.function.arguments

                    logger.info(f"SupervisorAgent: calling tool '{name}' with {arguments}")
                    result = _execute_tool(name, arguments)
                    logger.debug(f"SupervisorAgent: tool result = {result[:100]}...")

                    # LEARNING: Tool results go back as role="tool" messages.
                    # The tool_call_id links this result to the specific request.
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      result,
                    })

            # ── Case 2: LLM has enough info — final answer ─────────────────
            else:
                answer = message.content or "No answer generated."
                logger.info(f"SupervisorAgent: final answer in {iteration + 1} iteration(s)")
                return answer

        # Fallback if loop exhausted
        return "I was unable to complete the analysis within the allowed steps. Please try a more specific question."
