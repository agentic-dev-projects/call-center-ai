"""
Tool Agent

Uses LLM to decide which tool to call
"""

from agents.base_agent import BaseAgent
from agents.schemas import CallRecord
from config.settings import settings
from openai import OpenAI

from tools.tool_registry import TOOLS


class ToolAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ToolAgent")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def process(self, record: CallRecord) -> CallRecord:

        if not record.summary:
            return record

        prompt = f"""
Given the following call summary:

{record.summary}

Decide if a tool should be used.

Available tools:
- check_outage(area)

Return JSON:
{{
  "tool": "<tool_name or none>",
  "input": "<input if needed>"
}}
"""

        response = self.client.chat.completions.create(
            model=settings.TOOL_MODEL,
            messages=[
                {"role": "system", "content": "You decide tool usage."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        import json
        decision = json.loads(response.choices[0].message.content)

        tool_name = decision.get("tool")

        if tool_name in TOOLS:
            tool_input = decision.get("input", "")
            result = TOOLS[tool_name](tool_input)

            record.action_items.append(f"Tool Result: {result}")

        return record