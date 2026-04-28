"""
Summarization Agent

Responsibilities:
- Generate structured insights from transcript
"""

import json
from agents.base_agent import BaseAgent
from agents.schemas import CallRecord, CallStatus
from config.settings import settings
from openai import OpenAI


class SummarizationAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SummarizationAgent")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def process(self, record: CallRecord) -> CallRecord:

        if not record.raw_transcript:
            raise ValueError("Transcript missing")

        # Load prompt
        with open("config/prompts/summarization_v1.txt", "r") as f:
            prompt_template = f.read()

        prompt = prompt_template.format(transcript=record.raw_transcript)

        # Call LLM
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON-only response generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}  
        )

        content = response.choices[0].message.content

        # Try parsing JSON output
        try:
            parsed = json.loads(content)
        except Exception:
            raise ValueError("Failed to parse LLM output as JSON")

        # Update record
        record.summary = parsed.get("summary")
        record.key_points = parsed.get("key_points")
        record.action_items = parsed.get("action_items")

        record.status = CallStatus.SUMMARIZED

        return record