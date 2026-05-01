"""
QA Scoring Agent

Evaluates call quality using LLM function calling
"""

from agents.base_agent import BaseAgent
from agents.schemas import CallRecord, CallStatus
from config.settings import settings
from openai import OpenAI


class QAScoringAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="QAScoringAgent")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def process(self, record: CallRecord) -> CallRecord:

        if not record.raw_transcript:
            raise ValueError("Transcript missing")

        # Function schema (forces structured output)
        function_schema = {
            "name": "score_call",
            "description": "Score the quality of a customer support call",
            "parameters": {
                "type": "object",
                "properties": {
                    "empathy": {"type": "number"},
                    "resolution": {"type": "number"},
                    "tone": {"type": "number"},
                    "professionalism": {"type": "number"},
                    "overall_score": {"type": "number"}
                },
                "required": ["empathy", "resolution", "tone", "professionalism", "overall_score"]
            }
        }

        # Prompt
        prompt = f"""
                    Evaluate the following customer support call transcript:

                    {record.raw_transcript}

                    Score each category from 1 to 5:
                    - empathy
                    - resolution
                    - tone
                    - professionalism

                    Also compute overall_score as average.

                    Return ONLY structured JSON.
                    """

        response = self.client.chat.completions.create(
            model=settings.QA_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict QA evaluator."},
                {"role": "user", "content": prompt}
            ],
            functions=[function_schema],
            function_call={"name": "score_call"},
            temperature=0
        )

        result = response.choices[0].message.function_call.arguments

        # Convert string → dict
        import json
        parsed = json.loads(result)

        record.qa_scores = parsed
        record.status = CallStatus.SCORED

        return record