"""
Routing Agent

Decides next step based on pipeline state
"""

from agents.base_agent import BaseAgent
from agents.schemas import CallRecord


class RoutingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="RoutingAgent")

    def process(self, record: CallRecord) -> str:

        if not hasattr(record, "status"):
            return "end"
        # ❗ If failure
        if record.status == "failed":
            return "end"

        # ✅ NEW: If transcript missing → transcribe
        if not record.raw_transcript:
            return "transcription"

        # Skip transcription if already present
        if record.raw_transcript and record.input_type == "json_transcript":
            return "summarization"

        # After QA → decide escalation
        if record.qa_scores:
            score = record.qa_scores.get("overall_score", 5)

            if score < 3:
                return "escalate"
            else:
                return "end"

        # If transcript exists but no summary
        if record.raw_transcript and not record.summary:
            return "summarization"

        # If summary exists but no QA
        if record.summary and not record.qa_scores:
            return "qa"

        return "end"