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
        """
        Returns next step name (node key)
        """

        # ❗ Case 1: Failure anywhere
        if record.status == "failed":
            return "end"

        # ❗ Case 2: Skip transcription if transcript already exists
        if record.raw_transcript and record.input_type == "json_transcript":
            return "summarization"

        # ❗ Case 3: After QA, decide escalation
        if record.qa_scores:
            score = record.qa_scores.get("overall_score", 5)

            if score < 3:
                return "escalate"
            else:
                return "end"

        # ❗ Default flow
        if record.raw_transcript and not record.summary:
            return "summarization"

        if record.summary and not record.qa_scores:
            return "qa"

        return "end"