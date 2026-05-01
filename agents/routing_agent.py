"""
Routing Agent

Decides next step based on pipeline state
"""

from agents.base_agent import BaseAgent
from agents.schemas import CallRecord, CallStatus
from config.settings import settings


class RoutingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="RoutingAgent")

    def process(self, record: CallRecord) -> str:

        if record.status == CallStatus.FAILED:
            return "end"

        # Stage 1: no transcript yet → transcribe
        if not record.raw_transcript:
            return "transcription"

        # Stage 2: transcript exists, not yet summarized → summarize
        if not record.summary:
            return "summarization"

        # Stage 3: summarized, not yet scored → score
        if not record.qa_scores:
            return "qa"

        # Stage 4: scored → escalate if below threshold, otherwise done
        score = record.qa_scores.get("overall_score", 5)
        if score < settings.ESCALATION_SCORE_THRESHOLD:
            return "escalate"

        return "end"
