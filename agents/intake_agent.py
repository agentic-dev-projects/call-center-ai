"""
Call Intake Agent

Responsibilities:
- Detect input type (audio vs JSON)
- Validate input
- Create initial CallRecord
"""

from agents.base_agent import BaseAgent
from agents.schemas import CallRecord, InputType, generate_call_id


class IntakeValidationError(Exception):
    pass


class CallIntakeAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="CallIntakeAgent")

    def process(self, input_data):
        """
        Entry point for all pipeline inputs
        """

        # Case 1: JSON input
        if isinstance(input_data, dict):
            return self._handle_json(input_data)

        # Case 2: Audio file path
        elif isinstance(input_data, str):
            return self._handle_audio(input_data)

        else:
            raise IntakeValidationError("Unsupported input type")

    # ----------------------------
    # JSON HANDLING
    # ----------------------------
    def _handle_json(self, data: dict) -> CallRecord:
        if "transcript" not in data or not data["transcript"]:
            raise IntakeValidationError("Transcript missing or empty")

        call_id = generate_call_id(data["transcript"])

        return CallRecord(
            call_id=call_id,
            input_type=InputType.JSON_TRANSCRIPT,
            raw_transcript=data["transcript"],
            agent_name=data.get("agent_name"),
            customer_id=data.get("customer_id"),
            duration_seconds=data.get("duration_seconds"),
        )

    # ----------------------------
    # AUDIO HANDLING
    # ----------------------------
    def _handle_audio(self, file_path: str) -> CallRecord:
        if not file_path.endswith((".mp3", ".wav", ".m4a")):
            raise IntakeValidationError("Unsupported audio format")

        call_id = generate_call_id(file_path)

        return CallRecord(
            call_id=call_id,
            input_type=InputType.AUDIO,
            audio_path=file_path,
        )