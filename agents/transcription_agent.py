"""
Transcription Agent

Responsibilities:
- Convert audio → text using Whisper API
- Update CallRecord
"""

from agents.base_agent import BaseAgent
from agents.schemas import CallRecord, CallStatus, InputType
from utils.audio_preprocessor import preprocess_audio
from config.settings import settings
from openai import OpenAI


class TranscriptionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="TranscriptionAgent")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def process(self, record: CallRecord) -> CallRecord:

        # Skip if already transcript
        if record.input_type == InputType.JSON_TRANSCRIPT:
            return record

        if not record.audio_path:
            raise ValueError("Audio path missing")

        # Step 1: preprocess audio
        processed_path = "temp_processed.wav"
        preprocess_audio(record.audio_path, processed_path)

        # Step 2: call Whisper API
        with open(processed_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        # Step 3: update record
        record.raw_transcript = response.text
        record.status = CallStatus.TRANSCRIBED

        return record