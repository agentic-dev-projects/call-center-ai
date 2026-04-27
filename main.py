from agents.intake_agent import CallIntakeAgent
from agents.transcription_agent import TranscriptionAgent

if __name__ == "__main__":

    intake = CallIntakeAgent()
    transcription = TranscriptionAgent()

    # Test audio input
    input_audio = "data/sample_audio/sample.mp3"

    record = intake.run(input_audio)
    record = transcription.run(record)

    if isinstance(record, dict):
        print(record)
    else:
        print(record.model_dump())