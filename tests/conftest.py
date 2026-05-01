"""
Shared pytest fixtures used across all agent tests.
"""

import pytest
from agents.schemas import CallRecord, InputType, CallStatus


@pytest.fixture
def json_record():
    """A valid record that came in as a JSON transcript — no audio, ready to summarize."""
    return CallRecord(
        call_id="test_json_001",
        input_type=InputType.JSON_TRANSCRIPT,
        raw_transcript=(
            "Agent: Thank you for calling support, how can I help? "
            "Customer: I have a billing issue on my last invoice. "
            "Agent: I can look into that for you. "
            "Customer: The charge seems incorrect."
        ),
        status=CallStatus.PENDING,
    )


@pytest.fixture
def audio_record():
    """A valid record that came in as an audio file path — needs transcription first."""
    return CallRecord(
        call_id="test_audio_002",
        input_type=InputType.AUDIO,
        audio_path="data/sample_audio/sample.mp3",
        status=CallStatus.PENDING,
    )


@pytest.fixture
def summarized_record(json_record):
    """A record that has already been summarized — ready for QA scoring."""
    json_record.summary = "Customer called about an incorrect billing charge. Agent offered to investigate."
    json_record.key_points = ["Billing dispute", "Agent to investigate charge"]
    json_record.action_items = ["Review invoice #1234"]
    json_record.status = CallStatus.SUMMARIZED
    return json_record


@pytest.fixture
def scored_record(summarized_record):
    """A record with good QA scores — should route to end."""
    summarized_record.qa_scores = {
        "empathy": 4.0,
        "resolution": 4.0,
        "tone": 4.0,
        "professionalism": 5.0,
        "overall_score": 4.25,
    }
    summarized_record.status = CallStatus.SCORED
    return summarized_record


@pytest.fixture
def low_score_record(summarized_record):
    """A record with poor QA scores — should trigger escalation."""
    summarized_record.qa_scores = {
        "empathy": 1.0,
        "resolution": 2.0,
        "tone": 1.0,
        "professionalism": 2.0,
        "overall_score": 1.5,
    }
    summarized_record.status = CallStatus.SCORED
    return summarized_record
