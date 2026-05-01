"""
Tests for CallIntakeAgent.

No mocking needed — intake does pure validation, no API calls.
"""

import pytest
from agents.intake_agent import CallIntakeAgent
from agents.schemas import InputType, CallStatus


@pytest.fixture
def agent():
    return CallIntakeAgent()


# ── JSON input ────────────────────────────────────────────────────────────────

def test_json_intake_creates_record(agent):
    record = agent.run({"transcript": "Hello, how can I help?"})
    assert record.input_type == InputType.JSON_TRANSCRIPT
    assert record.raw_transcript == "Hello, how can I help?"
    assert record.status == CallStatus.PENDING


def test_json_intake_sets_metadata(agent):
    record = agent.run({
        "transcript": "Call transcript here.",
        "agent_name": "Alice",
        "customer_id": "CUS_001",
        "duration_seconds": 120.0,
    })
    assert record.agent_name == "Alice"
    assert record.customer_id == "CUS_001"
    assert record.duration_seconds == 120.0


def test_json_intake_generates_deterministic_call_id(agent):
    transcript = "Same transcript every time."
    r1 = agent.run({"transcript": transcript})
    r2 = agent.run({"transcript": transcript})
    assert r1.call_id == r2.call_id


def test_json_intake_missing_transcript_returns_failed(agent):
    record = agent.run({})
    assert record.status == CallStatus.FAILED


def test_json_intake_empty_transcript_returns_failed(agent):
    record = agent.run({"transcript": ""})
    assert record.status == CallStatus.FAILED


# ── Audio input ───────────────────────────────────────────────────────────────

def test_audio_intake_creates_record(agent):
    record = agent.run("data/sample_audio/sample.mp3")
    assert record.input_type == InputType.AUDIO
    assert record.audio_path == "data/sample_audio/sample.mp3"
    assert record.status == CallStatus.PENDING


def test_audio_intake_accepts_wav(agent):
    record = agent.run("call.wav")
    assert record.input_type == InputType.AUDIO


def test_audio_intake_accepts_m4a(agent):
    record = agent.run("call.m4a")
    assert record.input_type == InputType.AUDIO


def test_audio_intake_rejects_txt_returns_failed(agent):
    record = agent.run("transcript.txt")
    assert record.status == CallStatus.FAILED


def test_audio_intake_rejects_pdf_returns_failed(agent):
    record = agent.run("report.pdf")
    assert record.status == CallStatus.FAILED


# ── Unsupported input ─────────────────────────────────────────────────────────

def test_unsupported_input_type_returns_failed(agent):
    record = agent.run(12345)
    assert record.status == CallStatus.FAILED
