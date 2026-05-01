"""
Tests for RoutingAgent.

Pure logic — no API calls, no mocking needed.
"""

import pytest
from agents.routing_agent import RoutingAgent
from agents.schemas import CallRecord, InputType, CallStatus


@pytest.fixture
def agent():
    return RoutingAgent()


# ── Routing from initial states ───────────────────────────────────────────────

def test_routes_to_transcription_when_no_transcript(agent, audio_record):
    assert agent.run(audio_record) == "transcription"


def test_routes_to_summarization_for_json_with_transcript(agent, json_record):
    # json_record already has raw_transcript set
    assert agent.run(json_record) == "summarization"


def test_routes_to_summarization_after_transcription(agent, audio_record):
    # Simulate TranscriptionAgent having run
    audio_record.raw_transcript = "Customer called about an outage."
    audio_record.status = CallStatus.TRANSCRIBED
    assert agent.run(audio_record) == "summarization"


# ── Routing after summarization ───────────────────────────────────────────────

def test_routes_to_qa_after_summarization(agent, summarized_record):
    assert agent.run(summarized_record) == "qa"


# ── Routing after QA scoring ──────────────────────────────────────────────────

def test_routes_to_end_for_good_score(agent, scored_record):
    assert agent.run(scored_record) == "end"


def test_routes_to_escalate_for_low_score(agent, low_score_record):
    assert agent.run(low_score_record) == "escalate"


def test_escalation_boundary_exactly_at_threshold(agent, summarized_record):
    # overall_score == ESCALATION_SCORE_THRESHOLD (3.0) should NOT escalate
    summarized_record.qa_scores = {"overall_score": 3.0}
    summarized_record.status = CallStatus.SCORED
    assert agent.run(summarized_record) == "end"


def test_escalation_just_below_threshold(agent, summarized_record):
    summarized_record.qa_scores = {"overall_score": 2.9}
    summarized_record.status = CallStatus.SCORED
    assert agent.run(summarized_record) == "escalate"


# ── Routing on failure ────────────────────────────────────────────────────────

def test_routes_to_end_on_failed_status(agent):
    record = CallRecord(
        call_id="fail_001",
        input_type=InputType.AUDIO,
        status=CallStatus.FAILED,
    )
    assert agent.run(record) == "end"
