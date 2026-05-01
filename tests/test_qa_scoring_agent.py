"""
Tests for QAScoringAgent.

OpenAI function-calling response is mocked — we test score parsing and
status transitions, not the LLM's scoring judgment.
"""

import json
import pytest
from unittest.mock import MagicMock

from agents.qa_scoring_agent import QAScoringAgent
from agents.schemas import CallRecord, InputType, CallStatus


MOCK_SCORES = {
    "empathy": 4.0,
    "resolution": 3.0,
    "tone": 4.0,
    "professionalism": 5.0,
    "overall_score": 4.0,
}


def _make_agent_with_mock_scores(scores: dict) -> QAScoringAgent:
    """Return a QAScoringAgent whose OpenAI client returns the given scores."""
    agent = QAScoringAgent()
    mock_fn_call = MagicMock()
    mock_fn_call.arguments = json.dumps(scores)
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(function_call=mock_fn_call))]
    )
    return agent


# ── Happy path ────────────────────────────────────────────────────────────────

def test_qa_scoring_sets_scores(json_record):
    agent = _make_agent_with_mock_scores(MOCK_SCORES)
    result = agent.process(json_record)

    assert result.qa_scores == MOCK_SCORES
    assert result.status == CallStatus.SCORED


def test_qa_scoring_overall_score_present(json_record):
    agent = _make_agent_with_mock_scores(MOCK_SCORES)
    result = agent.process(json_record)

    assert "overall_score" in result.qa_scores
    assert result.qa_scores["overall_score"] == 4.0


def test_qa_scoring_all_dimensions_present(json_record):
    agent = _make_agent_with_mock_scores(MOCK_SCORES)
    result = agent.process(json_record)

    for dim in ("empathy", "resolution", "tone", "professionalism"):
        assert dim in result.qa_scores


# ── Low scores ────────────────────────────────────────────────────────────────

def test_qa_scoring_accepts_low_scores(json_record):
    low_scores = {**MOCK_SCORES, "overall_score": 1.5, "empathy": 1.0}
    agent = _make_agent_with_mock_scores(low_scores)
    result = agent.process(json_record)

    assert result.qa_scores["overall_score"] == 1.5
    assert result.status == CallStatus.SCORED  # scoring itself always succeeds


# ── Error handling ────────────────────────────────────────────────────────────

def test_missing_transcript_returns_failed():
    record = CallRecord(call_id="x", input_type=InputType.JSON_TRANSCRIPT)
    agent = QAScoringAgent()
    result = agent.run(record)

    assert result.status == CallStatus.FAILED
    assert result.error is not None
