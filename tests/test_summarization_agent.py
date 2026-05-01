"""
Tests for SummarizationAgent.

LLM calls and ChromaDB are mocked — we test the agent's control flow,
not the correctness of external services.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from agents.summarization_agent import SummarizationAgent
from agents.schemas import CallStatus


LLM_RESPONSE = {
    "summary": "Customer called about an incorrect billing charge.",
    "key_points": ["Billing dispute", "Agent to investigate"],
    "action_items": ["Review invoice"],
}


def _mock_completion(content: dict) -> MagicMock:
    """Build a fake OpenAI chat completion response."""
    mock = MagicMock()
    mock.choices[0].message.content = json.dumps(content)
    return mock


# ── Cache hit ─────────────────────────────────────────────────────────────────

def test_cache_hit_skips_llm(json_record):
    with patch("agents.summarization_agent.get_from_cache", return_value=LLM_RESPONSE), \
         patch("agents.summarization_agent.store_in_cache") as mock_store:

        agent = SummarizationAgent()
        result = agent.run(json_record)

    assert result.summary == LLM_RESPONSE["summary"]
    assert result.key_points == LLM_RESPONSE["key_points"]
    assert result.status == CallStatus.SUMMARIZED
    mock_store.assert_not_called()  # nothing new to cache


# ── Cache miss → LLM call ─────────────────────────────────────────────────────

def test_cache_miss_calls_llm_and_stores_result(json_record):
    with patch("agents.summarization_agent.get_from_cache", return_value=None), \
         patch("agents.summarization_agent.store_in_cache") as mock_store, \
         patch("agents.summarization_agent.store_chunks"), \
         patch("agents.summarization_agent.retrieve", return_value=["chunk1", "chunk2"]):

        agent = SummarizationAgent()
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _mock_completion(LLM_RESPONSE)

        result = agent.process(json_record)

    assert result.summary == LLM_RESPONSE["summary"]
    assert result.status == CallStatus.SUMMARIZED
    mock_store.assert_called_once()  # result was cached for next time


def test_cache_miss_with_empty_retrieval_uses_full_transcript(json_record):
    """When RAG returns nothing, the agent should fall back to the full transcript."""
    with patch("agents.summarization_agent.get_from_cache", return_value=None), \
         patch("agents.summarization_agent.store_in_cache"), \
         patch("agents.summarization_agent.store_chunks"), \
         patch("agents.summarization_agent.retrieve", return_value=[]):

        agent = SummarizationAgent()
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _mock_completion(LLM_RESPONSE)

        result = agent.process(json_record)

    assert result.status == CallStatus.SUMMARIZED


# ── Error handling ────────────────────────────────────────────────────────────

def test_missing_transcript_returns_failed():
    from agents.schemas import CallRecord, InputType
    record = CallRecord(call_id="x", input_type=InputType.JSON_TRANSCRIPT)

    agent = SummarizationAgent()
    result = agent.run(record)

    assert result.status == CallStatus.FAILED


def test_llm_returns_invalid_json_returns_failed(json_record):
    with patch("agents.summarization_agent.get_from_cache", return_value=None), \
         patch("agents.summarization_agent.store_chunks"), \
         patch("agents.summarization_agent.retrieve", return_value=["chunk"]):

        agent = SummarizationAgent()
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="not valid json {{"))]
        )

        result = agent.run(json_record)

    assert result.status == CallStatus.FAILED
