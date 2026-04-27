"""
Schemas for Call Center AI

Defines structured data models used across agents.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import hashlib


class InputType(str, Enum):
    AUDIO = "audio"
    JSON_TRANSCRIPT = "json_transcript"


class CallStatus(str, Enum):
    PENDING = "pending"
    TRANSCRIBED = "transcribed"
    SUMMARIZED = "summarized"
    SCORED = "scored"
    FAILED = "failed"


class CallRecord(BaseModel):
    """
    Core data structure passed across all agents
    """

    call_id: str
    input_type: InputType
    status: CallStatus = CallStatus.PENDING

    # Input data
    audio_path: Optional[str] = None
    raw_transcript: Optional[str] = None

    # Metadata
    agent_name: Optional[str] = None
    customer_id: Optional[str] = None
    duration_seconds: Optional[float] = None

    # Output fields (filled later)
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    action_items: Optional[List[str]] = None
    qa_scores: Optional[dict] = None

    error: Optional[str] = None


def generate_call_id(input_data: str) -> str:
    """
    Generates deterministic call_id using SHA256
    """
    return hashlib.sha256(input_data.encode()).hexdigest()[:16]