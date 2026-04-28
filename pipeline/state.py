"""
Pipeline State Definition
"""

from typing import TypedDict
from agents.schemas import CallRecord


class PipelineState(TypedDict):
    record: CallRecord