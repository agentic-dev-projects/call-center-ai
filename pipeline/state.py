"""
Pipeline State Definition
"""

from typing import TypedDict
from agents.schemas import CallRecord


class PipelineState(TypedDict, total=False):
    record: CallRecord
    next: str