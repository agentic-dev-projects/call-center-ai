"""
Base Agent Class

Defines the standard interface for all agents in the system.

Every agent must:
- Implement `process()`
- Use `run()` as entry point
"""

import time
from abc import ABC, abstractmethod
from typing import Any
from utils.logger import logger


class BaseAgent(ABC):
    """
    Abstract Base Class for all agents
    """

    def __init__(self, name: str):
        self.name = name

    def run(self, input_data: Any) -> Any:
        """
        Standard execution wrapper for all agents.

        LEARNING: This is the single integration point for cross-cutting
        concerns — logging, error handling, and now AgentOps tracking.
        Every agent inherits this behaviour for free; none of them need
        to know AgentOps exists. This is the Open/Closed principle:
        open for extension (add tracking here), closed for modification
        (no changes needed in individual agents).
        """
        from ops.agentops_tracker import record_action

        logger.info(f"[{self.name}] Starting execution")
        start = time.time()

        record_action(self.name, "agent_start")

        try:
            result = self.process(input_data)
            elapsed = round(time.time() - start, 3)
            logger.info(f"[{self.name}] Completed successfully in {elapsed}s")
            record_action(self.name, "agent_end", {"elapsed_s": elapsed, "status": "success"})
            return result

        except Exception as e:
            elapsed = round(time.time() - start, 3)
            logger.error(f"[{self.name}] Failed: {str(e)}")
            record_action(self.name, "agent_end", {"elapsed_s": elapsed, "status": "error", "error": str(e)})
            return self.handle_error(e)

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass

    def handle_error(self, error: Exception, input_data=None):
        from agents.schemas import CallRecord, CallStatus

        if isinstance(input_data, CallRecord):
            input_data.status = CallStatus.FAILED
            input_data.error = str(error)
            return input_data

        return CallRecord(
            call_id="error",
            input_type="audio",
            status=CallStatus.FAILED,
            error=str(error)
        )