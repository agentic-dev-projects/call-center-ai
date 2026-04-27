"""
Base Agent Class

Defines the standard interface for all agents in the system.

Every agent must:
- Implement `process()`
- Use `run()` as entry point
"""

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
        Standard execution wrapper for all agents

        Flow:
        1. Log start
        2. Call process()
        3. Handle errors
        4. Log success/failure
        """

        logger.info(f"[{self.name}] Starting execution")

        try:
            result = self.process(input_data)
            logger.info(f"[{self.name}] Completed successfully")
            return result

        except Exception as e:
            logger.error(f"[{self.name}] Failed: {str(e)}")
            return self.handle_error(e)

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Each agent implements this method
        """
        pass

    def handle_error(self, error: Exception):
        """
        Default error handling logic
        """
        return {
            "status": "FAILED",
            "error": str(error),
            "agent": self.name
        }