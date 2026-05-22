"""
Base class for all guardrails.

LEARNING: ABSTRACT BASE CLASSES AS CONTRACTS
══════════════════════════════════════════════

An abstract base class (ABC) defines an interface — a contract every
subclass must honour. Here:
  - Every guardrail must implement `check(text) -> GuardrailResult`
  - The base class can't be instantiated directly (enforced by ABC)
  - This lets GuardrailRunner treat all guardrails identically

Compare to duck-typing: ABC enforces the contract at class-definition
time rather than at call time, so you catch missing implementations
immediately when the class is defined.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class GuardrailViolation:
    """
    A single rule violation found by a guardrail.

    LEARNING: Dataclasses are a clean alternative to plain dicts for
    structured data. They give you __repr__, __eq__, and type hints
    for free, with zero boilerplate.
    """
    guardrail: str          # which guardrail caught this
    code: str               # machine-readable code, e.g. "PII_PHONE"
    message: str            # human-readable description
    severity: str = "high"  # "low" | "medium" | "high"


@dataclass
class GuardrailResult:
    """
    Aggregate result from one guardrail's check.

    passed=True  → text is clean, pipeline may continue
    passed=False → violations found; caller decides whether to block or warn
    """
    passed: bool
    violations: List[GuardrailViolation] = field(default_factory=list)

    @classmethod
    def ok(cls) -> "GuardrailResult":
        return cls(passed=True)

    @classmethod
    def fail(cls, violations: List[GuardrailViolation]) -> "GuardrailResult":
        return cls(passed=False, violations=violations)


class BaseGuardrail(ABC):
    """
    Abstract base for all guardrails.

    LEARNING: OPEN/CLOSED PRINCIPLE
    Adding a new guardrail = subclass + add to GuardrailRunner's list.
    Existing code never changes. Same pattern used in BaseAgent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'pii_input'."""

    @abstractmethod
    def check(self, text: str) -> GuardrailResult:
        """
        Inspect text and return a GuardrailResult.
        Must be side-effect free — never modifies state.
        """
