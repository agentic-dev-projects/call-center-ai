"""
Output guardrails — validate what leaves the pipeline.

Two checks:

  1. PIILeakageGuardrail  — detect PII that leaked into the generated summary
  2. CompletenessGuardrail — ensure the summary has the expected fields

LEARNING: WHY OUTPUT GUARDRAILS?
══════════════════════════════════

Input guardrails flag PII in the raw transcript — but the LLM might still
reproduce it verbatim in the summary or action items. Output guardrails
catch that second exposure point.

Completeness checks protect downstream consumers: if the CRM system
expects action_items and gets None, it may silently drop the record or
crash. Checking here gives a clear error message rather than a mystery
NoneType exception later.
"""

import re
from typing import List, Optional

from guardrails.base_guardrail import BaseGuardrail, GuardrailResult, GuardrailViolation
from agents.schemas import CallRecord
from utils.logger import logger


# Re-use the same PII patterns from input side
_PII_PATTERNS = {
    "PII_PHONE": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "PII_EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "PII_SSN":   r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "PII_CREDIT_CARD": r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13}|6(?:011|5\d{2})\d{12})\b",
}


class PIILeakageGuardrail(BaseGuardrail):
    """
    Checks whether the generated summary / key_points / action_items
    contain PII that shouldn't appear in the processed output.

    LEARNING: The LLM often copies verbatim from the transcript. If a
    customer said "my card number is 4111 1111 1111 1111", there's a real
    chance that appears in the summary too. This guardrail catches it before
    the record is stored or displayed.

    Severity: high — leaking PII in stored outputs is a compliance breach.
    """

    @property
    def name(self) -> str:
        return "pii_output"

    def _check_text(self, label: str, text: Optional[str]) -> List[GuardrailViolation]:
        if not text:
            return []
        violations = []
        for code, pattern in _PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                logger.warning(f"PIILeakageGuardrail: {code} in {label} — {len(matches)} instance(s)")
                violations.append(GuardrailViolation(
                    guardrail=self.name,
                    code=f"{code}_IN_{label.upper()}",
                    message=f"{code} leaked into {label} ({len(matches)} instance(s)).",
                    severity="high",
                ))
        return violations

    def check_record(self, record: CallRecord) -> GuardrailResult:
        """Check the full CallRecord rather than a plain string."""
        violations: List[GuardrailViolation] = []

        violations += self._check_text("summary", record.summary)

        for item in (record.key_points or []):
            violations += self._check_text("key_points", item)

        for item in (record.action_items or []):
            violations += self._check_text("action_items", item)

        if violations:
            return GuardrailResult.fail(violations)
        return GuardrailResult.ok()

    def check(self, text: str) -> GuardrailResult:
        """Plain-text check (satisfies BaseGuardrail interface)."""
        violations: List[GuardrailViolation] = []
        for code, pattern in _PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append(GuardrailViolation(
                    guardrail=self.name,
                    code=code,
                    message=f"{code} found in output ({len(matches)} instance(s)).",
                    severity="high",
                ))
        if violations:
            return GuardrailResult.fail(violations)
        return GuardrailResult.ok()


class CompletenessGuardrail(BaseGuardrail):
    """
    Ensures the pipeline produced a usable output.

    LEARNING: LLMs occasionally return empty strings or null for structured
    fields, especially when the transcript is very short or low quality.
    This guardrail catches incomplete outputs before they propagate to the
    UI or storage layer, so operators know to review/retry rather than
    silently serving a broken record.

    Severity: medium — the pipeline ran but output is degraded.
    """

    @property
    def name(self) -> str:
        return "completeness"

    def check(self, text: str) -> GuardrailResult:
        """Not used directly — use check_record instead."""
        return GuardrailResult.ok()

    def check_record(self, record: CallRecord) -> GuardrailResult:
        violations: List[GuardrailViolation] = []

        if not record.summary or len(record.summary.strip()) < 20:
            violations.append(GuardrailViolation(
                guardrail=self.name,
                code="MISSING_SUMMARY",
                message="Summary is missing or too short (< 20 chars).",
                severity="medium",
            ))

        if not record.key_points:
            violations.append(GuardrailViolation(
                guardrail=self.name,
                code="MISSING_KEY_POINTS",
                message="key_points list is empty or None.",
                severity="medium",
            ))

        if not record.action_items:
            violations.append(GuardrailViolation(
                guardrail=self.name,
                code="MISSING_ACTION_ITEMS",
                message="action_items list is empty or None.",
                severity="low",
            ))

        if violations:
            return GuardrailResult.fail(violations)
        return GuardrailResult.ok()
