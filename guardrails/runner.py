"""
GuardrailRunner — orchestrates multiple guardrails in sequence.

LEARNING: THE RUNNER PATTERN
══════════════════════════════

Instead of the pipeline calling each guardrail individually, a Runner:
  1. Holds an ordered list of guardrails
  2. Runs each one against the same input
  3. Collects ALL violations (not just the first)
  4. Returns a single aggregate result

This means one failed guardrail doesn't shadow another's violations —
the caller gets the full picture.

BLOCKING vs WARN-ONLY
══════════════════════
Not all violations should stop the pipeline. We use severity to decide:
  - "high"   → block (default for injection, credit card leakage)
  - "medium" → warn + continue (PII in transcript, short summary)
  - "low"    → log only (missing action_items)

The runner exposes `should_block()` so the caller decides policy.
"""

from typing import List

from guardrails.base_guardrail import BaseGuardrail, GuardrailResult, GuardrailViolation
from guardrails.input_guardrails import (
    EmptyInputGuardrail,
    PIIGuardrail,
    PromptInjectionGuardrail,
)
from guardrails.output_guardrails import PIILeakageGuardrail, CompletenessGuardrail
from agents.schemas import CallRecord
from utils.logger import logger


class GuardrailRunner:
    """
    Runs a list of guardrails and aggregates results.

    LEARNING: Dependency injection — guardrails are passed in, not hardcoded.
    This makes testing easy (inject mocks) and lets you configure different
    sets of guardrails for different environments (prod vs staging).
    """

    def __init__(self, guardrails: List[BaseGuardrail], block_on: str = "high"):
        """
        Args:
            guardrails: ordered list of guardrails to run
            block_on:   minimum severity that triggers a block
                        "high" → only block on high
                        "medium" → block on medium and high
                        "low" → block on everything
        """
        self._guardrails = guardrails
        self._severity_rank = {"low": 0, "medium": 1, "high": 2}
        self._block_threshold = self._severity_rank.get(block_on, 2)

    def run(self, text: str) -> GuardrailResult:
        """Run all guardrails against a plain text string."""
        all_violations: List[GuardrailViolation] = []

        for guard in self._guardrails:
            result = guard.check(text)
            if not result.passed:
                logger.debug(f"GuardrailRunner: {guard.name} — {len(result.violations)} violation(s)")
                all_violations.extend(result.violations)

        if not all_violations:
            return GuardrailResult.ok()
        return GuardrailResult.fail(all_violations)

    def run_on_record(self, record: CallRecord) -> GuardrailResult:
        """Run record-aware guardrails (output side)."""
        all_violations: List[GuardrailViolation] = []

        for guard in self._guardrails:
            if hasattr(guard, "check_record"):
                result = guard.check_record(record)
            else:
                # Fallback: check summary text
                result = guard.check(record.summary or "")
            if not result.passed:
                all_violations.extend(result.violations)

        if not all_violations:
            return GuardrailResult.ok()
        return GuardrailResult.fail(all_violations)

    def should_block(self, result: GuardrailResult) -> bool:
        """Return True if any violation meets or exceeds the block threshold."""
        if result.passed:
            return False
        for v in result.violations:
            if self._severity_rank.get(v.severity, 0) >= self._block_threshold:
                return True
        return False


# ---------------------------------------------------------------------------
# Convenience singletons — pre-configured runners for common use cases
# ---------------------------------------------------------------------------

def run_input_guardrails(transcript: str) -> GuardrailResult:
    """
    Run all input guardrails against a raw transcript.

    Returns GuardrailResult with all violations.
    Blocks on: EMPTY_INPUT, PROMPT_INJECTION (high severity).
    Warns on:  PII, TOO_SHORT, GIBBERISH (medium severity).

    LEARNING: This function is the single call site the pipeline uses.
    Adding a new input guardrail = add it to the list here. The pipeline
    code never changes (Open/Closed Principle).
    """
    runner = GuardrailRunner(
        guardrails=[
            EmptyInputGuardrail(),
            PromptInjectionGuardrail(),  # check injection before PII to fail fast
            PIIGuardrail(),
        ],
        block_on="high",
    )
    result = runner.run(transcript)

    if not result.passed:
        high = [v for v in result.violations if v.severity == "high"]
        medium = [v for v in result.violations if v.severity == "medium"]
        logger.info(
            f"Input guardrails: {len(high)} blocking, {len(medium)} warnings "
            f"— block={runner.should_block(result)}"
        )

    # Attach should_block decision to result for convenience
    result.blocked = runner.should_block(result)  # type: ignore[attr-defined]
    return result


def run_output_guardrails(record: CallRecord) -> GuardrailResult:
    """
    Run all output guardrails against a completed CallRecord.

    Returns GuardrailResult with all violations.
    Never blocks (output is already generated) — violations are warnings
    that get stored on the record for operator review.

    LEARNING: Output guardrails are mostly advisory. We can't un-generate
    the LLM output, but we can flag it for human review and prevent it
    from being stored or displayed without a warning.
    """
    runner = GuardrailRunner(
        guardrails=[
            PIILeakageGuardrail(),
            CompletenessGuardrail(),
        ],
        block_on="high",
    )
    result = runner.run_on_record(record)

    if not result.passed:
        logger.warning(
            f"Output guardrails: {len(result.violations)} violation(s) for call {record.call_id}"
        )

    result.blocked = False  # type: ignore[attr-defined]  # output guardrails never block
    return result
