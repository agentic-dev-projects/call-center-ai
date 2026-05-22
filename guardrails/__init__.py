"""
Guardrails — Milestone 17

LEARNING: WHAT GUARDRAILS DO
══════════════════════════════

In a production AI pipeline, you have two trust boundaries:

  1. INPUT  — what comes INTO the pipeline from untrusted users
  2. OUTPUT — what leaves the pipeline and goes back to users / downstream systems

Guardrails sit at each boundary and act as a gate:

  [user input] → INPUT GUARDRAIL → [pipeline] → OUTPUT GUARDRAIL → [user]

Why both?

  Input guardrails prevent:
    - Wasted compute on garbage input (empty / gibberish calls)
    - PII being processed when it shouldn't be
    - Prompt injection: adversarial text that tries to hijack the LLM
      e.g. "Ignore all previous instructions and ..."

  Output guardrails prevent:
    - PII leaking into logs / dashboards
    - Hallucinated summaries being served with no grounding
    - Incomplete outputs that would mislead agents / customers

Architecture:
  BaseGuardrail (abstract)
    ├── EmptyInputGuardrail
    ├── PIIGuardrail
    ├── PromptInjectionGuardrail
    └── CompletenessGuardrail (output side)

  GuardrailRunner — runs a list of guardrails in sequence,
                    collects all violations, returns pass/fail + details
"""

from guardrails.base_guardrail import BaseGuardrail, GuardrailResult, GuardrailViolation
from guardrails.runner import GuardrailRunner, run_input_guardrails, run_output_guardrails

__all__ = [
    "BaseGuardrail",
    "GuardrailResult",
    "GuardrailViolation",
    "GuardrailRunner",
    "run_input_guardrails",
    "run_output_guardrails",
]
