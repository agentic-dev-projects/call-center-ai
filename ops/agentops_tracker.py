"""
AgentOps Tracking — Milestone 16

LEARNING: HOW AGENTOPS WORKS
═════════════════════════════

AgentOps is an LLMOps platform focused on agent-level observability.
Where LangSmith traces the LangGraph DAG, AgentOps tracks each individual
agent as a "session" with its own metrics.

Key concepts:

1. SESSION
   A session = one top-level pipeline run (one call processed).
   AgentOps records: start time, end time, total cost, total tokens, errors.

2. LLM EVENT
   Every OpenAI/LLM call inside a session is captured as an LLM event:
   - model name, prompt, completion
   - token counts → cost estimate
   - latency

3. ACTION EVENT
   Non-LLM steps (cache lookups, DB reads, tool calls) logged as actions.
   Lets you see what percentage of time is LLM vs infrastructure.

4. ERROR EVENT
   Any exception inside a session is captured with full stack trace.

How our integration works:
  ┌──────────────────────────────────────────────────────┐
  │  pipeline.invoke()                                    │
  │    agentops.start_session()  ← session begins        │
  │                                                       │
  │    IntakeAgent.run()                                  │
  │      record_action("intake", ...)                     │
  │                                                       │
  │    SummarizationAgent.run()                           │
  │      record_action("summarization_start", ...)        │
  │      [OpenAI call — auto-captured by agentops]        │
  │      record_action("summarization_end", ...)          │
  │                                                       │
  │    agentops.end_session("Success")  ← session ends   │
  └──────────────────────────────────────────────────────┘

LangSmith vs AgentOps — when to use which:
  LangSmith  → debugging LangChain/LangGraph flow, prompt inspection
  AgentOps   → production monitoring, cost tracking, error rate dashboards
"""

from utils.logger import logger
from config.settings import settings

_agentops_enabled = False


def setup_agentops() -> bool:
    """
    Initialise AgentOps SDK (0.3.x API).

    Call once at application startup (before any pipeline runs).
    Returns True if initialised, False if skipped (no API key).

    LEARNING: AgentOps 0.4.x rewrote its internals to use OpenTelemetry
    and introduced a 'NonRecordingSpan' bug in many environments. We pin
    to 0.3.x which has a simpler, stable API:
      - instrument_llm_calls=True  → auto-patches OpenAI client
      - auto_start_session=False   → we manage session lifecycle manually
    """
    global _agentops_enabled

    if not settings.AGENTOPS_API_KEY:
        logger.info("AgentOps: no AGENTOPS_API_KEY set — tracking disabled")
        return False

    try:
        import agentops
        agentops.init(
            api_key=settings.AGENTOPS_API_KEY,
            default_tags=["call-center-ai", settings.ENV],
            instrument_llm_calls=True,
            auto_start_session=True,   # start one session per app process
        )
        _agentops_enabled = True
        logger.info("AgentOps: initialised successfully")
        return True

    except ImportError:
        logger.warning("AgentOps: package not installed — run: pip install agentops==0.3.21")
        return False
    except Exception as exc:
        logger.warning(f"AgentOps: init failed — {exc}")
        return False


def is_enabled() -> bool:
    return _agentops_enabled


class AgentOpsSession:
    """
    Context manager for a single pipeline run session.

    LEARNING: Context managers (with statement) guarantee cleanup even
    when exceptions occur. We use this pattern so end_session() is always
    called — whether the pipeline succeeds or fails.

    Usage:
        with AgentOpsSession(call_id="abc123") as session:
            run_pipeline(...)
    """

    def __init__(self, call_id: str = "unknown"):
        self.call_id = call_id
        self._session = None

    def __enter__(self):
        if _agentops_enabled:
            try:
                import agentops
                self._session = agentops.start_session(
                    tags=["call_id:" + self.call_id]
                )
                logger.info(f"AgentOps: session started for call {self.call_id}")
            except Exception as exc:
                logger.warning(f"AgentOps: failed to start session — {exc}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _agentops_enabled and self._session:
            try:
                import agentops
                outcome = "Fail" if exc_type else "Success"
                agentops.end_session(outcome)
                logger.info(f"AgentOps: session ended — {outcome}")
            except Exception as exc:
                logger.warning(f"AgentOps: failed to end session — {exc}")
        return False   # don't suppress exceptions


def record_action(agent_name: str, action: str, metadata: dict = None) -> None:
    """
    Log a non-LLM agent action to the current AgentOps session.

    LEARNING: Not every interesting thing in your pipeline is an LLM call.
    Cache hits, DB lookups, tool invocations — these are "actions".
    Tracking them alongside LLM events gives you the full cost picture:
      Total latency = LLM latency + infra latency
    """
    if not _agentops_enabled:
        return

    try:
        import agentops
        agentops.record(agentops.ActionEvent(
            action_type=action,
            params={"agent": agent_name, **(metadata or {})},
        ))
    except Exception as exc:
        logger.warning(f"AgentOps: record_action failed — {exc}")
