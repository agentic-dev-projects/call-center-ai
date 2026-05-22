"""
LangGraph Pipeline
"""
from utils.logger import logger
from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState

from agents.intake_agent import CallIntakeAgent
from agents.transcription_agent import TranscriptionAgent
from agents.summarization_agent import SummarizationAgent
from agents.qa_scoring_agent import QAScoringAgent
from agents.routing_agent import RoutingAgent
from agents.tool_agent import ToolAgent

from guardrails.runner import run_input_guardrails, run_output_guardrails
from agents.schemas import CallStatus
from db.sqlite_store import save_call

from ops.tracing import setup_langsmith
from ops.agentops_tracker import setup_agentops, AgentOpsSession

# Initialise observability once at module load time.
# LEARNING: Module-level setup runs exactly once per process (Python's import
# cache prevents re-execution). This is the right place for SDK init calls
# that must happen before any LangChain/LangGraph objects are created.
setup_langsmith()
setup_agentops()

# Initialize agents
intake = CallIntakeAgent()
transcription = TranscriptionAgent()
summarization = SummarizationAgent()
qa = QAScoringAgent()
router = RoutingAgent()
tool_agent = ToolAgent()


# ----------------------------
# NODE FUNCTIONS
# ----------------------------

def input_guard_node(state: PipelineState):
    """
    Gate-keeper: runs input guardrails before any agent touches the transcript.

    LEARNING: By placing this as the graph entry point, we guarantee that
    no compute is wasted on invalid/malicious input. If the guardrail blocks,
    we set status=FAILED and route to END — the LLM is never called.
    """
    record = state["record"]
    transcript = record.raw_transcript or ""

    result = run_input_guardrails(transcript)
    violations = [v.message for v in result.violations]

    if violations:
        record.guardrail_violations = violations

    if getattr(result, "blocked", False):
        record.guardrail_blocked = True
        record.status = CallStatus.FAILED
        record.error = "Blocked by input guardrail: " + "; ".join(
            v.message for v in result.violations if v.severity == "high"
        )
        logger.warning(f"Input guardrail BLOCKED call {record.call_id}: {record.error}")

    return {"record": record}


def output_guard_node(state: PipelineState):
    """
    Final check: runs output guardrails after all agents have completed.

    LEARNING: Output guardrails are advisory — we never discard generated
    output, but we annotate the record so operators can review flagged calls.
    """
    record = state["record"]
    result = run_output_guardrails(record)

    if not result.passed:
        existing = record.guardrail_violations or []
        new_violations = [v.message for v in result.violations]
        record.guardrail_violations = existing + new_violations
        logger.info(f"Output guardrails flagged {len(new_violations)} issue(s) for call {record.call_id}")

    return {"record": record}


def persist_node(state: PipelineState):
    """
    Persist the completed CallRecord to SQLite.

    LEARNING: Separating persistence into its own graph node means:
      - Every exit path through the graph hits this node
      - The pipeline logic (agents) stays decoupled from storage
      - You can swap SQLite → Postgres by changing only sqlite_store.py

    This node runs last, after output_guard, so the full record
    (including guardrail violations) is saved in one write.
    """
    record = state["record"]
    try:
        save_call(record)
    except Exception as exc:
        logger.warning(f"SQLite: failed to save call {record.call_id} — {exc}")
    return {"record": record}


def intake_node(state: PipelineState):
    record = intake.run(state["record"])
    return {"record": record}


def transcription_node(state: PipelineState):
    record = transcription.run(state["record"])
    return {"record": record}


def summarization_node(state: PipelineState):
    record = summarization.run(state["record"])
    return {"record": record}


def qa_node(state: PipelineState):
    record = qa.run(state["record"])
    return {"record": record}

def escalate_node(state):
    record = state["record"]
    record.error = "Low QA score - escalation required"
    return {"record": record}

def route_decision(state):
    return state["next"]

def tool_node(state):
    record = tool_agent.run(state["record"])
    return {"record": record}


# ----------------------------
# BUILD GRAPH
# ----------------------------

def build_graph():

    graph = StateGraph(PipelineState)

    graph.add_node("intake", intake_node)
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("transcription", transcription_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("qa", qa_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("tool", tool_node)
    graph.add_node("output_guard", output_guard_node)
    graph.add_node("persist", persist_node)

    def router_node(state: PipelineState):
        next_step = router.run(state["record"])
        logger.info(f"Routing decision: {next_step}")
        return {"next": next_step}

    graph.add_node("router", router_node)

    # LEARNING: Guardrails sandwich the agent pipeline.
    #
    # Entry point is intake (creates the CallRecord from raw input).
    # After intake, input_guard inspects raw_transcript — at this point
    # the CallRecord exists with the transcript populated, but no LLM
    # has been called yet. Blocking here = zero wasted compute.
    graph.set_entry_point("intake")
    graph.add_edge("intake", "input_guard")

    # If blocked, save to DB then END; otherwise continue to routing
    def after_input_guard(state: PipelineState):
        return "end" if state["record"].guardrail_blocked else "continue"

    graph.add_conditional_edges(
        "input_guard",
        after_input_guard,
        {"end": "persist", "continue": "router"},   # blocked calls still persisted
    )

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "transcription": "transcription",
            "summarization": "summarization",
            "qa": "qa",
            "escalate": "escalate",
            "tool": "tool",
            "end": "output_guard",
        }
    )

    graph.add_edge("transcription", "router")
    graph.add_edge("summarization", "router")
    graph.add_edge("qa", "router")
    graph.add_edge("qa", "tool")
    graph.add_edge("tool", "router")
    graph.add_edge("escalate", "output_guard")
    graph.add_edge("output_guard", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


def run_pipeline_with_tracking(input_data) -> dict:
    """
    Run the pipeline wrapped in an AgentOps session.

    LEARNING: This is a thin wrapper that adds session-level tracking
    around the existing graph.invoke() call. The pipeline itself doesn't
    change — observability is layered on top, not baked in.

    The AgentOpsSession context manager guarantees end_session() is called
    even if an exception propagates out of graph.invoke().
    """
    graph = build_graph()
    call_id = getattr(input_data, "call_id", "unknown") if not isinstance(input_data, dict) else "unknown"

    with AgentOpsSession(call_id=call_id):
        state = {"record": input_data}
        final_state = graph.invoke(state)
        return final_state