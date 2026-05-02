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
    graph.add_node("transcription", transcription_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("qa", qa_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("tool", tool_node)

    def router_node(state: PipelineState):
        next_step = router.run(state["record"])
        logger.info(f"Routing decision: {next_step}")
        return {"next": next_step}

    graph.add_node("router", router_node)

    graph.set_entry_point("intake")

    graph.add_edge("intake", "router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "transcription": "transcription",
            "summarization": "summarization",
            "qa": "qa",
            "escalate": "escalate",
            "tool": "tool",
            "end": END,
        }
    )

    graph.add_edge("transcription", "router")
    graph.add_edge("summarization", "router")
    graph.add_edge("qa", "router")
    graph.add_edge("qa", "tool")
    graph.add_edge("tool", "router")
    graph.add_edge("escalate", END)

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