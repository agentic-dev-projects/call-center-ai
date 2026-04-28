"""
LangGraph Pipeline
"""

from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState

from agents.intake_agent import CallIntakeAgent
from agents.transcription_agent import TranscriptionAgent
from agents.summarization_agent import SummarizationAgent
from agents.qa_scoring_agent import QAScoringAgent


# Initialize agents
intake = CallIntakeAgent()
transcription = TranscriptionAgent()
summarization = SummarizationAgent()
qa = QAScoringAgent()


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


# ----------------------------
# BUILD GRAPH
# ----------------------------

def build_graph():

    graph = StateGraph(PipelineState)

    graph.add_node("intake", intake_node)
    graph.add_node("transcription", transcription_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("qa", qa_node)

    # Define flow
    graph.set_entry_point("intake")

    graph.add_edge("intake", "transcription")
    graph.add_edge("transcription", "summarization")
    graph.add_edge("summarization", "qa")

    graph.add_edge("qa", END)

    return graph.compile()