"""
Pipeline Orchestrator
"""

from pipeline.graph import build_graph


def run_pipeline(input_data):

    graph = build_graph()

    state = {"record": input_data}

    result = graph.invoke(state)

    return result["record"]