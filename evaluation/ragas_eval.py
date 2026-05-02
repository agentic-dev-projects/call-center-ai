"""
RAGAS evaluation wrapper — RAG-specific metrics via LLM judges.

RAGAS (Retrieval Augmented Generation Assessment) measures four dimensions:

1. Faithfulness
   - Does the generated answer contain ONLY claims that are supported by the
     retrieved contexts?
   - LLM decomposes the answer into atomic claims, then checks each against
     the contexts.
   - Score: fraction of claims supported. 1.0 = fully grounded.

2. Answer Relevancy
   - Is the answer directly relevant to the question?
   - LLM reverse-engineers hypothetical questions from the answer, then
     measures cosine similarity to the original question.
   - Score: average similarity. 1.0 = perfectly on-topic.

3. Context Recall
   - Does the retrieved context cover everything in the ground-truth answer?
   - LLM checks how many sentences in the reference can be attributed to the
     retrieved contexts.
   - Score: fraction attributed. 1.0 = full coverage.

4. Context Precision
   - Are the retrieved chunks actually relevant (no noise in the retrieval)?
   - LLM scores each chunk: did it help produce the reference answer?
   - Score: precision@k averaged across ranks. 1.0 = all chunks useful.

IMPORTANT: All four metrics call the LLM under the hood.
"""

from typing import List

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import settings
from utils.logger import logger


def run_ragas(
    question: str,
    answer: str,
    retrieved_contexts: List[str],
    reference_answer: str,
) -> dict:
    """
    Evaluate a single RAG response with all four RAGAS metrics.

    Args:
        question:           The user's original question / call summary request
        answer:             The LLM-generated answer / summary
        retrieved_contexts: List of RAG chunks fed to the LLM
        reference_answer:   Human-annotated ground-truth answer

    Returns:
        Dict with keys: faithfulness, answer_relevancy, context_recall,
                        context_precision  (each a float 0–1)
    """
    logger.info(f"Running RAGAS on question: {question[:60]}...")

    # RAGAS 0.4 uses SingleTurnSample + EvaluationDataset
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts,
        reference=reference_answer,
    )
    dataset = EvaluationDataset(samples=[sample])

    # Provide LLM + embeddings explicitly (uses our already-configured key)
    llm = ChatOpenAI(
        model=settings.QA_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(api_key=settings.OPENAI_API_KEY)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm,
        embeddings=embeddings,
    )

    # result.scores is a list of dicts, one per sample
    scores = result.scores[0] if result.scores else {}

    return {
        "faithfulness":       round(float(scores.get("faithfulness",       0) or 0), 4),
        "answer_relevancy":   round(float(scores.get("answer_relevancy",   0) or 0), 4),
        "context_recall":     round(float(scores.get("context_recall",     0) or 0), 4),
        "context_precision":  round(float(scores.get("context_precision",  0) or 0), 4),
    }
