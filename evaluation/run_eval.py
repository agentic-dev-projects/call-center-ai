"""
Batch Evaluation Runner — Milestone 15

Loads eval_dataset.json and runs all evaluation metrics on each sample.

Evaluation pipeline per sample:
  1. Token F1     — lexical overlap (reference vs generated summary)
  2. ROUGE-L      — LCS-based overlap
  3. BERTScore    — semantic similarity via BERT embeddings
  4. RAGAS        — RAG-specific: faithfulness, answer_relevancy,
                    context_recall, context_precision (requires LLM calls)

Output:
  - Printed summary table
  - JSON results saved to data/eval_results.json

Usage:
    python -m evaluation.run_eval
    python -m evaluation.run_eval --skip-ragas   # offline only, no LLM calls
"""

import json
import argparse
from pathlib import Path

from evaluation.metrics import evaluate_summary
from evaluation.ragas_eval import run_ragas
from utils.logger import logger

EVAL_DATASET_PATH = Path(__file__).parent.parent / "data" / "eval_dataset.json"
EVAL_RESULTS_PATH = Path(__file__).parent.parent / "data" / "eval_results.json"


def _load_dataset() -> list:
    with open(EVAL_DATASET_PATH, "r") as f:
        return json.load(f)


def run_evaluation(skip_ragas: bool = False) -> list:
    """
    Evaluate all samples in eval_dataset.json.

    Args:
        skip_ragas: If True, skip RAGAS metrics (no LLM calls needed).

    Returns:
        List of per-sample result dicts.
    """
    dataset = _load_dataset()
    all_results = []

    for sample in dataset:
        sample_id = sample["id"]
        scenario  = sample["scenario"]
        logger.info(f"Evaluating sample {sample_id} ({scenario}) ...")

        reference_summary = sample["reference_summary"]
        generated_summary = sample["generated_summary"]

        # ------------------------------------------------------------------
        # 1–3. Offline metrics (Token F1, ROUGE-L, BERTScore)
        # ------------------------------------------------------------------
        offline_scores = evaluate_summary(reference_summary, generated_summary)

        # ------------------------------------------------------------------
        # 4. RAGAS (requires OpenAI calls)
        # ------------------------------------------------------------------
        ragas_scores = {}
        if not skip_ragas:
            try:
                ragas_scores = run_ragas(
                    question=f"Summarize this call about: {scenario}",
                    answer=generated_summary,
                    retrieved_contexts=sample["retrieved_contexts"],
                    reference_answer=reference_summary,
                )
            except Exception as e:
                logger.warning(f"RAGAS failed for {sample_id}: {e}")
                ragas_scores = {
                    "faithfulness":      None,
                    "answer_relevancy":  None,
                    "context_recall":    None,
                    "context_precision": None,
                    "error": str(e),
                }

        result = {
            "id":       sample_id,
            "scenario": scenario,
            "scores": {
                **offline_scores,
                "ragas": ragas_scores,
            },
            "expected_qa_scores": sample.get("expected_qa_scores", {}),
        }
        all_results.append(result)

        _print_sample_result(result)

    # Save all results
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {EVAL_RESULTS_PATH}")
    _print_aggregate_summary(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def _print_sample_result(result: dict) -> None:
    s   = result["scores"]
    tf1 = s["token_f1"]
    rl  = s["rouge_l"]
    bs  = s["bertscore"]
    rag = s.get("ragas", {})

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    print(f"\n{'='*60}")
    print(f"Sample : {result['id']}  |  {result['scenario']}")
    print(f"{'='*60}")
    print(f"  Token F1   — P:{_fmt(tf1['precision'])}  R:{_fmt(tf1['recall'])}  F1:{_fmt(tf1['f1'])}")
    print(f"  ROUGE-L    — P:{_fmt(rl['precision'])}   R:{_fmt(rl['recall'])}   F1:{_fmt(rl['f1'])}")
    print(f"  BERTScore  — P:{_fmt(bs['precision'])}  R:{_fmt(bs['recall'])}  F1:{_fmt(bs['f1'])}")

    if rag:
        if "error" in rag:
            print(f"  RAGAS      — ERROR: {rag['error']}")
        else:
            print(f"  RAGAS      — Faithfulness:{rag.get('faithfulness', 'N/A'):.3f}"
                  f"  AnswerRel:{rag.get('answer_relevancy', 'N/A'):.3f}"
                  f"  CtxRecall:{rag.get('context_recall', 'N/A'):.3f}"
                  f"  CtxPrec:{rag.get('context_precision', 'N/A'):.3f}")
    else:
        print(f"  RAGAS      — skipped (--skip-ragas)")


def _print_aggregate_summary(results: list) -> None:
    """Print average scores across all samples."""
    def avg(key_path):
        vals = []
        for r in results:
            s = r["scores"]
            keys = key_path.split(".")
            v = s
            for k in keys:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, float):
                vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else None

    print(f"\n{'='*60}")
    print("AGGREGATE AVERAGES")
    print(f"{'='*60}")
    print(f"  Token F1   F1: {avg('token_f1.f1')}")
    print(f"  ROUGE-L    F1: {avg('rouge_l.f1')}")
    print(f"  BERTScore  F1: {avg('bertscore.f1')}")
    print(f"  RAGAS faithfulness:     {avg('ragas.faithfulness')}")
    print(f"  RAGAS answer_relevancy: {avg('ragas.answer_relevancy')}")
    print(f"  RAGAS context_recall:   {avg('ragas.context_recall')}")
    print(f"  RAGAS context_precision:{avg('ragas.context_precision')}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip RAGAS metrics (no LLM calls)",
    )
    args = parser.parse_args()

    run_evaluation(skip_ragas=args.skip_ragas)
