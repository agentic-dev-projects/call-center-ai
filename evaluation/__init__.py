"""
Evaluation Harness

Modules:
- metrics.py   — Token F1, ROUGE-L, BERTScore (offline, no LLM)
- ragas_eval.py — RAGAS wrapper (faithfulness, answer relevancy, context recall/precision)
- run_eval.py  — batch runner over eval_dataset.json
"""
