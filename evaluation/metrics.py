"""
Offline evaluation metrics — no LLM calls needed.

Three metrics are implemented here:

1. Token F1
   - Treats each text as a bag of tokens (words)
   - Precision  = |predicted ∩ reference| / |predicted|
   - Recall     = |predicted ∩ reference| / |reference|
   - F1         = harmonic mean of Precision and Recall
   - Cheap to compute; good for checking lexical coverage

2. ROUGE-L
   - Based on Longest Common Subsequence (LCS) of tokens
   - Captures in-order word overlap even with gaps
   - Better than raw n-gram overlap for paraphrase-heavy summaries

3. BERTScore
   - Embeds reference and candidate tokens with BERT
   - Matches each candidate token to its closest reference token
   - Precision/Recall/F1 based on cosine similarity of embeddings
   - Handles synonyms and semantic paraphrasing; no exact match needed
"""

import re
import os
from collections import Counter
from typing import List

# Force HuggingFace libraries to use the local cache only.
# Without this, transformers + huggingface_hub attempt a version-check HTTP
# call on every load — blocked by the corporate proxy with a 403.
# The model must already be cached (run once on a non-proxied network first).
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

from utils.logger import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

def token_f1(reference: str, candidate: str) -> dict:
    """
    Bag-of-words token overlap between reference and candidate.

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    ref_tokens = Counter(_tokenize(reference))
    cand_tokens = Counter(_tokenize(candidate))

    # Intersection: take min count for each shared token
    common = sum((ref_tokens & cand_tokens).values())

    precision = common / sum(cand_tokens.values()) if cand_tokens else 0.0
    recall    = common / sum(ref_tokens.values())  if ref_tokens  else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def rouge_l(reference: str, candidate: str) -> dict:
    """
    ROUGE-L score using Longest Common Subsequence.

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    result = scorer.score(reference, candidate)["rougeL"]

    return {
        "precision": round(result.precision, 4),
        "recall":    round(result.recall,    4),
        "f1":        round(result.fmeasure,  4),
    }


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def bertscore(reference: str, candidate: str, lang: str = "en") -> dict:
    """
    BERTScore: semantic similarity via contextual BERT embeddings.

    Uses roberta-large by default (the bert_score library default for English).
    Model is cached in ~/.cache/huggingface/ after first download.
    Returns None values with an "error" key if the model cannot be fetched.

    Returns:
        {"precision": float, "recall": float, "f1": float}
        or {"precision": None, "recall": None, "f1": None, "error": str}
    """
    try:
        P, R, F = bert_score_fn(
            [candidate],
            [reference],
            lang=lang,
            verbose=False,
        )
        return {
            "precision": round(P[0].item(), 4),
            "recall":    round(R[0].item(), 4),
            "f1":        round(F[0].item(), 4),
        }
    except Exception as exc:
        logger.warning(f"BERTScore unavailable: {exc}")
        return {
            "precision": None,
            "recall":    None,
            "f1":        None,
            "error": "Model download required (HuggingFace unreachable — run outside proxy)",
        }


# ---------------------------------------------------------------------------
# Convenience: run all three on a single pair
# ---------------------------------------------------------------------------

def evaluate_summary(reference: str, candidate: str) -> dict:
    """
    Run Token F1, ROUGE-L, and BERTScore on a reference/candidate pair.

    Returns a dict with keys: token_f1, rouge_l, bertscore
    """
    logger.info("Running Token F1 ...")
    tf1 = token_f1(reference, candidate)

    logger.info("Running ROUGE-L ...")
    rl = rouge_l(reference, candidate)

    logger.info("Running BERTScore ...")
    bs = bertscore(reference, candidate)

    return {
        "token_f1":  tf1,
        "rouge_l":   rl,
        "bertscore": bs,
    }
