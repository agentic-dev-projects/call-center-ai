"""
BM25 Sparse Retriever — keyword-based search without a vector DB.

LEARNING: WHAT IS BM25?
════════════════════════

BM25 (Best Match 25) is the gold-standard sparse retrieval algorithm.
It's what Google used before neural search, and it's still used today
as a first-stage retriever or hybrid component.

It improves on raw TF-IDF in two ways:

  1. TERM SATURATION (k1 parameter)
     Raw TF: seeing a word 10× scores 10× higher than seeing it once.
     BM25:   the score saturates — the 10th occurrence adds far less than
             the 1st. Real documents don't get better just by repeating.

         score = TF * (k1 + 1) / (TF + k1)

  2. LENGTH NORMALISATION (b parameter)
     A long document naturally has more term occurrences than a short one —
     but that doesn't mean it's more relevant.
     BM25 penalises long documents by comparing against the average length:

         score = TF * (k1 + 1) / (TF + k1 * (1 - b + b * |D| / avgdl))

  IDF component (same idea as TF-IDF):
     Rare words that appear in few documents are more informative.

         IDF = log((N - df + 0.5) / (df + 0.5) + 1)

  Full BM25 score for query Q against document D:
     sum over each query term t of: IDF(t) * TF_normalised(t, D)

WHEN BM25 WINS OVER DENSE VECTORS:
  - Exact keyword matches: product codes, names, error codes
  - Short queries with rare terms
  - Zero-shot (no fine-tuning needed)

WHEN DENSE WINS OVER BM25:
  - Paraphrasing and synonyms
  - Semantic similarity without shared keywords

That's why HYBRID = the best of both.
"""

import math
import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from utils.logger import logger


def _tokenize(text: str) -> List[str]:
    """
    Lowercase + split on non-alphanumeric characters.

    LEARNING: Tokenisation matters for BM25 because it works on exact token
    matches. "billing" and "Billing" should be the same token, so we lowercase.
    Punctuation is noise, so we strip it.
    """
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """
    Stateful BM25 retriever over a fixed corpus of text chunks.

    LEARNING: BM25 is not a persistent index like ChromaDB — it lives in
    memory and is rebuilt from the corpus each time. For a call-center
    transcript (10–30 chunks), this takes microseconds and needs no DB.

    Usage:
        retriever = BM25Retriever(chunks)
        results = retriever.retrieve("billing dispute refund", top_k=5)
    """

    def __init__(self, chunks: List[str]):
        if not chunks:
            raise ValueError("BM25Retriever requires at least one chunk")

        self._chunks = chunks
        tokenized = [_tokenize(chunk) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug(f"BM25Retriever: indexed {len(chunks)} chunk(s)")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return top_k chunks with their BM25 scores.

        Returns:
            List of (chunk_text, bm25_score), sorted descending by score.

        LEARNING: BM25 scores are not normalised to [0,1]. They're raw floats.
        What matters is the RANK, not the absolute value — which is exactly
        what RRF uses later.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair each chunk with its score, sort descending
        ranked = sorted(
            zip(self._chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top = ranked[:top_k]
        logger.debug(
            f"BM25: top scores = {[round(s, 3) for _, s in top]}"
        )
        return top

    def get_scores(self, query: str) -> List[float]:
        """Raw scores for all chunks — used by RRF fusion."""
        query_tokens = _tokenize(query)
        if not query_tokens:
            return [0.0] * len(self._chunks)
        return list(self._bm25.get_scores(query_tokens))
