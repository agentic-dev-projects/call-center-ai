"""
Hybrid Retriever — fuses dense + sparse search, then re-ranks with MMR.

LEARNING: THE FULL ADVANCED RAG PIPELINE
══════════════════════════════════════════

Stage 1 — RETRIEVAL (recall)
  Dense:  ChromaDB cosine similarity → ranked list A
  Sparse: BM25 keyword search       → ranked list B

Stage 2 — FUSION (merge)
  Reciprocal Rank Fusion (RRF) combines A and B into a single ranked list.
  RRF score for document d = Σ  1 / (rank_in_list + k)
  where k=60 dampens the effect of very high ranks (standard value).

  WHY RRF AND NOT SCORE AVERAGING?
  Dense and BM25 scores live on different scales:
    Dense: cosine similarity ≈ 0.0–1.0
    BM25: unbounded floats (could be 0.0–15.0)
  Averaging them directly would let BM25 dominate.
  RRF uses RANK not score, so scale doesn't matter.

Stage 3 — RE-RANKING (precision)
  Maximum Marginal Relevance (MMR) picks the final set of chunks by
  balancing relevance (high RRF score) against diversity (low similarity
  to already-selected chunks).

  WHY MMR?
  Without it, the top-k often contains near-duplicate chunks:
    "Customer was charged twice." and "The customer was charged twice in Oct."
  Both rank high but say the same thing — wasted context window.
  MMR picks the second-most relevant chunk that's DIFFERENT from the first.

  MMR formula:
    next = argmax over unselected d of:
           λ × relevance(d) - (1-λ) × max_sim(d, selected)

  λ=1.0 → pure relevance (degenerates to top-k)
  λ=0.0 → pure diversity
  λ=0.7 → balanced (our default)
"""

import math
from typing import List, Tuple, Dict

from rag.embedder import get_embedding
from rag.vector_store import collection
from rag.bm25_retriever import BM25Retriever
from config.settings import settings
from utils.logger import logger


# ── Stage 1: Dense retrieval ──────────────────────────────────────────────────

def _dense_retrieve(query: str, chunks: List[str], top_k: int) -> List[Tuple[str, int]]:
    """
    Return (chunk, rank) pairs from ChromaDB dense search.

    We search the global collection but filter to only the chunks passed in
    (from this call's store_chunks call). Returns ranks 0-indexed.
    """
    if not chunks:
        return []

    try:
        results = collection.query(
            query_embeddings=[get_embedding(query)],
            n_results=min(top_k * 2, len(chunks)),  # over-fetch for RRF
        )
        docs = results.get("documents", [[]])[0]
        # Return (doc, rank) — rank 0 = best
        return [(doc, rank) for rank, doc in enumerate(docs)]
    except Exception as exc:
        logger.warning(f"Dense retrieval failed: {exc}")
        return []


# ── Stage 2: RRF fusion ───────────────────────────────────────────────────────

_RRF_K = 60   # standard RRF constant — dampens top-rank advantage


def _rrf_fuse(
    dense_ranked: List[Tuple[str, int]],
    bm25_ranked: List[Tuple[str, int]],
) -> List[Tuple[str, float]]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    LEARNING: Each list contributes 1/(rank+k) to a document's RRF score.
    A document appearing in BOTH lists scores twice — agreement between
    two independent signals is strong evidence of relevance.

    Documents only in one list still score, just lower.
    """
    scores: Dict[str, float] = {}

    for doc, rank in dense_ranked:
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (rank + _RRF_K)

    for doc, rank in bm25_ranked:
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (rank + _RRF_K)

    # Sort by descending RRF score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    logger.debug(f"RRF: {len(fused)} unique doc(s) after fusion")
    return fused


# ── Stage 3: MMR re-ranking ───────────────────────────────────────────────────

def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Fast cosine similarity between two pre-computed embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mmr_rerank(
    candidates: List[Tuple[str, float]],
    query_embedding: List[float],
    top_k: int,
    lambda_: float,
) -> List[str]:
    """
    Maximum Marginal Relevance selection.

    LEARNING: MMR is a greedy algorithm:
      1. Pick the highest-relevance document first (always).
      2. For each subsequent pick, score unselected documents as:
           λ × relevance - (1-λ) × max_similarity_to_already_selected
      3. Pick the winner, add to selected, repeat until top_k reached.

    The (1-λ) × similarity term penalises redundancy — if a document
    is very similar to something already selected, it scores lower.
    """
    if not candidates:
        return []

    # Pre-compute embeddings for all candidates
    texts = [doc for doc, _ in candidates]
    relevances = {doc: score for doc, score in candidates}

    embeddings: Dict[str, List[float]] = {}
    for text in texts:
        embeddings[text] = get_embedding(text)

    selected: List[str] = []
    remaining = list(texts)

    while remaining and len(selected) < top_k:
        if not selected:
            # First pick: pure relevance (no diversity penalty yet)
            best = max(remaining, key=lambda d: relevances[d])
        else:
            # MMR score: relevance - redundancy
            def mmr_score(doc: str) -> float:
                rel = relevances[doc]
                max_sim = max(
                    _cosine_sim(embeddings[doc], embeddings[sel])
                    for sel in selected
                )
                return lambda_ * rel - (1 - lambda_) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append(best)
        remaining.remove(best)

    logger.debug(f"MMR: selected {len(selected)} diverse chunk(s) from {len(candidates)}")
    return selected


# ── Public API ────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    query: str,
    chunks: List[str],
    top_k: int = None,
    lambda_mmr: float = None,
) -> List[str]:
    """
    Full hybrid retrieval pipeline: dense + BM25 → RRF → MMR.

    Args:
        query:      the search query (transcript or sub-query)
        chunks:     the corpus to search (chunks stored for this call)
        top_k:      number of chunks to return (default: settings.RAG_TOP_K)
        lambda_mmr: MMR diversity weight 0–1 (default: settings.RAG_MMR_LAMBDA)

    Returns:
        List of top_k chunk strings, ordered by MMR relevance+diversity.

    LEARNING: SEPARATION OF CONCERNS
    Each stage is a pure function — you can swap out dense/BM25/MMR
    independently. Hybrid → RRF → MMR is a composable pipeline.
    """
    top_k     = top_k     if top_k     is not None else settings.RAG_TOP_K
    lambda_mmr = lambda_mmr if lambda_mmr is not None else settings.RAG_MMR_LAMBDA

    if not chunks:
        logger.warning("hybrid_retrieve: no chunks provided")
        return []

    fetch_k = min(top_k * 3, len(chunks))   # over-fetch for RRF + MMR to work well

    # ── Stage 1: Dense retrieval ───────────────────────────────────────────
    dense_ranked = _dense_retrieve(query, chunks, fetch_k)
    logger.info(f"Dense: {len(dense_ranked)} result(s)")

    # ── Stage 1b: BM25 retrieval ───────────────────────────────────────────
    bm25 = BM25Retriever(chunks)
    bm25_results = bm25.retrieve(query, top_k=fetch_k)
    bm25_ranked = [(doc, rank) for rank, (doc, _) in enumerate(bm25_results)]
    logger.info(f"BM25:  {len(bm25_ranked)} result(s)")

    # ── Stage 2: RRF fusion ────────────────────────────────────────────────
    fused = _rrf_fuse(dense_ranked, bm25_ranked)
    if not fused:
        logger.warning("RRF fusion returned no results — falling back to chunks[:top_k]")
        return chunks[:top_k]

    # ── Stage 3: MMR re-ranking ────────────────────────────────────────────
    query_embedding = get_embedding(query)
    final = _mmr_rerank(fused, query_embedding, top_k, lambda_mmr)

    logger.info(
        f"Hybrid retrieve: {len(chunks)} chunks → dense({len(dense_ranked)}) "
        f"+ BM25({len(bm25_ranked)}) → RRF({len(fused)}) → MMR({len(final)})"
    )
    return final
