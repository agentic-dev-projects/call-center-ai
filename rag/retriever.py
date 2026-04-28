"""
Retriever: Fetches relevant chunks
"""

from rag.vector_store import collection
from rag.embedder import get_embedding


def retrieve(query: str, top_k: int = 3):
    results = collection.query(
        query_embeddings=[get_embedding(query)],
        n_results=top_k
    )

    # ✅ FIX
    if not results or "documents" not in results:
        return []

    if not results["documents"] or not results["documents"][0]:
        return []

    return results["documents"][0]