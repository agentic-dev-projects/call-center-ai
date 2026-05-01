"""
Retriever: Fetches relevant chunks
"""

from rag.vector_store import collection
from rag.embedder import get_embedding
from config.settings import settings


def retrieve(query: str, top_k: int = None):
    top_k = top_k if top_k is not None else settings.RAG_TOP_K

    results = collection.query(
        query_embeddings=[get_embedding(query)],
        n_results=top_k
    )

    if not results or "documents" not in results:
        return []

    if not results["documents"] or not results["documents"][0]:
        return []

    return results["documents"][0]
