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

    return results["documents"][0]