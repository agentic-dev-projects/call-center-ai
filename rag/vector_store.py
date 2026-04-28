"""
Vector Store using ChromaDB
"""

import chromadb
from rag.embedder import get_embedding

client = chromadb.Client()

collection = client.get_or_create_collection(name="call_center")


def store_chunks(call_id: str, chunks: list):
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{call_id}_{i}"],
            documents=[chunk],
            embeddings=[get_embedding(chunk)],
            metadatas=[{"call_id": call_id}]
        )