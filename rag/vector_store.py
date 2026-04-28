"""
Vector Store using ChromaDB
"""

import chromadb
from rag.embedder import get_embedding
from chromadb.config import Settings
from db.chroma_client import client


collection = client.get_or_create_collection(name="call_center")


def store_chunks(call_id: str, chunks: list):
    for i, chunk in enumerate(chunks):
        collection.upsert(
            ids=[f"{call_id}_{i}"],
            documents=[chunk],
            embeddings=[get_embedding(chunk)],
            metadatas=[{"call_id": call_id}]
        )