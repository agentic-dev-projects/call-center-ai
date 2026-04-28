"""
Semantic Cache using ChromaDB
"""

import chromadb
import json
from sentence_transformers import SentenceTransformer

from chromadb.config import Settings

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client(
    Settings(
        persist_directory="./chroma_cache",
        anonymized_telemetry=False
    )
)
collection = client.get_or_create_collection(name="semantic_cache")


def get_embedding(text: str):
    return model.encode(text).tolist()


def get_from_cache(query: str, threshold: float = 0.85):
    results = collection.query(
        query_embeddings=[get_embedding(query)],
        n_results=1
    )

    # ✅ SAFE GUARDS
    if not results or "documents" not in results:
        return None

    if not results["documents"] or not results["documents"][0]:
        return None

    if not results["distances"] or not results["distances"][0]:
        return None

    score = results["distances"][0][0]

    if score < (1 - threshold):
        return json.loads(results["metadatas"][0][0]["response"])

    return None


def store_in_cache(query: str, response: dict):
    collection.add(
        ids=[query],
        documents=[query],
        embeddings=[get_embedding(query)],
        metadatas=[{"response": json.dumps(response)}]
    )
    client.persist()