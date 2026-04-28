"""
Semantic Cache using ChromaDB
"""

import hashlib
import json
from sentence_transformers import SentenceTransformer

from db.chroma_client import client

model = SentenceTransformer("all-MiniLM-L6-v2")

# cosine space: ChromaDB returns (1 - cosine_similarity) as distance,
# so similarity = 1 - distance is exact (no approximation needed).
collection = client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": "cosine"}
)


def get_embedding(text: str):
    return model.encode(text).tolist()

def generate_id(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_from_cache(query: str, similarity_threshold: float = 0.85):
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["metadatas", "distances"]
    )

    if not results or not results.get("ids") or not results["ids"][0]:
        print("❌ CACHE MISS (empty collection)")
        return None

    # With cosine space: distance = 1 - cosine_similarity, so similarity = 1 - distance
    distance = results["distances"][0][0]
    similarity = 1.0 - distance

    print(f"Closest match similarity: {similarity:.4f}")

    if similarity < similarity_threshold:
        print(f"❌ CACHE MISS (similarity {similarity:.4f} < threshold {similarity_threshold})")
        return None

    print(f"🚀 CACHE HIT (similarity: {similarity:.4f})")
    return json.loads(results["metadatas"][0][0]["response"])


def store_in_cache(query: str, response: dict):
    cache_id = generate_id(query)
    embedding = get_embedding(query)

    # upsert instead of add — prevents DuplicateIDError if same query is cached twice
    collection.upsert(
        ids=[cache_id],
        embeddings=[embedding],
        documents=[query],
        metadatas=[{"response": json.dumps(response)}]
    )