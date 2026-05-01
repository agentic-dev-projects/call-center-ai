"""
Semantic Cache using ChromaDB
"""

import hashlib
import json
from sentence_transformers import SentenceTransformer

from db.chroma_client import client
from config.settings import settings
from utils.logger import logger

model = SentenceTransformer("all-MiniLM-L6-v2")

# cosine space: distance = 1 - cosine_similarity, so similarity = 1 - distance exactly.
collection = client.get_or_create_collection(
    name="semantic_cache",
    metadata={"hnsw:space": "cosine"}
)


def get_embedding(text: str):
    return model.encode(text).tolist()


def generate_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def get_from_cache(query: str, similarity_threshold: float = None):
    threshold = similarity_threshold if similarity_threshold is not None else settings.CACHE_SIMILARITY_THRESHOLD
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["metadatas", "distances"]
    )

    if not results or not results.get("ids") or not results["ids"][0]:
        logger.debug("Cache MISS — collection is empty")
        return None

    distance = results["distances"][0][0]
    similarity = 1.0 - distance  # exact for cosine space

    logger.info(f"Cache lookup | similarity={similarity:.4f} | threshold={threshold}")

    if similarity < threshold:
        logger.info(f"Cache MISS | similarity={similarity:.4f} below threshold={threshold}")
        return None

    logger.info(f"Cache HIT  | similarity={similarity:.4f}")
    return json.loads(results["metadatas"][0][0]["response"])


def store_in_cache(query: str, response: dict):
    cache_id = generate_id(query)
    embedding = get_embedding(query)

    # upsert prevents DuplicateIDError when the same query is cached more than once
    collection.upsert(
        ids=[cache_id],
        embeddings=[embedding],
        documents=[query],
        metadatas=[{"response": json.dumps(response)}]
    )
    logger.info(f"Cache STORE | id={cache_id[:8]}…")
