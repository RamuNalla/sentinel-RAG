import gc
import os
import shutil
import sys
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config

CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "cache_db")
)

embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={"device": config.EMBEDDING_DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)


def _create_cache_store() -> Chroma:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return Chroma(
        collection_name="semantic_cache",
        persist_directory=CACHE_DIR,
        embedding_function=embeddings,
    )


cache_store = _create_cache_store()


def check_cache(query: str, distance_threshold: Optional[float] = None):
    """
    Check if a similar query exists in the cache.
    Chroma uses L2 distance by default. Lower is better.
    """
    threshold = (
        distance_threshold
        if distance_threshold is not None
        else config.CACHE_DISTANCE_THRESHOLD
    )
    print(f"--- CHECKING CACHE FOR: '{query}' ---")
    results = cache_store.similarity_search_with_score(query, k=1)

    if results:
        doc, distance = results[0]
        if distance < threshold:
            print(f"--- 🟢 CACHE HIT! (Distance: {distance:.4f}) ---")
            return doc.metadata["answer"]
        print(f"--- 🔴 CACHE MISS! Closest match too far (Distance: {distance:.4f}) ---")
        return None

    print("--- 🔴 CACHE MISS! Cache is empty. ---")
    return None


def save_to_cache(query: str, answer: str):
    """Save the query and its generated answer to the cache."""
    cache_store.add_texts(
        texts=[query],
        metadatas=[{"answer": answer}],
    )
    print("--- 💾 SAVED NEW ANSWER TO CACHE ---")


def clear_cache():
    """Remove all cached entries (safe for notebooks — avoids rmtree when possible)."""
    global cache_store

    try:
        ids = cache_store.get(include=[]).get("ids") or []
        if ids:
            cache_store.delete(ids)
        print(f"--- CACHE CLEARED ({len(ids)} entries removed) ---")
        return
    except Exception as exc:
        print(f"--- CACHE RESET (reason: {exc}) ---")

    # Fallback only when the existing client/collection is corrupted.
    try:
        cache_store.delete_collection()
    except Exception:
        pass

    del cache_store
    gc.collect()

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    cache_store = _create_cache_store()
    print("--- CACHE CLEARED ---")
