import os
import sys
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

# Initialize embeddings specifically for the cache
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True}
)

# We create a separate directory for the cache database
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache_db")

# Initialize the Cache Vector Store
cache_store = Chroma(
    collection_name="semantic_cache",
    persist_directory=CACHE_DIR,
    embedding_function=embeddings
)

def check_cache(query: str, distance_threshold: float = 0.2):
    """
    Check if a similar query exists in the cache.
    Chroma uses L2 distance by default. Lower is better. 
    Distance < 0.2 is roughly equivalent to > 0.90 cosine similarity.
    """
    print(f"--- CHECKING CACHE FOR: '{query}' ---")
    results = cache_store.similarity_search_with_score(query, k=1)
    
    if results:
        doc, distance = results[0]
        if distance < distance_threshold:
            print(f"--- 🟢 CACHE HIT! (Distance: {distance:.4f}) ---")
            return doc.metadata["answer"]
        else:
            print(f"--- 🔴 CACHE MISS! Closest match too far (Distance: {distance:.4f}) ---")
            return None
            
    print("--- 🔴 CACHE MISS! Cache is empty. ---")
    return None

def save_to_cache(query: str, answer: str):
    """Save the query and its generated answer to the cache."""
    cache_store.add_texts(
        texts=[query],
        metadatas=[{"answer": answer}]
    )
    print("--- 💾 SAVED NEW ANSWER TO CACHE ---")