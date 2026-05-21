import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import cache
from src.graph import app


def process_query(question: str) -> dict:
    """
    Route a question through the semantic cache or LangGraph agent.

    Returns:
        dict with keys: answer, from_cache, latency_seconds
    """
    print(f"\n==================================================")
    print(f"[USER QUERY]: {question}")
    print(f"==================================================")

    start_time = time.time()

    cached_answer = cache.check_cache(question)
    if cached_answer:
        latency = time.time() - start_time
        print("\n[FINAL ANSWER FROM CACHE]:")
        print(cached_answer)
        print(f"\n⚡ [LATENCY]: {latency:.4f} seconds ⚡")
        return {
            "answer": cached_answer,
            "from_cache": True,
            "latency_seconds": latency,
        }

    print("\n--- INITIATING AGENTIC WORKFLOW ---")
    inputs = {"question": question, "retries": 0, "web_search_count": 0}
    final_state = dict(inputs)

    for output in app.stream(inputs):
        for node, update in output.items():
            final_state.update(update)

    answer = final_state.get("generation", "")
    cache.save_to_cache(question, answer)

    latency = time.time() - start_time
    print("\n[FINAL ANSWER FROM AGENT]:")
    print(answer)
    print(f"\n🐢 [LATENCY]: {latency:.4f} seconds 🐢")

    return {
        "answer": answer,
        "from_cache": False,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    print("Welcome to SentinelRAG Gateway. Type 'exit' to quit.")
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        process_query(user_input)
