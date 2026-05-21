import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cache import check_cache, save_to_cache
from src.graph import app

def process_query(question: str):
    print(f"\n==================================================")
    print(f"[USER QUERY]: {question}")
    print(f"==================================================")
    
    start_time = time.time()
    
    # 1. Intercept with Cache
    cached_answer = check_cache(question)
    
    if cached_answer:
        end_time = time.time()
        print("\n[FINAL ANSWER FROM CACHE]:")
        print(cached_answer)
        print(f"\n⚡ [LATENCY]: {end_time - start_time:.4f} seconds ⚡")
        return cached_answer
    
    # 2. If Cache Miss, Route to LangGraph Agent
    print("\n--- INITIATING AGENTIC WORKFLOW ---")
    inputs = {"question": question, "retries": 0}
    final_state = None
    
    for output in app.stream(inputs):
        for key, value in output.items():
            final_state = value
            # Nodes will print their own logs from Phase 3
            pass 
    
    answer = final_state["generation"]
    
    # 3. Save the expensive output to Cache for next time
    save_to_cache(question, answer)
    
    end_time = time.time()
    print("\n[FINAL ANSWER FROM AGENT]:")
    print(answer)
    print(f"\n🐢 [LATENCY]: {end_time - start_time:.4f} seconds 🐢")
    
    return answer

if __name__ == "__main__":
    # You can run this file directly to test via terminal input
    print("Welcome to SentinelRAG Gateway. Type 'exit' to quit.")
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        process_query(user_input)