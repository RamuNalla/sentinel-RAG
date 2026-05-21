import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import cache
from src.graph import app

_phoenix_session = None
DEFAULT_PHOENIX_URL = "http://localhost:6006/"


def setup_observability(project_name: str = "sentinel-rag", launch_ui: bool = True):
    """Start Phoenix UI (optional) and instrument LangChain/LangGraph LLM calls."""
    global _phoenix_session
    if _phoenix_session is not None:
        return _phoenix_session

    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    session = px.launch_app() if launch_ui else None
    tracer_provider = register(project_name=project_name)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    if session is not None:
        print(f"👁️ Observability Dashboard running at: {session.url}")
    elif launch_ui:
        print(
            "⚠️ Phoenix UI did not start (port may be in use). "
            f"Try opening {DEFAULT_PHOENIX_URL} or restart the kernel."
        )
    else:
        print("🔭 OpenTelemetry tracing enabled (Phoenix UI launch skipped).")

    _phoenix_session = session
    return session


def _documents_to_contexts(documents) -> list[str]:
    if not documents:
        return []
    return [
        doc.page_content
        for doc in documents
        if getattr(doc, "page_content", None)
    ]


def process_query(question: str, use_cache: bool = True) -> dict:
    """
    Route a question through the semantic cache or LangGraph agent.

    Returns:
        dict with keys: answer, from_cache, latency_seconds, contexts
    """
    print(f"\n==================================================")
    print(f"[USER QUERY]: {question}")
    print(f"==================================================")

    start_time = time.time()

    if use_cache:
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
                "contexts": [],
            }

    print("\n--- INITIATING AGENTIC WORKFLOW ---")
    inputs = {"question": question, "retries": 0, "web_search_count": 0}
    final_state = dict(inputs)

    for output in app.stream(inputs):
        for node, update in output.items():
            final_state.update(update)

    answer = final_state.get("generation", "")
    contexts = _documents_to_contexts(final_state.get("documents", []))

    if use_cache:
        cache.save_to_cache(question, answer)

    latency = time.time() - start_time
    print("\n[FINAL ANSWER FROM AGENT]:")
    print(answer)
    print(f"\n🐢 [LATENCY]: {latency:.4f} seconds 🐢")

    return {
        "answer": answer,
        "from_cache": False,
        "latency_seconds": latency,
        "contexts": contexts,
    }


if __name__ == "__main__":
    setup_observability()
    print("Welcome to SentinelRAG Gateway. Type 'exit' to quit.")
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        process_query(user_input)
