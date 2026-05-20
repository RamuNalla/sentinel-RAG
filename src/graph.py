import sys
import os
from typing import List, NotRequired, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.nodes import retrieve, grade_documents, generate_answer, web_search, format_docs
from src.graders import hallucination_grader, answer_grader, is_grade_yes

MAX_GENERATION_RETRIES = 3
MAX_WEB_SEARCHES = 2

# ==========================================
# 1. Define State
# ==========================================
class GraphState(TypedDict):
    """Represents the state of our graph."""

    question: str
    documents: NotRequired[List[Document]]
    generation: NotRequired[str]
    retries: NotRequired[int]
    web_search_count: NotRequired[int]


# ==========================================
# 2. Define Conditional Routing Logic
# ==========================================
def decide_to_generate(state):
    """Route to generation or web search based on document relevance grading."""
    print("---EDGE: EVALUATE RETRIEVAL RESULTS---")
    filtered_documents = state.get("documents", [])

    if not filtered_documents:
        print("  - DECISION: All documents irrelevant. Routing to Web Search.")
        return "web_search"
    print("  - DECISION: Documents relevant. Routing to Generate.")
    return "generate_answer"


def check_hallucinations(state):
    """Check groundedness and whether the answer resolves the question."""
    print("---EDGE: CHECK FOR HALLUCINATIONS---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    retries = state.get("retries", 0)
    web_search_count = state.get("web_search_count", 0)

    if retries >= MAX_GENERATION_RETRIES:
        print("  - MAX RETRIES REACHED. Ending process.")
        return "max_retries"

    facts = format_docs(documents) if documents else ""
    score = hallucination_grader.invoke({"documents": facts, "generation": generation})

    if is_grade_yes(score.binary_score):
        print("  - DECISION: Answer is Grounded (No Hallucination).")
        print("---EDGE: CHECK IF ANSWER RESOLVES QUERY---")
        answer_score = answer_grader.invoke({"question": question, "generation": generation})
        if is_grade_yes(answer_score.binary_score):
            print("  - DECISION: Answer is Useful and Resolves Query.")
            return "useful"
        print("  - DECISION: Answer does not resolve query.")
        if web_search_count >= MAX_WEB_SEARCHES:
            print("  - MAX WEB SEARCHES REACHED. Ending process.")
            return "max_retries"
        print("  - Routing to Web Search.")
        return "not_useful"

    print("  - DECISION: HALLUCINATION DETECTED.")
    if web_search_count < 1:
        print("  - Routing to Web Search for additional context.")
        return "need_web_search"
    print("  - Retrying generation with existing context.")
    return "not_supported"


# ==========================================
# 3. Build & Compile the Graph
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("web_search", web_search)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate_answer": "generate_answer",
    },
)

workflow.add_edge("web_search", "generate_answer")

workflow.add_conditional_edges(
    "generate_answer",
    check_hallucinations,
    {
        "not_supported": "generate_answer",
        "not_useful": "web_search",
        "need_web_search": "web_search",
        "useful": END,
        "max_retries": END,
    },
)

app = workflow.compile()
