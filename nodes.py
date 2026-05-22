import os
import sys
from typing import Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.graders import llm, retrieval_grader, question_rewriter, is_grade_yes

# --- Setup Retriever ---
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={'device': config.EMBEDDING_DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = Chroma(
    persist_directory=os.path.join(os.path.dirname(__file__), "..", config.CHROMA_PATH),
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Setup Generator ---
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:""",
    input_variables=["question", "context"],
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = prompt | llm | StrOutputParser()

# --- Setup Web Search ---
web_search_tool = TavilySearchResults(k=3)

# ==========================================
# GRAPH NODES
# ==========================================

def retrieve(state: Dict):
    """Retrieve documents from vector store."""
    print("---NODE: RETRIEVE FROM VECTOR DB---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state: Dict):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---NODE: GRADE DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if is_grade_yes(score.binary_score):
            print("  - GRADE: Document Relevant")
            filtered_docs.append(d)
        else:
            print("  - GRADE: Document Irrelevant (REJECTED)")

    return {"documents": filtered_docs, "question": question}


def generate_answer(state: Dict):
    """Generate answer using the retrieved or web-searched documents."""
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    retries = state.get("retries", 0)

    if not documents:
        generation = "I don't have enough context to answer this question."
    else:
        generation = rag_chain.invoke({"context": format_docs(documents), "question": question})

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "retries": retries + 1,
    }


def web_search(state: Dict):
    """Web search using a rewritten query (CRAG fallback)."""
    print("---NODE: WEB SEARCH (CRAG FALLBACK)---")
    question = state["question"]
    documents = list(state.get("documents", []))

    rewritten = question_rewriter.invoke({"question": question})
    search_query = rewritten.query
    print(f"  - Rewritten search query: {search_query}")

    results = web_search_tool.invoke({"query": search_query})
    contents = [r.get("content", "") for r in results if r.get("content")]
    web_results = "\n".join(contents) if contents else "No web search results found."
    web_results_doc = Document(page_content=web_results)

    return {
        "documents": documents + [web_results_doc],
        "question": question,
        "web_search_count": state.get("web_search_count", 0) + 1,
    }
