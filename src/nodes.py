import os
import sys
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config
from src.graders import llm, retrieval_grader, hallucination_grader, answer_grader

# --- Setup Retriever ---
embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# Note: Ensure the path to chroma_db is correct based on where you run the script
vectorstore = Chroma(persist_directory=os.path.join(os.path.dirname(__file__), "..", config.CHROMA_PATH), embedding_function=embeddings)
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
    """Retrieve documents from vector store"""
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
        grade = score.binary_score
        
        if grade == "yes":
            print("  - GRADE: Document Relevant")
            filtered_docs.append(d)
        else:
            print("  - GRADE: Document Irrelevant (REJECTED)")
            
    return {"documents": filtered_docs, "question": question}

def generate_answer(state: Dict):
    """Generate answer using the retrieved or web-searched documents"""
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    retries = state.get("retries", 0)
    
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {"documents": documents, "question": question, "generation": generation, "retries": retries + 1}

def web_search(state: Dict):
    """Web search based on the re-phrased question."""
    print("---NODE: WEB SEARCH (CRAG FALLBACK)---")
    question = state["question"]
    documents = state["documents"]
    
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results)
    
    documents.append(web_results_doc)
    return {"documents": documents, "question": question}