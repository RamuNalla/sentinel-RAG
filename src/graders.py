import sys
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# Ensure we can import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

# Initialize the LLM (Temperature 0 is crucial for graders to be deterministic)
llm = ChatGroq(model=config.LLM_MODEL, temperature=0)

# ==========================================
# 1. Document Grader (Context Relevance)
# ==========================================
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_doc_grader = llm.with_structured_output(GradeDocuments)

system_prompt_doc_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

doc_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_doc_grader),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
retrieval_grader = doc_grader_prompt | structured_llm_doc_grader


# ==========================================
# 2. Hallucination Grader (Groundedness)
# ==========================================
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

system_prompt_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in and supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_hallucination),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader


# ==========================================
# 3. Answer Grader (Resolution)
# ==========================================
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

system_prompt_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_answer),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
answer_grader = answer_prompt | structured_llm_answer_grader


# ==========================================
# 4. Query Rewriter (Optimization)
# ==========================================
system_prompt_rewriter = """You are a question re-writer that converts an input user question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_rewriter),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)
question_rewriter = rewrite_prompt | llm | StrOutputParser()