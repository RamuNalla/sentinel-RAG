import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configurations
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama3-70b-8192"

# Database path
CHROMA_PATH = "chroma_db"
DATA_PATH = "data/self_rag_paper.pdf"