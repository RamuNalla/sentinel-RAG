import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from project root (works from notebooks/ too)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model configurations
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DEVICE = "cpu"  # use "mps" on Apple Silicon or "cuda" with GPU
LLM_MODEL = "llama-3.3-70b-versatile"

# Semantic cache: Chroma L2 distance — lower is more similar
CACHE_DISTANCE_THRESHOLD = 0.25

# Database path
CHROMA_PATH = "chroma_db"
DATA_PATH = "data/self_rag_paper.pdf"