import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import config

def ingest_data():
    print(f"Loading document from {config.DATA_PATH}...")
    loader = PyPDFLoader(config.DATA_PATH)
    docs = loader.load()
    
    print(f"Document loaded. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    print("Initializing HuggingFace BGE Embeddings (Running Locally)...")
    model_kwargs = {'device': 'mps'} # Change to 'cuda' or 'mps' if you have GPU/Mac M-series
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("Creating Chroma Vector Store...")
    # This will save the database locally to the CHROMA_PATH
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=config.CHROMA_PATH
    )
    
    print(f"Success! Vector database saved to ./{config.CHROMA_PATH}")

if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists("data"):
        print("Please create a 'data' folder and put your PDF inside it.")
    else:
        ingest_data()