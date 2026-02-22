import os
from langchain_chroma import Chroma
from app.rag.embeddings import get_embeddings
from dotenv import load_dotenv

load_dotenv()


CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vectorstore")

def get_vectorstore() -> Chroma:
    """
    Initializes and returns the Chroma vectorstore.
    """
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
