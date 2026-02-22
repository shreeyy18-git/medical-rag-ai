import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

_embeddings_instance = None

def get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """
    Returns the HuggingFace Endpoint embeddings model (Singleton).
    This uses the modern langchain-huggingface package for more stable API calls.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        if not HF_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables")
            
        _embeddings_instance = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=HF_TOKEN,
            model=EMBEDDING_MODEL_NAME
        )
    return _embeddings_instance
