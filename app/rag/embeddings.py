import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

_embeddings_instance = None

def get_embeddings() -> HuggingFaceInferenceAPIEmbeddings:
    """
    Returns the HuggingFace Inference API embeddings model (Singleton).
    This avoids loading the model locally, saving significant memory.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        if not HF_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment variables")
            
        _embeddings_instance = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_TOKEN,
            model_name=EMBEDDING_MODEL_NAME
        )
    return _embeddings_instance
