import os
import sys
import logging
import warnings
from dotenv import load_dotenv

# Aggressively suppress HuggingFace/Transformers logs and warnings BEFORE importing them
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embeddings_instance = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Returns the HuggingFace embeddings model configured for the project (Singleton).
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings_instance
