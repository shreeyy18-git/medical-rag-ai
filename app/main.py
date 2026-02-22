from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes_chat import router as chat_router

app = FastAPI(title="Medical RAG AI Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Medical RAG AI is running"}

@app.get("/debug")
def debug_info():
    import os
    from app.rag.embeddings import HF_TOKEN
    from app.rag.vectorstore import CHROMA_PERSIST_DIR, get_vectorstore
    
    results = {
        "HUGGINGFACEHUB_API_TOKEN": "SET" if HF_TOKEN else "MISSING",
        "GROQ_API_KEY": "SET" if os.getenv("GROQ_API_KEY") else "MISSING",
        "GOOGLE_API_KEY": "SET" if os.getenv("GOOGLE_API_KEY") else "MISSING",
        "CHROMA_PERSIST_DIR": CHROMA_PERSIST_DIR,
        "DB_EXISTS": os.path.exists(CHROMA_PERSIST_DIR),
        "FILES_IN_DB": os.listdir(CHROMA_PERSIST_DIR) if os.path.exists(CHROMA_PERSIST_DIR) else []
    }
    
    try:
        vs = get_vectorstore()
        results["VECTORSTORE_STATUS"] = "CONNECTED"
    except Exception as e:
        results["VECTORSTORE_STATUS"] = f"ERROR: {str(e)}"
        
    return results

