import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from parser_chunker import ParserChunker
from embed_index import EmbedIndexManager
from rag_engine import RAGEngine, QueryProcessor

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Zodiac RAG Chatbot",
    description="A RAG-powered astrology chatbot based on Linda Goodman's Sun Signs and Love Signs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for components
parser_chunker = None
embed_index_manager = None
rag_engine = None

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)


def initialize_components():
    """Initialize RAG components"""
    global parser_chunker, embed_index_manager, rag_engine
    
    if parser_chunker is None:
        parser_chunker = ParserChunker(CHUNK_SIZE, CHUNK_OVERLAP)
    
    if embed_index_manager is None:
        embed_index_manager = EmbedIndexManager(
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            index_path=FAISS_INDEX_PATH
        )
        
        # Try to load existing index
        if not embed_index_manager.load_index():
            print("No existing index found. Please upload PDFs first.")
    
    if rag_engine is None:
        rag_engine = RAGEngine(
            embedding_index_manager=embed_index_manager,
            model=os.getenv("OPENAI_MODEL", "gpt-4")
        )


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    initialize_components()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        initialize_components()
        
        # Check if OpenAI API key is set
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "OpenAI API key not configured",
                    "details": "Please set OPENAI_API_KEY in your environment variables"
                }
            )
        
        # Get index stats
        stats = embed_index_manager.get_stats() if embed_index_manager else {"status": "Not initialized"}
        
        return {
            "status": "healthy",
            "openai_configured": bool(openai_key),
            "index_stats": stats,
            "components_initialized": all([parser_chunker, embed_index_manager, rag_engine])
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(e)
            }
        )


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize components if needed
        initialize_components()
        
        # Process PDF
        print(f"Processing PDF: {file.filename}")
        if parser_chunker is None:
            raise HTTPException(status_code=500, detail="Parser not initialized")
        
        chunks = parser_chunker.process_pdf(file_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from PDF")
        
        # Create embeddings and add to index
        if embed_index_manager is None:
            raise HTTPException(status_code=500, detail="Embedding manager not initialized")
        
        embed_index_manager.process_chunks(chunks)
        
        # Save index
        embed_index_manager.save_index()
        
        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}",
            "chunks_created": len(chunks),
            "filename": file.filename
        }
        
    except Exception as e:
        # Clean up file if processing failed
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask")
async def ask_question(query: str = Form(...), top_k: Optional[int] = Form(5)):
    """Ask a question to the RAG chatbot"""
    try:
        # Initialize components if needed
        initialize_components()
        
        # Clean and validate query
        query = QueryProcessor.clean_query(query)
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if query is astrology-related
        if not QueryProcessor.is_astrology_related(query):
            return {
                "answer": "I'm designed to answer questions about astrology, zodiac signs, and Linda Goodman's insights. Please ask me about astrology-related topics!",
                "sources": [],
                "context_used": False,
                "query": query
            }
        
        # Process query
        if rag_engine is None:
            raise HTTPException(status_code=500, detail="RAG engine not initialized")
        
        result = rag_engine.process_query(query, top_k or 5)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/suggestions")
async def get_suggestions():
    """Get suggested questions"""
    return {
        "suggestions": QueryProcessor.suggest_queries()
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        initialize_components()
        stats = embed_index_manager.get_stats() if embed_index_manager else {}
        
        # Count uploaded files
        uploaded_files = []
        if os.path.exists(UPLOAD_DIR):
            uploaded_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.pdf')]
        
        return {
            "index_stats": stats,
            "uploaded_files": uploaded_files,
            "upload_dir": UPLOAD_DIR,
            "index_path": FAISS_INDEX_PATH
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.delete("/reset")
async def reset_index():
    """Reset the entire index (dangerous operation)"""
    try:
        global embed_index_manager, rag_engine
        
        # Remove index files
        faiss_path = f"{FAISS_INDEX_PATH}.faiss"
        metadata_path = f"{FAISS_INDEX_PATH}_metadata.json"
        
        if os.path.exists(faiss_path):
            os.remove(faiss_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        # Reset components
        embed_index_manager = None
        rag_engine = None
        
        # Reinitialize
        initialize_components()
        
        return {
            "status": "success",
            "message": "Index reset successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting index: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 