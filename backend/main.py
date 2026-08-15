import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Resolve local imports cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import APIConfig
from backend.routes.rag import router as rag_router
from backend.dependencies import get_rag_graph

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FastAPIMain")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup validation events and cleanups."""
    logger.info("Initializing RAG Voice Generator API...")
    try:
        # Validate critical environment variables
        APIConfig.validate()
        logger.info("API Configuration validated successfully.")
    except Exception as e:
        logger.critical(f"Startup configuration validation failed: {e}")
        # Terminate startup sequence safely
        raise RuntimeError(f"Startup failed: {e}")
    yield
    logger.info("Shutting down API...")

# Initialize FastAPI application
app = FastAPI(
    title="RAG Voice Generator API",
    description="Multilingual RAG and voice generation backend orchestrating LangGraph workflows.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register RAG route controllers
app.include_router(rag_router)

# =================================================================
# HEALTH CHECK ENDPOINTS
# =================================================================

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Lightweight monitoring endpoint for API liveness."""
    return {"status": "ok"}


@app.get("/health/ready", tags=["Monitoring"])
async def ready_check():
    """Validates connectivity to Qdrant and checks LangGraph readiness."""
    try:
        import graph.nodes as nodes
        # If the retriever singleton is already loaded, reuse its client to avoid lock collisions
        if nodes._retriever is not None:
            qdrant_ok = nodes._retriever.client.collection_exists(nodes._retriever.collection_name)
        else:
            # If not loaded, open a transient raw client to check connection without loading BGE-M3
            from qdrant_client import QdrantClient
            client = QdrantClient(path=APIConfig.QDRANT_DB_PATH)
            qdrant_ok = client.collection_exists(APIConfig.QDRANT_COLLECTION)
            client.close()
        
        if qdrant_ok:
            return {
                "status": "ready",
                "qdrant": "ok",
                "graph": "ok"
            }
        else:
            raise HTTPException(
                status_code=503,
                detail={"status": "not_ready", "qdrant": "collection_missing", "graph": "ok"}
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "qdrant": f"error: {e}", "graph": "ok"}
        )
