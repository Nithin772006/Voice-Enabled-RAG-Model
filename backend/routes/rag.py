import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

# Resolve local imports cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.config import APIConfig
from backend.schemas import RAGRequest, RAGResponse, SourceMetadata
from backend.dependencies import get_rag_graph

logger = logging.getLogger("RAGRouter")
router = APIRouter(prefix="/api/v1")

# Regex validating safe audio filenames (preventing path traversal entirely)
SAFE_AUDIO_FILENAME_REGEX = re.compile(r"^response_\d+_[a-z]+\.wav$")

@router.post("/rag/query", response_model=RAGResponse)
async def query_rag(request: RAGRequest, graph=Depends(get_rag_graph)):
    """Runs the complete RAG + Voice LangGraph workflow for a user query."""
    logger.info(f"Received API query request: {repr(request.query)} (lang: {request.language})")
    
    try:
        # 1. Invoke the LangGraph workflow
        state = graph.invoke({
            "query": request.query,
            "language": request.language
        })
        
        # 2. Handle critical system error outcomes from Graph
        if state.get("status") == "error":
            err_msg = state.get("error", "An unexpected error occurred during graph execution.")
            logger.error(f"Graph execution failed: {err_msg}")
            
            # Distinguish user input validation error (400) from system crash (500)
            if "query" in err_msg.lower() or "language" in err_msg.lower():
                raise HTTPException(status_code=400, detail=err_msg)
            else:
                raise HTTPException(status_code=500, detail=err_msg)
                
        # 3. Format sources payload
        sources = []
        for src in state.get("sources", []):
            sources.append(
                SourceMetadata(
                    chunk_id=src.get("chunk_id", "unknown_source"),
                    score=src.get("score"),
                    language=src.get("language"),
                    text=src.get("text")
                )
            )
            
        # 4. Map the physical audio path to a secure, public API endpoint path
        audio_url = None
        physical_audio_path = state.get("audio_path")
        if physical_audio_path and os.path.exists(physical_audio_path):
            audio_filename = os.path.basename(physical_audio_path)
            # Public-facing HTTP-accessible endpoint URL
            audio_url = f"/api/v1/audio/{audio_filename}"
            
        # 5. Compile standardized structured RAGResponse
        return RAGResponse(
            status=state.get("status", "success"),
            query=state.get("query"),
            language=state.get("language"),
            answer=state.get("answer", ""),
            sources=sources,
            audio_url=audio_url,
            error=state.get("error")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected backend exception: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.get("/audio/{audio_filename}")
async def get_audio_file(audio_filename: str):
    """Serves the generated WAV audio response files.
    Strictly sanitizes inputs to prevent directory traversal attacks.
    """
    logger.info(f"Requested audio file download: '{audio_filename}'")
    
    # 1. Regex validation to ensure the filename only conforms to safe parameters
    if not SAFE_AUDIO_FILENAME_REGEX.match(audio_filename):
        logger.warning(f"Rejected malicious/invalid audio filename: '{audio_filename}'")
        raise HTTPException(status_code=400, detail="Invalid audio filename format.")
        
    # 2. Construct the resolved physical path in AUDIO_DIR
    audio_file_path = os.path.join(APIConfig.AUDIO_DIR, audio_filename)
    
    # 3. Verify file presence on disk
    if not os.path.exists(audio_file_path) or not os.path.isfile(audio_file_path):
        logger.warning(f"Requested audio file not found on disk: {audio_file_path}")
        raise HTTPException(status_code=404, detail="Audio file not found.")
        
    # 4. Return file using FastAPI's FileResponse
    return FileResponse(
        path=audio_file_path,
        media_type="audio/wav",
        filename=audio_filename
    )
