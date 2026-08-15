import os
from typing import List, Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded from the project root .env
load_dotenv()

class APIConfig:
    """Central configuration management class loading from environment variables."""
    # API subscription key
    SARVAM_API_KEY: Optional[str] = os.getenv("SARVAM_API_KEY")
    
    # CORS Origins (comma-separated list for Next.js app integration)
    CORS_ORIGINS: List[str] = [
        origin.strip() 
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") 
        if origin.strip()
    ]
    
    # Directories
    AUDIO_DIR: str = os.getenv("AUDIO_DIR", "data/audio_responses")
    QDRANT_DB_PATH: str = os.getenv("QDRANT_DB_PATH", "data/qdrant_db")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "tamil_rag_chunks")
    
    # Server runtime bindings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    @classmethod
    def validate(cls) -> None:
        """Validates critical environment configurations before starting uvicorn."""
        if not cls.SARVAM_API_KEY:
            raise ValueError("Required environment variable 'SARVAM_API_KEY' is missing or empty.")
            
        # Ensure audio output folder exists
        os.makedirs(cls.AUDIO_DIR, exist_ok=True)
