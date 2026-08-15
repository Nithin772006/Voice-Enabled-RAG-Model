from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from guardrails.policies import ALLOWED_LANGUAGES

class RAGRequest(BaseModel):
    """Pydantic model validating incoming multilingual RAG query parameters."""
    query: str = Field(
        ..., 
        description="The query string to be answered using the RAG index.",
        min_length=1
    )
    language: Optional[str] = Field(
        default="tamil",
        description="Target response language (english, tamil, hindi, telugu)."
    )

    @field_validator("query")
    @classmethod
    def validate_query_text(cls, v: str) -> str:
        """Strips leading/trailing whitespaces and ensures query is not empty."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty or contain only whitespace.")
        return stripped

    @field_validator("language")
    @classmethod
    def validate_language_code(cls, v: Optional[str]) -> Optional[str]:
        """Ensures the target language is supported by our pipeline."""
        if v is None:
            return "tamil"
            
        lang = v.lower().strip()
        if lang not in ALLOWED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{lang}'. Supported values are: {', '.join(ALLOWED_LANGUAGES)}"
            )
        return lang

class SourceMetadata(BaseModel):
    """Metadata schema representing a single retrieved RAG context segment source."""
    chunk_id: str = Field(..., description="Unique hash representing the indexed text chunk.")
    score: Optional[float] = Field(default=None, description="Similarity score calculated by Qdrant (0.0 to 1.0).")
    language: Optional[str] = Field(default=None, description="Document source language.")
    text: Optional[str] = Field(default=None, description="The matched context snippet text.")

class RAGResponse(BaseModel):
    """Pydantic model representing the standardized structured response schema returned to clients."""
    status: str = Field(..., description="Request resolution status: 'success' | 'fallback' | 'error'")
    query: str = Field(..., description="The original normalized request query.")
    language: str = Field(..., description="Mapped target language code.")
    answer: str = Field(..., description="Grounded LLM-generated text or pre-translated fallback refusal.")
    sources: List[SourceMetadata] = Field(default_factory=list, description="List of vector references cited as evidence.")
    audio_url: Optional[str] = Field(default=None, description="HTTP URL to fetch the synthesized output WAV audio.")
    error: Optional[str] = Field(default=None, description="Error detail reason string if status is 'error' or 'fallback'.")
