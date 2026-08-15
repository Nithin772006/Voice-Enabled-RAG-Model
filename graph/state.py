from typing import TypedDict, List, Dict, Any, Optional

class RAGGraphState(TypedDict):
    """The strongly typed state representing the complete lifecycle of a single query request."""
    # Input Processing
    query: str
    language: str
    
    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]
    retrieval_passed: bool
    
    # Generation & Grounding
    prompt: Optional[str]
    answer: Optional[str]
    answer_valid: bool
    
    # Speech Synthesis
    audio_path: Optional[str]
    
    # Outputs & Attributions
    sources: List[Dict[str, Any]]
    error: Optional[str]
    status: str  # "success" | "fallback" | "error"
