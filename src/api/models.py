from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TextQueryRequest(BaseModel):
    query: str = Field(..., example="இந்தியாவின் தலைநகரம் என்ன?")
    mode: Optional[str] = "fast" # "fast" or "quality"
    top_k: Optional[int] = 5
    candidate_k: Optional[int] = 20
    alpha: Optional[float] = 0.70

class RetrieveOnlyRequest(BaseModel):
    query: str = Field(..., example="Manhattan project immediate impact")
    top_k: Optional[int] = 5
    candidate_k: Optional[int] = 20
    alpha: Optional[float] = 0.70

class BenchmarkRequest(BaseModel):
    num_queries: Optional[int] = 20
    mode: Optional[str] = "fast"

class SourcePassage(BaseModel):
    rank: int
    chunk_id: str
    text: str
    score: float
    original_record_id: str
    passage_idx: int
    is_selected: int

class QueryResponse(BaseModel):
    request_id: str
    query: str
    normalized_query: str
    language: str
    mode: str
    answer_mode: str # "retrieved" vs "generated"
    answer: str
    sources: List[SourcePassage]
    confidence: float
    is_grounded: bool
    status: str
    cache_hit: bool
    latencies_ms: Dict[str, float]
    guardrail_signals: Dict[str, Any]
    retrieval_debug: Dict[str, Any]
    rerank_debug: List[Dict[str, Any]]
    transcript: Optional[str] = None
