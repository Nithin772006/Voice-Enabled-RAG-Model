from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChunkMetadata(BaseModel):
    selected: bool = Field(default=False, description="Whether the passage was marked as selected containing the answer")
    chunk_index: int = Field(default=0, description="The index of the chunk within the parent document")

class RetrievedChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the text chunk")
    document_id: str = Field(..., description="Identifier of the source document")
    language: str = Field(..., description="Language of the chunk text")
    text: str = Field(..., description="Actual text content of the chunk")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata, description="Associated metadata for the chunk")

class RAGPromptInput(BaseModel):
    query: str = Field(..., description="The user query")
    retrieved_chunks: List[RetrievedChunk] = Field(..., description="List of retrieved chunks to serve as context")
    language: str = Field(..., description="The target language code for generating the answer (e.g., 'tamil', 'english')")

class LLMOutput(BaseModel):
    answer: str = Field(..., description="The generated grounded answer")
    sources: List[str] = Field(default_factory=list, description="List of source chunk_ids used to formulate the answer")
