from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from llm.schemas import RetrievedChunk

class ValidationStatus(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"
    OUT_OF_DOMAIN = "out_of_domain"
    LOW_CONFIDENCE = "low_confidence"
    ERROR = "error"

class InputValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether the user input passed validation")
    status: ValidationStatus = Field(..., description="The validation status category")
    reason: Optional[str] = Field(default=None, description="Detailed explanation if validation failed")
    sanitized_query: Optional[str] = Field(default=None, description="Cleaned/sanitized query text")

class RetrievalValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether retrieved context matches required constraints")
    status: ValidationStatus = Field(..., description="The retrieval status category")
    reason: Optional[str] = Field(default=None, description="Explanation if retrieval check failed")
    filtered_chunks: List[RetrievedChunk] = Field(default_factory=list, description="Verified list of retrieved chunks")

class OutputValidationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether generated output passed containment and grounding rules")
    status: ValidationStatus = Field(..., description="The output status category")
    reason: Optional[str] = Field(default=None, description="Explanation if output validation failed")
    final_output: Optional[Dict[str, Any]] = Field(default=None, description="Standardized output dictionary matching schema")
