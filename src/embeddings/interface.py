from typing import List
import numpy as np

class BaseEmbeddingModel:
    """Abstract interface for configurable embedding models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def get_dimension(self) -> int:
        raise NotImplementedError
