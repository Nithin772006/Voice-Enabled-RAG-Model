import hashlib
from typing import Dict, Any, Optional
import numpy as np

class EmbeddingCache:
    """In-memory LRU cache for query vector embeddings to ensure low latency."""
    
    def __init__(self, max_capacity: int = 2000):
        self.max_capacity = max_capacity
        self.cache: Dict[str, np.ndarray] = {}

    def _hash_key(self, text: str, model_name: str) -> str:
        return hashlib.sha256(f"{model_name}:{text.strip().lower()}".encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        key = self._hash_key(text, model_name)
        return self.cache.get(key, None)

    def put(self, text: str, model_name: str, embedding: np.ndarray):
        if len(self.cache) >= self.max_capacity:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        key = self._hash_key(text, model_name)
        self.cache[key] = embedding

    def clear(self):
        self.cache.clear()

class QueryCache:
    """In-memory LRU cache for full query RAG pipeline responses."""
    
    def __init__(self, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _hash_key(self, query: str, mode: str) -> str:
        return hashlib.sha256(f"{mode}:{query.strip().lower()}".encode("utf-8")).hexdigest()

    def get(self, query: str, mode: str) -> Optional[Dict[str, Any]]:
        key = self._hash_key(query, mode)
        return self.cache.get(key, None)

    def put(self, query: str, mode: str, response: Dict[str, Any]):
        if len(self.cache) >= self.max_capacity:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        key = self._hash_key(query, mode)
        self.cache[key] = response

    def clear(self):
        self.cache.clear()
