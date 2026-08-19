import os
import time
import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
from src.embeddings.e5_embeddings import MultilingualE5Embedding

class DenseIndexer:
    """FAISS-based dense vector indexer supporting cosine similarity search."""
    
    def __init__(self, embedding_model: MultilingualE5Embedding = None):
        self.embedding_model = embedding_model or MultilingualE5Embedding()
        self.index: faiss.Index = None
        self.dimension: int = 0

    def load_model(self):
        self.embedding_model.load_model()
        self.dimension = self.embedding_model.get_dimension()

    def build_index(self, texts: List[str], batch_size: int = 128) -> np.ndarray:
        self.load_model()
        if not texts:
            raise ValueError("No texts provided to build dense index.")

        print(f"[DENSE INDEX] Encoding {len(texts)} chunks in batches of {batch_size}...")
        t0 = time.perf_counter()
        embeddings = self.embedding_model.embed_documents(texts, batch_size=batch_size)
        encode_sec = time.perf_counter() - t0
        print(f"[DENSE INDEX] Encoded {len(texts)} vectors in {encode_sec:.2f}s")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        print(f"[DENSE INDEX] FAISS IndexFlatIP built with {self.index.ntotal} vectors.")
        return embeddings

    def add_vectors(self, embeddings: np.ndarray):
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, top_k: int = 20) -> Tuple[List[int], List[float], float, float]:
        """Returns: (indices, scores, embedding_ms, faiss_search_ms)"""
        if self.index is None or self.index.ntotal == 0:
            return [], [], 0.0, 0.0

        t0 = time.perf_counter()
        query_vec = self.embedding_model.embed_query(query)
        t1 = time.perf_counter()
        scores, indices = self.index.search(query_vec, top_k)
        t2 = time.perf_counter()

        embed_ms = (t1 - t0) * 1000.0
        faiss_ms = (t2 - t1) * 1000.0

        return indices[0].tolist(), scores[0].tolist(), embed_ms, faiss_ms

    def save(self, file_path: str):
        if self.index is None:
            raise ValueError("Cannot save uninitialized index.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        faiss.write_index(self.index, file_path)
        print(f"[DENSE INDEX] Saved FAISS index to {file_path}")

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"FAISS index file not found: {file_path}")
        self.index = faiss.read_index(file_path)
        self.dimension = self.index.d
        print(f"[DENSE INDEX] Loaded FAISS index ({self.index.ntotal} vectors, d={self.dimension}) from {file_path}")

    def diagnostics(self) -> Dict[str, Any]:
        diag = self.embedding_model.diagnostics()
        diag["index_total_vectors"] = self.index.ntotal if self.index else 0
        return diag
