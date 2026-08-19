import os
import re
import pickle
import time
import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi

class SparseIndexer:
    """BM25-based sparse lexical indexer supporting multilingual text retrieval."""
    
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.corpus_tokens: List[List[str]] = []

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        # Support lowercased word tokens across Indic and Latin scripts
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def build_index(self, texts: List[str]):
        t0 = time.perf_counter()
        self.corpus_tokens = [self.tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        build_sec = time.perf_counter() - t0
        print(f"[SPARSE INDEX] BM25 Index built over {len(texts)} chunks in {build_sec:.2f}s")

    def search(self, query: str, top_k: int = 20) -> Tuple[List[int], List[float], float]:
        if self.bm25 is None or not self.corpus_tokens:
            return [], [], 0.0

        t0 = time.perf_counter()
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return [], [], 0.0

        scores = self.bm25.get_scores(q_tokens)
        top_k = min(top_k, len(scores))
        
        # Get top-k indices sorted by BM25 score descending
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()
        top_scores = [float(scores[i]) for i in top_indices]
        search_ms = (time.perf_counter() - t0) * 1000.0

        return top_indices, top_scores, search_ms

    def save(self, file_path: str):
        if self.bm25 is None:
            raise ValueError("Cannot save uninitialized BM25 index.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "corpus_tokens": self.corpus_tokens}, f)
        print(f"[SPARSE INDEX] Saved BM25 index to {file_path}")

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"BM25 index file not found: {file_path}")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus_tokens = data["corpus_tokens"]
        print(f"[SPARSE INDEX] Loaded BM25 index ({len(self.corpus_tokens)} documents) from {file_path}")
