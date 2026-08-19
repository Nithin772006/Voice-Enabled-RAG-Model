import time
import numpy as np
from typing import List, Dict, Any, Tuple
from rag_system.indexing import DenseIndexer, SparseIndexer
from rag_system.config import CANDIDATE_K, MAX_UNION_CANDIDATES


class HybridRetriever:
    """Hybrid Retriever combining Dense FAISS and Sparse BM25 candidate retrieval."""

    def __init__(
        self,
        dense_indexer: DenseIndexer,
        sparse_indexer: SparseIndexer,
        rag_documents: List[Dict[str, Any]],
        candidate_k: int = CANDIDATE_K,
        max_union_candidates: int = MAX_UNION_CANDIDATES,
    ):
        self.dense_indexer = dense_indexer
        self.sparse_indexer = sparse_indexer
        self.rag_documents = rag_documents
        self.candidate_k = candidate_k
        self.max_union_candidates = max_union_candidates

    def _candidate_from_index(
        self,
        idx: int,
        *,
        vector_score: float = 0.0,
        bm25_score: float = 0.0,
        rank: int = 0,
        source: str = "hybrid",
    ) -> Dict[str, Any]:
        doc = self.rag_documents[idx]
        return {
            "doc_index": idx,
            "text": doc["text"],
            "vector_score": float(vector_score),
            "bm25_score": float(bm25_score),
            "rank": rank,
            "source": source,
            "metadata": doc,
        }

    def retrieve_dense(self, query: str, top_k: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        top_k = top_k or self.candidate_k
        timings = {}
        t_start = time.perf_counter()
        query_emb = self.dense_indexer.encode_query(query)
        timings["embedding_ms"] = (time.perf_counter() - t_start) * 1000.0

        t_start = time.perf_counter()
        scores, indices = self.dense_indexer.search(query_emb, top_k)
        timings["faiss_ms"] = (time.perf_counter() - t_start) * 1000.0
        timings["total_retrieval_ms"] = timings["embedding_ms"] + timings["faiss_ms"]

        candidates = [
            self._candidate_from_index(int(idx), vector_score=float(score), rank=rank, source="dense")
            for rank, (idx, score) in enumerate(zip(indices, scores), 1)
            if 0 <= int(idx) < len(self.rag_documents)
        ]
        return candidates, timings

    def retrieve_bm25(self, query: str, top_k: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        top_k = top_k or self.candidate_k
        t_start = time.perf_counter()
        scores, indices = self.sparse_indexer.search(query, top_k)
        bm25_ms = (time.perf_counter() - t_start) * 1000.0
        candidates = [
            self._candidate_from_index(int(idx), bm25_score=float(score), rank=rank, source="bm25")
            for rank, (idx, score) in enumerate(zip(indices, scores), 1)
            if 0 <= int(idx) < len(self.rag_documents)
        ]
        return candidates, {"bm25_ms": bm25_ms, "total_retrieval_ms": bm25_ms}

    def retrieve_candidates(
        self, query: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Retrieve Top-20 Dense FAISS candidates + Top-20 Sparse BM25 candidates.
        Merges and deduplicates up to max_union_candidates (up to 40).
        Returns candidate list and granular timing stats (in ms).
        """
        timings = {}

        # 1. Embedding Latency
        t_start = time.perf_counter()
        query_emb = self.dense_indexer.encode_query(query)
        timings["embedding_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 2. FAISS Search Latency
        t_start = time.perf_counter()
        vector_scores, vector_indices = self.dense_indexer.search(query_emb, self.candidate_k)
        timings["faiss_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 3. BM25 Search Latency
        t_start = time.perf_counter()
        bm25_scores, bm25_indices = self.sparse_indexer.search(query, self.candidate_k)
        timings["bm25_ms"] = (time.perf_counter() - t_start) * 1000.0

        # 4. Candidate Union & Deduplication
        t_start = time.perf_counter()

        vec_map = {int(idx): float(score) for idx, score in zip(vector_indices, vector_scores)}
        bm25_map = {int(idx): float(score) for idx, score in zip(bm25_indices, bm25_scores)}

        # Union of candidate index sets
        candidate_idx_set = set(vec_map.keys()).union(set(bm25_map.keys()))

        candidates = []
        for idx in candidate_idx_set:
            if idx < 0 or idx >= len(self.rag_documents):
                continue
            v_score = vec_map.get(idx, 0.0)
            b_score = bm25_map.get(idx, 0.0)
            candidates.append(self._candidate_from_index(idx, vector_score=v_score, bm25_score=b_score))

        # Sort candidate pool by maximum normalized component score
        candidates.sort(key=lambda x: max(x["vector_score"], x["bm25_score"]), reverse=True)
        candidates = candidates[: self.max_union_candidates]

        timings["merge_ms"] = (time.perf_counter() - t_start) * 1000.0
        timings["total_retrieval_ms"] = (
            timings["embedding_ms"]
            + timings["faiss_ms"]
            + timings["bm25_ms"]
            + timings["merge_ms"]
        )

        return candidates, timings
