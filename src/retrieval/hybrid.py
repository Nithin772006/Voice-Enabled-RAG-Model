import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from src.indexing.dense_index import DenseIndexer
from src.indexing.sparse_index import SparseIndexer
from src.retrieval.dense import DenseRetriever
from src.retrieval.sparse import SparseRetriever

def min_max_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s - min_s < 1e-6:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]

class HybridRetriever:
    """Hybrid Retriever combining Dense Vector FAISS search and Sparse BM25 lexical search with Min-Max Score Normalization."""
    
    def __init__(
        self,
        dense_indexer: DenseIndexer,
        sparse_indexer: SparseIndexer,
        chunks: List[Dict[str, Any]],
        alpha: float = 0.70,
        rrf_k: int = 60
    ):
        self.dense_retriever = DenseRetriever(dense_indexer, chunks)
        self.sparse_retriever = SparseRetriever(sparse_indexer, chunks)
        self.chunks = chunks
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        candidate_k: int = 20,
        top_k: int = 5,
        alpha: Optional[float] = None,
        lang_filter: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
        t0 = time.perf_counter()
        use_alpha = alpha if alpha is not None else self.alpha
        
        # 1. Dense retrieval
        dense_results, embed_ms, faiss_search_ms = self.dense_retriever.retrieve(query, top_k=candidate_k)
        
        # 2. Sparse retrieval
        if use_alpha < 1.0:
            sparse_results, sparse_ms = self.sparse_retriever.retrieve(query, top_k=candidate_k)
        else:
            sparse_results, sparse_ms = [], 0.0

        # Extract raw scores for normalization
        dense_raw_scores = [c.get("vector_score", 0.0) for c in dense_results]
        sparse_raw_scores = [c.get("bm25_raw_score", 0.0) for c in sparse_results]

        dense_norm_scores = min_max_normalize(dense_raw_scores)
        sparse_norm_scores = min_max_normalize(sparse_raw_scores)

        candidate_map: Dict[str, Dict[str, Any]] = {}

        # Track debug info
        debug_faiss = []
        for res, raw_s, norm_s in zip(dense_results, dense_raw_scores, dense_norm_scores):
            c_id = res["chunk_id"]
            rank = res["dense_rank"]
            rrf = 1.0 / (self.rrf_k + rank)
            
            cand = dict(res)
            cand["vector_norm_score"] = float(norm_s)
            cand["rrf_score"] = rrf * use_alpha
            cand["combined_score"] = float(use_alpha * norm_s)
            candidate_map[c_id] = cand

            debug_faiss.append({
                "chunk_id": c_id,
                "original_record_id": res.get("original_record_id"),
                "raw_vector_score": round(float(raw_s), 4),
                "norm_vector_score": round(float(norm_s), 4),
                "rank": rank
            })

        debug_bm25 = []
        for res, raw_s, norm_s in zip(sparse_results, sparse_raw_scores, sparse_norm_scores):
            c_id = res["chunk_id"]
            rank = res["sparse_rank"]
            rrf = 1.0 / (self.rrf_k + rank)

            debug_bm25.append({
                "chunk_id": c_id,
                "original_record_id": res.get("original_record_id"),
                "raw_bm25_score": round(float(raw_s), 4),
                "norm_bm25_score": round(float(norm_s), 4),
                "rank": rank
            })

            if c_id in candidate_map:
                cand = candidate_map[c_id]
                cand["bm25_norm_score"] = float(norm_s)
                cand["rrf_score"] += rrf * (1.0 - use_alpha)
                cand["combined_score"] += float((1.0 - use_alpha) * norm_s)
                cand["sparse_rank"] = rank
            else:
                cand = dict(res)
                cand["vector_score"] = 0.0
                cand["vector_norm_score"] = 0.0
                cand["bm25_norm_score"] = float(norm_s)
                cand["rrf_score"] = rrf * (1.0 - use_alpha)
                cand["combined_score"] = float((1.0 - use_alpha) * norm_s)
                cand["dense_rank"] = 999
                candidate_map[c_id] = cand

        candidates = list(candidate_map.values())

        if lang_filter:
            filtered = [c for c in candidates if c.get("language") == lang_filter]
            if filtered:
                candidates = filtered

        # Sort candidates by combined score descending
        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        top_candidates = candidates[:top_k]

        total_retrieval_ms = (time.perf_counter() - t0) * 1000.0

        timings = {
            "embedding_ms": embed_ms,
            "dense_search_ms": faiss_search_ms,
            "sparse_search_ms": sparse_ms,
            "total_retrieval_ms": total_retrieval_ms
        }

        debug_merged = [
            {
                "chunk_id": c["chunk_id"],
                "original_record_id": c.get("original_record_id"),
                "combined_score": round(float(c["combined_score"]), 4),
                "vector_score": round(float(c.get("vector_score", 0.0)), 4),
                "bm25_norm_score": round(float(c.get("bm25_norm_score", 0.0)), 4)
            }
            for c in top_candidates
        ]

        retrieval_debug = {
            "query": query,
            "alpha": use_alpha,
            "faiss_candidates": debug_faiss,
            "bm25_candidates": debug_bm25,
            "merged_candidates": debug_merged
        }

        return top_candidates, timings, retrieval_debug
