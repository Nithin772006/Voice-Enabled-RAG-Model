from typing import List, Dict, Any, Tuple
from src.indexing.sparse_index import SparseIndexer

class SparseRetriever:
    """BM25 Lexical retriever."""
    
    def __init__(self, sparse_indexer: SparseIndexer, chunks: List[Dict[str, Any]]):
        self.indexer = sparse_indexer
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 20) -> Tuple[List[Dict[str, Any]], float]:
        indices, scores, search_ms = self.indexer.search(query, top_k=top_k)
        results = []
        max_bm25 = max(scores) if scores and max(scores) > 0 else 1.0
        
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if 0 <= idx < len(self.chunks):
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["bm25_raw_score"] = float(score)
                chunk_copy["bm25_norm_score"] = float(score / max_bm25)
                chunk_copy["sparse_rank"] = rank + 1
                results.append(chunk_copy)
        return results, search_ms
