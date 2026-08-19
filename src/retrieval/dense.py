from typing import List, Dict, Any, Tuple
from src.indexing.dense_index import DenseIndexer

class DenseRetriever:
    """Dense vector retriever using FAISS."""
    
    def __init__(self, dense_indexer: DenseIndexer, chunks: List[Dict[str, Any]]):
        self.indexer = dense_indexer
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 20) -> Tuple[List[Dict[str, Any]], float, float]:
        indices, scores, embed_ms, faiss_search_ms = self.indexer.search(query, top_k=top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if 0 <= idx < len(self.chunks):
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["vector_score"] = float(score)
                chunk_copy["dense_rank"] = rank + 1
                results.append(chunk_copy)
        return results, embed_ms, faiss_search_ms
