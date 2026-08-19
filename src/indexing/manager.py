import os
import json
import time
from typing import List, Dict, Any, Tuple
from src.indexing.dense_index import DenseIndexer
from src.indexing.sparse_index import SparseIndexer
from src.chunking.evaluator import get_chunker
from src.config.settings import INDEX_CACHE_DIR

class IndexManager:
    """Unified Index Manager for building, persisting, loading, and updating FAISS & BM25 indexes."""
    
    def __init__(self, cache_dir: str = INDEX_CACHE_DIR):
        self.cache_dir = cache_dir
        self.dense_indexer = DenseIndexer()
        self.sparse_indexer = SparseIndexer()
        self.chunks: List[Dict[str, Any]] = []

    def build_and_save(
        self,
        records: List[Dict[str, Any]],
        chunk_strategy: str = "sentence",
        batch_size: int = 32
    ) -> Tuple[int, float]:
        t0 = time.perf_counter()
        print(f"[INDEX MANAGER] Chunking {len(records)} dataset records using strategy '{chunk_strategy}'...")
        chunker = get_chunker(chunk_strategy)
        self.chunks = chunker.chunk_records(records)
        
        texts = [c["text"] for c in self.chunks]
        print(f"[INDEX MANAGER] Created {len(texts)} total text chunks.")

        print("[INDEX MANAGER] Building FAISS Dense Vector index...")
        self.dense_indexer.build_index(texts, batch_size=batch_size)

        print("[INDEX MANAGER] Building BM25 Sparse Lexical index...")
        self.sparse_indexer.build_index(texts)

        self.save_cache()
        elapsed_sec = time.perf_counter() - t0
        return len(self.chunks), elapsed_sec

    def save_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        faiss_path = os.path.join(self.cache_dir, "faiss.index")
        bm25_path = os.path.join(self.cache_dir, "bm25.pkl")
        chunks_path = os.path.join(self.cache_dir, "chunks.json")
        manifest_path = os.path.join(self.cache_dir, "cache_manifest.json")

        self.dense_indexer.save(faiss_path)
        self.sparse_indexer.save(bm25_path)

        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        manifest = {
            "total_chunks": len(self.chunks),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "faiss_path": faiss_path,
            "bm25_path": bm25_path,
            "chunks_path": chunks_path
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"[INDEX MANAGER] Successfully persisted complete index cache to {self.cache_dir}")

    def load_cache(self) -> bool:
        faiss_path = os.path.join(self.cache_dir, "faiss.index")
        bm25_path = os.path.join(self.cache_dir, "bm25.pkl")
        chunks_path = os.path.join(self.cache_dir, "chunks.json")

        if not (os.path.exists(faiss_path) and os.path.exists(bm25_path) and os.path.exists(chunks_path)):
            return False

        t0 = time.perf_counter()
        print(f"[INDEX MANAGER] Loading cached index from {self.cache_dir}...")
        self.dense_indexer.load(faiss_path)
        self.sparse_indexer.load(bm25_path)

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        elapsed_sec = time.perf_counter() - t0
        print(f"[INDEX MANAGER] Cache loaded successfully in {elapsed_sec:.2f}s ({len(self.chunks)} chunks)")
        return True
