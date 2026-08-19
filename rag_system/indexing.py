import os
import json
import pickle
import hashlib
import numpy as np
import faiss
import torch
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from rag_system.config import (
    EMBEDDING_MODEL_NAME,
    DEVICE,
    INDEX_CACHE_DIR,
    CACHE_MANIFEST_FILE,
    DATASET_NAME,
    DATASET_LANGUAGE,
    DATASET_SPLIT,
    DEFAULT_N_TRAIN_ROWS,
    DEFAULT_CHUNK_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SENTENCE_MAX_WORDS,
    PARAGRAPH_MAX_WORDS,
    INDEX_VERSION,
)
from rag_system.runtime import first_parameter_info, parameter_count


class DenseIndexer:
    """FAISS Dense Vector Indexer using intfloat/multilingual-e5-base."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.index = None
        self.dimension = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            if self.device == "cuda":
                self.model.half()
            self.model.eval()
            # Warmup pass to allocate CUDA kernels during startup
            _ = self.model.encode(["query: warmup"], normalize_embeddings=True, convert_to_numpy=True)
            if self.dimension is None and hasattr(self.model, "get_sentence_embedding_dimension"):
                try:
                    self.dimension = int(self.model.get_sentence_embedding_dimension())
                except Exception:
                    self.dimension = 768

    def load_model(self):
        self._load_model()
        return self.model

    def diagnostics(self) -> Dict[str, Any]:
        self._load_model()
        info = first_parameter_info(self.model)
        dim = self.dimension or (self.index.d if self.index is not None else 768)
        return {
            "model": self.model_name,
            "device": info["device"],
            "dtype": info["dtype"],
            "parameter_count": parameter_count(self.model),
            "embedding_dimension": dim,
        }

    def build_index(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self._load_model()
        formatted_passages = ["passage: " + t for t in texts]

        embeddings = self.model.encode(
            formatted_passages,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(embeddings, dtype="float32")
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        self._load_model()
        query_embedding = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(query_embedding, dtype="float32")

    def search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("FAISS index has not been initialized or loaded.")
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        return scores[0], indices[0]

    def save(self, filepath: str):
        if self.index is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            faiss.write_index(self.index, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"FAISS index file not found: {filepath}")
        self.index = faiss.read_index(filepath)
        self.dimension = self.index.d


class SparseIndexer:
    """BM25 Sparse Indexer."""

    def __init__(self):
        self.bm25 = None
        self.corpus_size = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.lower().split()

    def build_index(self, texts: List[str]):
        tokenized_corpus = [self.tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_size = len(tokenized_corpus)

    def search(self, query: str, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.bm25 is None:
            raise ValueError("BM25 index has not been initialized or loaded.")
        tokenized_query = self.tokenize(query)
        scores = np.asarray(self.bm25.get_scores(tokenized_query), dtype=np.float32)

        # Min-max normalization
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s > 1e-8:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(scores)

        top_indices = np.argsort(norm_scores)[::-1][:top_k]
        top_scores = norm_scores[top_indices]
        return top_scores, top_indices

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"BM25 index file not found: {filepath}")
        with open(filepath, "rb") as f:
            self.bm25 = pickle.load(f)
        self.corpus_size = self.bm25.corpus_size


def build_cache_manifest(
    rag_documents: List[Dict[str, Any]],
    *,
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    n_train_rows: int = DEFAULT_N_TRAIN_ROWS,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    legacy_assumed: bool = False,
) -> Dict[str, Any]:
    sample = "|".join(d.get("text", "")[:200] for d in rag_documents[:25])
    fingerprint = hashlib.sha256(sample.encode("utf-8")).hexdigest()
    return {
        "index_version": INDEX_VERSION,
        "dataset_name": DATASET_NAME,
        "dataset_language": DATASET_LANGUAGE,
        "dataset_split": DATASET_SPLIT,
        "n_train_rows": n_train_rows,
        "embedding_model_name": embedding_model_name,
        "chunk_strategy": chunk_strategy,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "sentence_max_words": SENTENCE_MAX_WORDS,
        "paragraph_max_words": PARAGRAPH_MAX_WORDS,
        "document_count": len(rag_documents),
        "document_fingerprint": fingerprint,
        "legacy_assumed": legacy_assumed,
    }


def expected_cache_manifest() -> Dict[str, Any]:
    return {
        "index_version": INDEX_VERSION,
        "dataset_name": DATASET_NAME,
        "dataset_language": DATASET_LANGUAGE,
        "dataset_split": DATASET_SPLIT,
        "n_train_rows": DEFAULT_N_TRAIN_ROWS,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "chunk_strategy": DEFAULT_CHUNK_STRATEGY,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "sentence_max_words": SENTENCE_MAX_WORDS,
        "paragraph_max_words": PARAGRAPH_MAX_WORDS,
    }


def validate_cache_manifest(manifest: Dict[str, Any], rag_documents: Optional[List[Dict[str, Any]]] = None) -> Tuple[bool, List[str]]:
    expected = expected_cache_manifest()
    reasons = []
    for key, expected_value in expected.items():
        actual = manifest.get(key)
        if actual != expected_value:
            reasons.append(f"{key}: cache={actual!r}, expected={expected_value!r}")

    if rag_documents is not None:
        doc_count = manifest.get("document_count")
        if doc_count is not None and int(doc_count) != len(rag_documents):
            reasons.append(f"document_count: cache={doc_count}, loaded={len(rag_documents)}")

    return not reasons, reasons


def save_pipeline_index(
    dense_indexer: DenseIndexer,
    sparse_indexer: SparseIndexer,
    rag_documents: List[Dict[str, Any]],
    cache_dir: str = INDEX_CACHE_DIR,
    manifest: Optional[Dict[str, Any]] = None,
):
    """Save FAISS index, BM25 index, document metadata, and cache manifest to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    dense_path = os.path.join(cache_dir, "faiss.index")
    sparse_path = os.path.join(cache_dir, "bm25.pkl")
    meta_path = os.path.join(cache_dir, "metadata.json")
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_FILE)

    dense_indexer.save(dense_path)
    sparse_indexer.save(sparse_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(rag_documents, f, ensure_ascii=False, indent=2)

    manifest = manifest or build_cache_manifest(rag_documents)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def load_pipeline_index(
    cache_dir: str = INDEX_CACHE_DIR,
) -> Tuple[DenseIndexer, SparseIndexer, List[Dict[str, Any]]]:
    """Load FAISS index, BM25 index, and metadata JSON from disk."""
    dense_path = os.path.join(cache_dir, "faiss.index")
    sparse_path = os.path.join(cache_dir, "bm25.pkl")
    meta_path = os.path.join(cache_dir, "metadata.json")
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_FILE)

    if not (os.path.exists(dense_path) and os.path.exists(sparse_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Index cache files missing in {cache_dir}")

    dense_indexer = DenseIndexer()
    dense_indexer.load(dense_path)

    sparse_indexer = SparseIndexer()
    sparse_indexer.load(sparse_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        rag_documents = json.load(f)

    if dense_indexer.index.ntotal != len(rag_documents):
        raise ValueError(
            f"Cache is corrupted: FAISS has {dense_indexer.index.ntotal} vectors but metadata has {len(rag_documents)} documents."
        )
    if sparse_indexer.corpus_size != len(rag_documents):
        raise ValueError(
            f"Cache is corrupted: BM25 has {sparse_indexer.corpus_size} documents but metadata has {len(rag_documents)} documents."
        )

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        valid, reasons = validate_cache_manifest(manifest, rag_documents)
        if not valid:
            reason_text = "; ".join(reasons)
            raise ValueError(f"Index cache is incompatible and must be rebuilt: {reason_text}")
    else:
        manifest = build_cache_manifest(rag_documents, legacy_assumed=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print("[WARNING] Cache manifest was missing. Accepted legacy cache after FAISS/BM25/metadata consistency checks.")

    return dense_indexer, sparse_indexer, rag_documents
