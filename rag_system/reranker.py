import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder

from rag_system.config import (
    RERANKER_MODEL_NAME,
    DEVICE,
    TOP_EVIDENCE_K,
    RERANKER_BM25_FUSION_WEIGHT,
    RERANKER_DENSE_FUSION_WEIGHT,
)
from rag_system.runtime import first_parameter_info, parameter_count


class MultilingualReranker:
    """BAAI/bge-reranker-v2-m3 Cross-Encoder Reranker."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        device: str = DEVICE,
        top_k: int = TOP_EVIDENCE_K,
    ):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k
        self.model = None

    def _load_model(self):
        if self.model is None:
            # Load BAAI/bge-reranker-v2-m3 CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device,
            )
            if self.device == "cuda":
                self.model.model.half()
            if hasattr(self.model, "model") and hasattr(self.model.model, "eval"):
                self.model.model.eval()
            # Warmup prediction to initialize CUDA kernels
            with torch.no_grad():
                _ = self.model.predict([["warmup query", "warmup passage"]], show_progress_bar=False)

    def load_model(self):
        self._load_model()
        return self.model

    def diagnostics(self) -> Dict[str, Any]:
        self._load_model()
        info = first_parameter_info(self.model.model)
        return {
            "model": self.model_name,
            "device": info["device"],
            "dtype": info["dtype"],
            "parameter_count": parameter_count(self.model.model),
        }

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Rerank query + passage candidate pairs using actual cross-attention.
        Returns top-k reranked evidence passages and reranking latency in ms.
        """
        if not candidates:
            return [], 0.0

        self._load_model()
        t_start = time.perf_counter()

        pairs = [[query, c["text"]] for c in candidates]

        # Compute cross-encoder raw logits
        with torch.no_grad():
            raw_scores = self.model.predict(pairs, batch_size=32, show_progress_bar=False)

        raw_scores = np.asarray(raw_scores, dtype=np.float32)

        # Sigmoid score is an uncalibrated reranker score, not a probability.
        sigmoid_scores = 1.0 / (1.0 + np.exp(-raw_scores))

        scored_candidates = []
        for i, c in enumerate(candidates):
            c_copy = dict(c)
            c_copy["reranker_raw_score"] = float(raw_scores[i])
            c_copy["reranker_score"] = float(sigmoid_scores[i])
            c_copy["combined_score"] = float(
                sigmoid_scores[i]
                + RERANKER_BM25_FUSION_WEIGHT * c.get("bm25_score", 0.0)
                + RERANKER_DENSE_FUSION_WEIGHT * c.get("vector_score", 0.0)
            )
            scored_candidates.append(c_copy)

        # Keep cross-encoder as the main signal, but reward exact lexical evidence.
        scored_candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        rerank_ms = (time.perf_counter() - t_start) * 1000.0

        top_evidence = scored_candidates[: self.top_k]

        return top_evidence, rerank_ms
