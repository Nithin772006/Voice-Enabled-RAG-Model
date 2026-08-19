import time
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config.settings import RERANKER_MODEL_NAME, DEVICE

class MultilingualReranker:
    """Optional Cross-Encoder Reranker using BAAI/bge-reranker-v2-m3 optimized for low latency."""
    
    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is None:
            t0 = time.perf_counter()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            load_sec = time.perf_counter() - t0
            print(f"[RERANKER] Preloaded {self.model_name} on {self.device} in {load_sec:.2f}s")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
        max_rerank_candidates: int = 5,
        enabled: bool = True
    ) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
        if not candidates or not enabled:
            debug_info = [{"chunk_id": c.get("chunk_id"), "reranker_score": 0.0} for c in candidates[:top_k]]
            return candidates[:top_k], 0.0, debug_info

        t0 = time.perf_counter()
        self.load_model()

        # Restrict candidate pool to top N to maintain low latency
        input_candidates = candidates[:max_rerank_candidates]

        pairs = [[query, cand["text"]] for cand in input_candidates]
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).cpu().float().numpy()

        sigmoid_scores = 1.0 / (1.0 + np.exp(-scores))

        reranked = []
        for cand, raw_score, norm_score in zip(input_candidates, scores, sigmoid_scores):
            cand_copy = dict(cand)
            cand_copy["reranker_raw_score"] = float(raw_score)
            cand_copy["reranker_score"] = float(norm_score)
            reranked.append(cand_copy)

        reranked.sort(key=lambda x: x["reranker_score"], reverse=True)
        top_reranked = reranked[:top_k]

        rerank_ms = (time.perf_counter() - t0) * 1000.0

        debug_reranked = [
            {
                "chunk_id": c["chunk_id"],
                "original_record_id": c.get("original_record_id"),
                "reranker_score": round(float(c["reranker_score"]), 4),
                "raw_score": round(float(c["reranker_raw_score"]), 4)
            }
            for c in top_reranked
        ]

        return top_reranked, rerank_ms, debug_reranked

    def diagnostics(self) -> Dict[str, Any]:
        self.load_model()
        param_cnt = sum(p.numel() for p in self.model.parameters())
        return {
            "model": self.model_name,
            "device": self.device,
            "parameter_count": param_cnt
        }
