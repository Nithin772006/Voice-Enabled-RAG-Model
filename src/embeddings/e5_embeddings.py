import time
import torch
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.embeddings.interface import BaseEmbeddingModel
from src.config.settings import EMBEDDING_MODEL_NAME, DEVICE

class MultilingualE5Embedding(BaseEmbeddingModel):
    """Multilingual E5 embedding wrapper optimized for low-latency Indic retrieval."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, device: str = DEVICE):
        super().__init__(model_name)
        self.device = device
        self.model = None

    def load_model(self):
        if self.model is None:
            t0 = time.perf_counter()
            self.model = SentenceTransformer(self.model_name, device=self.device)
            # Pre-warm model with dummy embedding to initialize CUDA kernels
            _ = self.model.encode(["query: warmup"], show_progress_bar=False, normalize_embeddings=True)
            load_sec = time.perf_counter() - t0
            print(f"[EMBEDDINGS] Loaded & pre-warmed {self.model_name} on {self.device} in {load_sec:.2f}s")

    def embed_documents(self, texts: List[str], batch_size: int = 128, show_progress: bool = True) -> np.ndarray:
        self.load_model()
        if not texts:
            return np.empty((0, self.get_dimension()), dtype=np.float32)

        prefixed_texts = [f"passage: {t}" if not t.startswith("passage: ") else t for t in texts]
        
        with torch.inference_mode():
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        self.load_model()
        prefixed_query = f"query: {query}" if not query.startswith("query: ") else query
        
        with torch.inference_mode():
            embedding = self.model.encode(
                prefixed_query,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
        return embedding.astype(np.float32).reshape(1, -1)

    def get_dimension(self) -> int:
        self.load_model()
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def diagnostics(self) -> Dict[str, Any]:
        self.load_model()
        param_cnt = sum(p.numel() for p in self.model.parameters())
        return {
            "model": self.model_name,
            "device": self.device,
            "embedding_dimension": self.get_dimension(),
            "parameter_count": param_cnt
        }
