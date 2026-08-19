import os
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_FILE = os.environ.get("TRAIN_FILE", os.path.join(BASE_DIR, "train", "tamtrain.parquet"))
VAL_FILE = os.environ.get("VAL_FILE", os.path.join(BASE_DIR, "validation", "tamval.parquet"))
INDEX_CACHE_DIR = os.environ.get("INDEX_CACHE_DIR", os.path.join(BASE_DIR, "index_cache"))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# API Keys
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")

# Default Model Selection
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Defaults
DEFAULT_N_TRAIN_ROWS = int(os.environ.get("DEFAULT_N_TRAIN_ROWS", "5000"))
PARQUET_BATCH_SIZE = int(os.environ.get("PARQUET_BATCH_SIZE", "500"))

# Retrieval settings
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "5"))
DEFAULT_CANDIDATE_K = int(os.environ.get("CANDIDATE_K", "20"))
DEFAULT_SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.50"))
DEFAULT_RERANKING_ENABLED = os.environ.get("RERANKING_ENABLED", "true").lower() in ["true", "1", "yes"]

# Guardrail settings
DEFAULT_GUARDRAIL_THRESHOLD = float(os.environ.get("GUARDRAIL_THRESHOLD", "0.65"))

# Device configuration
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
