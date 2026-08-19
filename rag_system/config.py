import os
import torch

BASE_DIR = r"D:\New folder\HH Goa"
TRAIN_FILE = os.path.join(BASE_DIR, "train", "tamtrain.parquet")
VAL_FILE = os.path.join(BASE_DIR, "validation", "tamval.parquet")
INDEX_CACHE_DIR = os.path.join(BASE_DIR, "index_cache")

# Dataset & Batching
DEFAULT_N_TRAIN_ROWS = 5000
PARQUET_BATCH_SIZE = 500
DATASET_NAME = "ai4bharat/MSMARCO-XI"
DATASET_SPLIT = "train"
DATASET_LANGUAGE = "tam"

# Models
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Index/cache compatibility
INDEX_VERSION = "2"
CACHE_MANIFEST_FILE = "cache_config.json"

# Chunking
DEFAULT_CHUNK_STRATEGY = "sentence"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20
SENTENCE_MAX_WORDS = 100
PARAGRAPH_MAX_WORDS = 120

# Retrieval & Reranking settings
CANDIDATE_K = 20          # Retrieves top-20 from FAISS and top-20 from BM25
MAX_UNION_CANDIDATES = 40  # Up to 40 candidates after deduplication
TOP_EVIDENCE_K = 5        # Top 5 passages passed to LLM
RERANKER_BM25_FUSION_WEIGHT = 0.35
RERANKER_DENSE_FUSION_WEIGHT = 0.05

# Guardrail defaults. Calibrate these with validation data before scaling.
DEFAULT_GUARDRAIL_THRESHOLD = 0.65
GUARDRAIL_MIN_MARGIN = 0.05
GUARDRAIL_MIN_QUERY_COVERAGE = 0.50
GUARDRAIL_MIN_ANCHOR_OVERLAP = 2

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
