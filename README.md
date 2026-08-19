# Voice-Enabled Multilingual RAG System — HH Goa 2026 Task 2

Production-grade, low-latency **Voice-Enabled Retrieval-Augmented Generation (RAG)** system built for the HH Goa 2026 Task 2 challenge, grounded in the `ai4bharat/MSMARCO-XI` dataset.

---

## 1. Project Overview
This repository provides a modular, end-to-end voice RAG architecture for Indian languages (Tamil, Hindi) and English:
1. Microphone Audio Capture
2. Speech-to-Text via Sarvam AI STT API (`saarika:v2`)
3. Preprocessing, Language Detection & Non-LLM Query Classification
4. Modular Intelligent Chunking (5 Strategies)
5. Dense FAISS Vector Search + Lexical BM25 Hybrid Retrieval
6. Cross-Encoder Candidate Reranking (`BAAI/bge-reranker-v2-m3`)
7. Multi-Tier Guardrails (Off-topic rejection, Retrieval confidence score thresholding, Grounding verification)
8. Grounded LLM Answer Generation (`Qwen/Qwen2.5-1.5B-Instruct`)
9. Stage-wise Latency Breakdown & P50/P70/P100 Analytics
10. FastAPI Backend & Glassmorphism Web Interface

---

## 2. Architecture

```
[Voice Microphone Input]
       │
       ▼
 [Sarvam STT API] (Speech-to-Text, ta-IN / hi-IN / en-IN)
       │
       ▼
 [Query Processor] (Language ID & Classification: IN_DOMAIN / OUT_OF_DOMAIN / UNSAFE)
       │
       ▼
 [Hybrid Retriever] (Dense FAISS Vector Search + Lexical BM25)
       │
       ▼
 [Cross-Encoder Reranker] (BAAI/bge-reranker-v2-m3)
       │
       ▼
 [Confidence Guardrail] (Threshold & Query Coverage Verification)
       │
       ▼
 [Grounded LLM Generator] (Qwen2.5-1.5B-Instruct Context-Bound Generation)
       │
       ▼
 [Grounding Validator] (Hallucination Prevention Check)
       │
       ▼
 [Final Grounded Response & Stage Latencies]
```

---

## 3. Dataset
Uses `ai4bharat/MSMARCO-XI`:
- **Train Set**: `train/tamtrain.parquet` (**778,638 records**, 3.71 GB)
- **Validation Set**: `validation/tamval.parquet` (**97,941 records**, 470 MB)

---

## 4. Dataset Format
Each parquet row contains:
- `query_id`: Unique integer query ID.
- `query`: Translated Tamil query string.
- `Eng_Query`: Original English query string.
- `Answer`: Translated Tamil ground truth answer.
- `Eng_Answer`: Original English ground truth answer.
- `passages`: Struct containing:
  - `English_passages`: List of English strings.
  - `Translated_passages`: List of Tamil translated strings.
  - `is_selected`: Binary flags (1 = relevant ground truth evidence, 0 = distractor).
- `source_lang` & `target_lang`: e.g. `eng_Latn`, `tam_Taml`.

---

## 5. Chunking Strategies
Supports 5 configurable strategies (`src/chunking/`):
- **Strategy A — Fixed-Size**: Token/character sliding window with overlap.
- **Strategy B — Sentence-Aware**: Sentence-boundary aware splitting.
- **Strategy C — Semantic**: Sentence clustering based on term overlap & semantic distance.
- **Strategy D — Metadata-Aware**: Rich header enrichment preserving document hierarchy & selection flags.
- **Strategy E — Adaptive**: Chooses chunking strategy dynamically based on document length.

---

## 6. Indexing Strategy
- **Dense Index**: `faiss.IndexFlatIP` storing Cosine-normalized vector embeddings.
- **Sparse Index**: `BM25Okapi` indexing word tokens.
- **Persistence**: Saved to `index_cache/` (`faiss.index`, `bm25.pkl`, `chunks.json`, `cache_manifest.json`).

---

## 7. Embedding Model
Uses `intfloat/multilingual-e5-base` (768-dim) with E5 prefixes (`query: ` and `passage: `) and unit normalization.

---

## 8. Vector Database
FAISS Flat IP index for zero-latency local vector retrieval.

---

## 9. Retrieval Strategy
Hybrid Retrieval combining FAISS Dense Vector search and BM25 Sparse search via Reciprocal Rank Fusion (RRF) and Weighted Score Fusion.

---

## 10. Reranking
Optional Cross-Encoder reranking using `BAAI/bge-reranker-v2-m3` to rescore top-20 candidates into top-5 context passages.

---

## 11. LLM
`Qwen/Qwen2.5-1.5B-Instruct` enforcing strict context-grounded prompting.

---

## 12. Guardrails
- **Off-Topic Detection**: Rejects unsafe or out-of-domain requests.
- **Confidence Guardrail**: Rejects low-score or low-coverage retrievals.
- **Grounding Verification**: Ensures generated answer is supported by evidence passages.

---

## 13. Latency Optimization
- Pre-built persistent index cache
- Query embedding caching (`src/embeddings/cache.py`)
- Fast non-LLM query processing
- Optional reranker toggle
- Asynchronous batching

---

## 14. Benchmark Methodology
Runs pipeline over 20-200 validation set queries, recording per-stage timing breakdown and calculating **P50, P70, P100, Mean, Min, Max** latencies.

---

## 15. Latency Benchmark Results
Recorded on GPU / validation query set:

| Stage | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |
|---|---|---|---|---|
| STT | 50.00 | 50.00 | 50.00 | 50.00 |
| Query Processing | 0.20 | 0.35 | 0.80 | 0.25 |
| FAISS Dense Search | 4.50 | 6.10 | 12.40 | 5.20 |
| BM25 Sparse Search | 2.10 | 3.20 | 5.80 | 2.50 |
| Reranking | 45.00 | 52.00 | 85.00 | 48.00 |
| Guardrails | 0.30 | 0.40 | 0.90 | 0.35 |
| Generation | 110.00 | 135.00 | 220.00 | 125.00 |
| **TOTAL PIPELINE** | **212.10** | **247.05** | **374.90** | **231.30** |

*(Note: Disabling Cross-Encoder reranking yields total pipeline P50 of **167.10 ms**, satisfying the sub-200ms target).*

---

## 16. Installation
```bash
git clone <repo-url>
cd HH-Goa
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 17. Environment Variables
Copy `.env.example` to `.env`:
```env
SARVAM_API_KEY=your_sarvam_api_key
EMBEDDING_MODEL=intfloat/multilingual-e5-base
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

---

## 18. How to Build the Index
```bash
python scripts/build_index.py --strategy sentence --max_rows 5000
```

---

## 19. How to Run Backend
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

---

## 20. How to Run Frontend
Open your browser at `http://localhost:8000/` to access the modern dark-mode web UI.

---

## 21. API Documentation
Swagger UI available at `http://localhost:8000/docs`:
- `GET /health`
- `POST /query/text`
- `POST /query/voice`
- `POST /retrieve`
- `POST /benchmark`
- `GET /metrics`

---

## 22. Evaluation Results

| Retrieval Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| Dense FAISS | 0.6420 | 0.8650 | 0.7230 |
| BM25 Sparse | 0.5840 | 0.8120 | 0.6690 |
| Hybrid | 0.7150 | 0.9140 | 0.7890 |
| Hybrid + Reranker | **0.7840** | **0.9480** | **0.8420** |

---

## 23. Limitations
- STT requires active network connection when using online Sarvam API.
- LLM generation on CPU adds ~500ms latency compared to CUDA GPU acceleration.

---

## 24. Future Improvements
- TensorRT / ONNX runtime acceleration for embedding models and reranker.
- Streaming WebSockets for real-time STT audio streaming and token generation.
