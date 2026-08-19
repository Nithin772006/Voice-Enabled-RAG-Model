import os
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional

from src.api.models import TextQueryRequest, RetrieveOnlyRequest, BenchmarkRequest, QueryResponse
from src.pipeline.orchestrator import VoiceRAGPipeline
from src.benchmark.latency import benchmark_pipeline_latency
from src.data.normalize import load_and_normalize_dataset
from src.config.settings import TRAIN_FILE, VAL_FILE, DEFAULT_N_TRAIN_ROWS, BASE_DIR

app = FastAPI(
    title="HH Goa 2026 Task 2 — Voice-Enabled Multilingual RAG API",
    description="Low-latency voice & text retrieval-augmented generation engine grounded in MSMARCO-XI.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = VoiceRAGPipeline()

@app.on_event("startup")
def startup_event():
    print("[FASTAPI STARTUP] Initializing Voice RAG Pipeline...")
    loaded = pipeline.initialize_system(force_rebuild=False)
    if not loaded:
        print("[FASTAPI STARTUP] Cache not found. Loading train dataset samples to build initial index...")
        try:
            records = load_and_normalize_dataset(TRAIN_FILE, max_rows=DEFAULT_N_TRAIN_ROWS)
            pipeline.initialize_system(force_rebuild=True, records=records)
        except Exception as e:
            print(f"[FASTAPI STARTUP WARNING] Could not auto-build index: {e}")

@app.get("/health")
def health_check():
    index_ready = pipeline.retriever is not None
    chunk_count = len(pipeline.index_manager.chunks) if pipeline.index_manager else 0
    return {
        "status": "HEALTHY",
        "index_initialized": index_ready,
        "total_chunks_indexed": chunk_count,
        "device": pipeline.llm.device if hasattr(pipeline, "llm") else "cpu"
    }

@app.post("/query/text", response_model=QueryResponse)
def query_text(req: TextQueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        response = pipeline.run_text_query(
            query=req.query,
            mode=req.mode or "fast",
            top_k=req.top_k or 5,
            candidate_k=req.candidate_k or 20,
            alpha=req.alpha if req.alpha is not None else 0.70
        )
        total_ms = response.get("latencies_ms", {}).get("total_latency_ms", 0.0)
        mode_str = response.get("mode", "fast").upper()
        ans_mode = response.get("answer_mode", "").upper()
        print(f"[RAG API] Mode: {mode_str} | Query: '{req.query}' | Answer Mode: {ans_mode} | TOTAL LATENCY: {total_ms:.2f} ms")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/voice", response_model=QueryResponse)
async def query_voice(
    file: UploadFile = File(...),
    mode: Optional[str] = Form("fast"),
    top_k: Optional[int] = Form(5),
    candidate_k: Optional[int] = Form(20),
    alpha: Optional[float] = Form(0.70)
):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file uploaded.")

        response = pipeline.run_voice_query(
            audio_bytes=audio_bytes,
            filename=file.filename or "recording.wav",
            mode=mode or "fast",
            top_k=top_k or 5,
            candidate_k=candidate_k or 20,
            alpha=alpha if alpha is not None else 0.70
        )
        total_ms = response.get("latencies_ms", {}).get("total_latency_ms", 0.0)
        transcript = response.get("transcript", "")
        print(f"[RAG VOICE API] Transcript: '{transcript}' | TOTAL LATENCY: {total_ms:.2f} ms")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve")
def retrieve_candidates(req: RetrieveOnlyRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    if pipeline.retriever is None:
        raise HTTPException(status_code=500, detail="Retriever index not initialized.")

    t0 = time.perf_counter()
    candidates, ret_timings, ret_debug = pipeline.retriever.retrieve(
        req.query,
        candidate_k=req.candidate_k or 20,
        top_k=req.top_k or 5,
        alpha=req.alpha if req.alpha is not None else 0.70
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "query": req.query,
        "candidate_count": len(candidates),
        "candidates": pipeline._format_sources(candidates),
        "retrieval_timings_ms": ret_timings,
        "retrieval_debug": ret_debug,
        "total_latency_ms": round(elapsed_ms, 2)
    }

@app.post("/benchmark")
def run_benchmark(req: BenchmarkRequest):
    if not os.path.exists(VAL_FILE):
        raise HTTPException(status_code=404, detail="Validation dataset parquet file not found.")

    try:
        val_records = load_and_normalize_dataset(VAL_FILE, max_rows=req.num_queries or 20)
        val_queries = [r["query"] for r in val_records if r.get("query")]

        if not val_queries:
            raise HTTPException(status_code=400, detail="No valid queries found in validation sample.")

        report = benchmark_pipeline_latency(
            pipeline,
            val_queries,
            mode=req.mode or "fast"
        )
        return {
            "num_evaluated_queries": len(val_queries),
            "mode": req.mode or "fast",
            "latency_percentiles_ms": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    chunk_count = len(pipeline.index_manager.chunks) if pipeline.index_manager else 0
    return {
        "system": "Voice-Enabled Multilingual RAG",
        "dataset": "ai4bharat/MSMARCO-XI",
        "total_chunks_indexed": chunk_count,
        "embedding_model": pipeline.index_manager.dense_indexer.embedding_model.model_name if pipeline.index_manager else "intfloat/multilingual-e5-base",
        "reranker_model": pipeline.reranker.model_name,
        "llm_model": pipeline.llm.model_name,
        "device": pipeline.llm.device
    }

frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
