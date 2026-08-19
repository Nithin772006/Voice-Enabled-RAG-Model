import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Any

from rag_system.config import (
    TRAIN_FILE,
    VAL_FILE,
    DEFAULT_N_TRAIN_ROWS,
    INDEX_CACHE_DIR,
    DEFAULT_GUARDRAIL_THRESHOLD,
    DEFAULT_CHUNK_STRATEGY,
    SENTENCE_MAX_WORDS,
    DEVICE,
)
from rag_system.runtime import (
    configure_utf8_terminal,
    print_unicode_diagnostic,
    print_hf_auth_status,
    cuda_memory_gb,
)
from rag_system.data_loader import load_msmarco_samples, extract_passages
from rag_system.chunking import OriginalPassageChunker, FixedSizeChunker, SentenceAwareChunker, ParagraphStructureChunker
from rag_system.indexing import (
    DenseIndexer,
    SparseIndexer,
    save_pipeline_index,
    load_pipeline_index,
)
from rag_system.retrieval import HybridRetriever
from rag_system.reranker import MultilingualReranker
from rag_system.guardrails import ConfidenceGuardrail
from rag_system.llm import GroundedLLM
from rag_system.eval import calibrate_guardrail_threshold
from rag_system.benchmark import benchmark_latency, print_latency_report


def print_banner():
    print("=" * 60)
    print("CUDA / GPU DIAGNOSTICS")
    print("=" * 60)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    else:
        print("GPU: None (CPU Mode)")
    print_hf_auth_status()
    print("=" * 60)


def _print_memory(label: str):
    mem = cuda_memory_gb()
    if mem["total_gb"] > 0:
        print(f"{label} GPU memory allocated: {mem['allocated_gb']:.2f} GB | reserved: {mem['reserved_gb']:.2f} GB")


def _print_model_diag(name: str, diag: Dict[str, Any], elapsed: float):
    params = diag.get("parameter_count", 0)
    print(f"[SUCCESS] {name} loaded in {elapsed:.2f}s")
    print(f"Model: {diag.get('model')}")
    print(f"Device: {diag.get('device')}")
    print(f"dtype: {diag.get('dtype')}")
    if params:
        print(f"Parameter count: {params:,}")
    if diag.get("embedding_dimension"):
        print(f"Embedding dimension: {diag.get('embedding_dimension')}")


def preload_models(retriever: HybridRetriever, reranker: MultilingualReranker, llm: GroundedLLM, load_llm: bool = True):
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)
    timings = {}

    print("\n[1/3] Loading embedding model...")
    _print_memory("[BEFORE E5]")
    start = time.perf_counter()
    retriever.dense_indexer.load_model()
    elapsed = time.perf_counter() - start
    timings["embedding_model_load_sec"] = elapsed
    _print_model_diag("E5 embedding model", retriever.dense_indexer.diagnostics(), elapsed)
    _print_memory("[AFTER E5]")

    print("\n[2/3] Loading reranker...")
    _print_memory("[BEFORE BGE]")
    start = time.perf_counter()
    reranker.load_model()
    elapsed = time.perf_counter() - start
    timings["reranker_model_load_sec"] = elapsed
    _print_model_diag("BGE reranker", reranker.diagnostics(), elapsed)
    _print_memory("[AFTER BGE]")

    print("\n[3/3] Loading LLM...")
    if load_llm:
        _print_memory("[BEFORE QWEN]")
        start = time.perf_counter()
        llm.load_model()
        elapsed = time.perf_counter() - start
        timings["llm_model_load_sec"] = elapsed
        _print_model_diag("Qwen LLM", llm.diagnostics(), elapsed)
        _print_memory("[AFTER QWEN]")
    else:
        timings["llm_model_load_sec"] = 0.0
        print("[WARNING] Qwen preload skipped by --no-llm-preload. It will load on first supported answer.")

    total = sum(timings.values())
    timings["total_model_load_sec"] = total
    print(f"\n[SUCCESS] Model initialization complete in {total:.2f}s")
    print("=" * 60)
    return timings


def initialize_rag_system(force_rebuild: bool = False, preload: bool = True, preload_llm: bool = True):
    """Load pre-built index cache from disk or build new index from parquet dataset."""
    startup_timings = {}
    print_banner()

    dense_indexer = None
    sparse_indexer = None
    rag_documents = []

    # Check if index cache exists
    if not force_rebuild and os.path.exists(os.path.join(INDEX_CACHE_DIR, "faiss.index")):
        print(f"\n[INFO] Loading pre-built index from disk: {INDEX_CACHE_DIR}...")
        t_start = time.perf_counter()
        try:
            dense_indexer, sparse_indexer, rag_documents = load_pipeline_index(INDEX_CACHE_DIR)
            load_sec = time.perf_counter() - t_start
            startup_timings["index_load_sec"] = load_sec
            print(f"[SUCCESS] Loaded {len(rag_documents)} chunks from cache in {load_sec:.2f}s!")
        except Exception as e:
            print(f"[WARNING] Failed to load cache ({e}).")
            print("[INFO] Rebuild required.")
            force_rebuild = True

    if force_rebuild or dense_indexer is None:
        print(f"\n[INFO] Building new index from {TRAIN_FILE} ({DEFAULT_N_TRAIN_ROWS} rows)...")
        train_samples = load_msmarco_samples(TRAIN_FILE, max_rows=DEFAULT_N_TRAIN_ROWS)

        raw_docs = []
        for row in train_samples:
            query = row.get("query", "")
            answer = row.get("answer", "")
            q_id = row.get("query_id", 0)
            translated_passages, selected_labels = extract_passages(row)

            for i, p_text in enumerate(translated_passages):
                if not p_text:
                    continue
                is_sel = selected_labels[i] if i < len(selected_labels) else 0
                raw_docs.append({
                    "text": p_text,
                    "query": query,
                    "answer": answer,
                    "query_id": q_id,
                    "passage_idx": i,
                    "is_selected": is_sel,
                })

        chunker = get_chunker(DEFAULT_CHUNK_STRATEGY)
        rag_documents = chunker.chunk_documents(raw_docs)
        chunk_texts = [d["text"] for d in rag_documents]

        print(f"[INFO] Total RAG chunks: {len(chunk_texts)}")

        dense_indexer = DenseIndexer()
        print("[INFO] Encoding passages and creating FAISS index...")
        dense_indexer.build_index(chunk_texts, batch_size=32)

        sparse_indexer = SparseIndexer()
        print("[INFO] Building BM25 index...")
        sparse_indexer.build_index(chunk_texts)

        print(f"[INFO] Saving index to disk ({INDEX_CACHE_DIR})...")
        save_pipeline_index(dense_indexer, sparse_indexer, rag_documents, INDEX_CACHE_DIR)
        print("[SUCCESS] Index build and disk persistence complete!")

    retriever = HybridRetriever(dense_indexer, sparse_indexer, rag_documents)
    reranker = MultilingualReranker()
    guardrail = ConfidenceGuardrail(threshold=DEFAULT_GUARDRAIL_THRESHOLD)
    llm = GroundedLLM()

    if preload:
        startup_timings.update(preload_models(retriever, reranker, llm, load_llm=preload_llm))

    return retriever, reranker, guardrail, llm, startup_timings


def get_chunker(name: str):
    name = (name or "sentence").lower()
    if name == "original":
        return OriginalPassageChunker()
    if name == "fixed":
        return FixedSizeChunker()
    if name == "paragraph":
        return ParagraphStructureChunker()
    return SentenceAwareChunker(max_words=SENTENCE_MAX_WORDS)


def answer_query(query: str, retriever: HybridRetriever, reranker: MultilingualReranker, guardrail: ConfidenceGuardrail, llm: GroundedLLM):
    query_start = time.perf_counter()
    candidates, retrieval_timings = retriever.retrieve_candidates(query)
    top_evidence, rerank_ms = reranker.rerank(query, candidates)
    guard_start = time.perf_counter()
    is_passed, top_conf = guardrail.evaluate(top_evidence, query)
    guardrail_ms = (time.perf_counter() - guard_start) * 1000.0

    gen_ms = 0.0
    if is_passed:
        answer, gen_ms = llm.generate_answer(query, top_evidence)
    else:
        answer = guardrail.get_refusal_message(query)

    retrieval_timings["reranking_ms"] = rerank_ms
    retrieval_timings["guardrail_ms"] = guardrail_ms
    retrieval_timings["total_retrieval_with_rerank_ms"] = retrieval_timings["total_retrieval_ms"] + rerank_ms
    retrieval_timings["generation_ms"] = gen_ms
    retrieval_timings["end_to_end_ms"] = (time.perf_counter() - query_start) * 1000.0
    return {
        "answer": answer,
        "evidence": top_evidence,
        "passed_guardrail": is_passed,
        "top_confidence": top_conf,
        "timings": retrieval_timings,
    }


def main():
    configure_utf8_terminal()
    print_unicode_diagnostic()
    force_rebuild = "--rebuild" in sys.argv
    no_llm_preload = "--no-llm-preload" in sys.argv
    retriever, reranker, guardrail, llm, startup_timings = initialize_rag_system(
        force_rebuild=force_rebuild,
        preload=True,
        preload_llm=not no_llm_preload,
    )

    print("\n" + "=" * 75)
    print("SYSTEM READY")
    print("Commands:")
    print(" - Type any question (e.g. 'இந்தியாவின் தலைநகரம் என்ன?')")
    print(" - Type 'eval' or 'benchmark' to run validation set latency benchmark")
    print(" - Type 'exit' or 'quit' to stop")
    print("=" * 75)

    while True:
        try:
            print()
            query = input("Question: ").strip()

            if not query:
                continue

            if query.lower() in ["exit", "quit"]:
                print("\nExiting. Good luck with your Voice-Enabled RAG challenge!")
                break

            if query.lower() == "benchmark":
                print("\n[INFO] Running benchmark on validation queries...")
                val_samples = load_msmarco_samples(VAL_FILE, max_rows=200)
                val_queries = [s["query"] for s in val_samples if s.get("query")]
                bench_res = benchmark_latency(val_queries, retriever, reranker, guardrail, llm=llm, n_runs=15)
                print_latency_report(bench_res)
                continue

            if query.lower() == "eval":
                from rag_system.eval import evaluate_retrieval_table

                print("\n[INFO] Running retrieval evaluation on validation queries...")
                val_samples = load_msmarco_samples(VAL_FILE, max_rows=50)
                table = evaluate_retrieval_table(val_samples, retriever, reranker=reranker)
                print("\n" + "=" * 75)
                print("RETRIEVAL EVALUATION")
                print("=" * 75)
                print(f"{'Metric':<10} | {'Dense FAISS':<12} | {'BM25':<12} | {'Hybrid':<12} | {'Hybrid + Reranker':<18}")
                print("-" * 75)
                for metric in ["Recall@1", "Recall@5", "MRR"]:
                    print(
                        f"{metric:<10} | "
                        f"{table['Dense FAISS'][metric]:<12.4f} | "
                        f"{table['BM25'][metric]:<12.4f} | "
                        f"{table['Hybrid'][metric]:<12.4f} | "
                        f"{table['Hybrid + Reranker'][metric]:<18.4f}"
                    )
                print(f"Evaluated validation queries: {int(table['Hybrid']['Evaluated'])}")
                print("=" * 75)
                continue

            result = answer_query(query, retriever, reranker, guardrail, llm)
            top_evidence = result["evidence"]
            retrieval_timings = result["timings"]
            rerank_ms = retrieval_timings["reranking_ms"]
            total_retrieval_ms = retrieval_timings["total_retrieval_with_rerank_ms"]

            # Display Retrieved Passages
            print("\n" + "=" * 75)
            print(f"TOP {len(top_evidence)} RETRIEVED EVIDENCE PASSAGES")
            print("=" * 75)

            for idx, item in enumerate(top_evidence, 1):
                print(f"\n[{idx}] Reranker Score: {item['reranker_score']:.4f}")
                print(f"    Raw Reranker Score: {item['reranker_raw_score']:.4f}")
                print(f"    Combined Retrieval Score: {item.get('combined_score', 0.0):.4f}")
                print(f"    FAISS Vector Cosine Score: {item['vector_score']:.4f}")
                print(f"    BM25 Normalized Score:     {item['bm25_score']:.4f}")
                print(f"    Passage Text: {item['text']}")

            # ============================================================
            # 3. CONFIDENCE GUARDRAIL CHECK
            # ============================================================
            print("\n" + "=" * 75)
            print("CONFIDENCE GUARDRAIL EVALUATION")
            print("=" * 75)

            is_passed = result["passed_guardrail"]
            top_conf = result["top_confidence"]
            print(f"Guardrail Threshold: {guardrail.threshold:.4f}")
            print(f"Calibrated Evidence Confidence: {top_conf:.4f}")
            if top_evidence:
                signals = top_evidence[0].get("guardrail_signals", {})
                if signals:
                    print(f"Score Margin: {signals.get('score_margin', 0.0):.4f}")
                    print(f"Query Coverage: {signals.get('query_coverage', 0.0):.4f}")
                    print(f"Anchor Overlap: {signals.get('anchor_overlap', 0)}")

            if is_passed:
                print("STATUS: [SUPPORTED]")
            else:
                print("STATUS: [INSUFFICIENT_EVIDENCE]")

            # ============================================================
            # 4. GROUNDED GENERATION / CONTROLLED REFUSAL
            # ============================================================
            print("\n" + "=" * 75)
            print("GROUNDED ANSWER")
            print("=" * 75)

            gen_ms = retrieval_timings["generation_ms"]
            answer = result["answer"]
            print(answer)

            # ============================================================
            # 5. LATENCY BREAKDOWN
            # ============================================================
            print("\n" + "=" * 75)
            print("PER-QUERY LATENCY BREAKDOWN (ms)")
            print("=" * 75)
            print(f"  Embedding Latency       : {retrieval_timings['embedding_ms']:8.2f} ms")
            print(f"  FAISS Search Latency    : {retrieval_timings['faiss_ms']:8.2f} ms")
            print(f"  BM25 Search Latency     : {retrieval_timings['bm25_ms']:8.2f} ms")
            print(f"  Reranking Latency       : {rerank_ms:8.2f} ms")
            print(f"  Guardrail Latency       : {retrieval_timings['guardrail_ms']:8.2f} ms")
            print(f"  Total Retrieval Latency : {total_retrieval_ms:8.2f} ms")
            print(f"  LLM Generation Latency  : {gen_ms:8.2f} ms")
            print(f"  -------------------------------------------")
            print(f"  Total System Pipeline   : {retrieval_timings['end_to_end_ms']:8.2f} ms")
            print("=" * 75)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()
