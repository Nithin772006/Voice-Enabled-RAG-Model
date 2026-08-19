import time
import numpy as np
from typing import List, Dict, Any
from rag_system.retrieval import HybridRetriever
from rag_system.reranker import MultilingualReranker
from rag_system.guardrails import ConfidenceGuardrail
from rag_system.llm import GroundedLLM


def benchmark_latency(
    queries: List[str],
    retriever: HybridRetriever,
    reranker: MultilingualReranker,
    guardrail: ConfidenceGuardrail,
    llm: GroundedLLM = None,
    n_runs: int = 20,
    warmup: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Run latency benchmark over queries and measure P50, P70, P100 percentiles for:
    - embedding latency
    - FAISS latency
    - BM25 latency
    - reranking latency
    - total retrieval latency
    - LLM generation latency (if LLM passed)
    """
    sample_queries = queries[:n_runs] if len(queries) >= n_runs else queries

    embedding_latencies = []
    faiss_latencies = []
    bm25_latencies = []
    rerank_latencies = []
    guardrail_latencies = []
    total_retrieval_latencies = []
    llm_latencies = []
    end_to_end_latencies = []

    print(f"\nRunning latency benchmark over {len(sample_queries)} test queries...")

    if warmup and sample_queries:
        warmup_query = next((q for q in sample_queries if q), None)
        if warmup_query:
            print("Warming up embedding model and reranker outside measured samples...")
            warm_candidates, _ = retriever.retrieve_candidates(warmup_query)
            warm_top, _ = reranker.rerank(warmup_query, warm_candidates)
            guardrail.evaluate(warm_top, warmup_query)

    for query in sample_queries:
        if not query:
            continue

        q_start = time.perf_counter()

        # Retrieval stage
        candidates, timings = retriever.retrieve_candidates(query)

        embedding_latencies.append(timings["embedding_ms"])
        faiss_latencies.append(timings["faiss_ms"])
        bm25_latencies.append(timings["bm25_ms"])

        # Reranking stage
        top_evidence, rerank_ms = reranker.rerank(query, candidates)
        rerank_latencies.append(rerank_ms)

        total_retrieval_ms = timings["total_retrieval_ms"] + rerank_ms
        total_retrieval_latencies.append(total_retrieval_ms)

        # Guardrail stage
        g_start = time.perf_counter()
        is_passed, _ = guardrail.evaluate(top_evidence, query)
        guard_ms = (time.perf_counter() - g_start) * 1000.0
        guardrail_latencies.append(guard_ms)

        # LLM Generation stage
        gen_ms = 0.0
        if is_passed and llm is not None:
            _, gen_ms = llm.generate_answer(query, top_evidence)
            llm_latencies.append(gen_ms)

        e2e_ms = (time.perf_counter() - q_start) * 1000.0
        end_to_end_latencies.append(e2e_ms)

    def calc_percentiles(lat_list: List[float]) -> Dict[str, float]:
        if not lat_list:
            return {"P50": 0.0, "P70": 0.0, "P100": 0.0}
        arr = np.array(lat_list, dtype=np.float32)
        return {
            "P50": float(np.percentile(arr, 50)),
            "P70": float(np.percentile(arr, 70)),
            "P100": float(np.percentile(arr, 100)),
        }

    benchmark_results = {
        "Embedding": calc_percentiles(embedding_latencies),
        "FAISS": calc_percentiles(faiss_latencies),
        "BM25": calc_percentiles(bm25_latencies),
        "Reranker": calc_percentiles(rerank_latencies),
        "Guardrail": calc_percentiles(guardrail_latencies),
        "Total Retrieval": calc_percentiles(total_retrieval_latencies),
    }

    if llm_latencies:
        benchmark_results["LLM"] = calc_percentiles(llm_latencies)
    benchmark_results["End-to-End"] = calc_percentiles(end_to_end_latencies)

    return benchmark_results


def print_latency_report(benchmark_results: Dict[str, Dict[str, float]]):
    """Format and print benchmark latency table."""
    print("\n" + "-" * 60)
    print("LATENCY BENCHMARK (ms)")
    print("-" * 60)
    header = f"{'Component':<18} {'P50':<10} {'P70':<10} {'P100':<10}"
    print(header)
    print("-" * 60)

    for component, metrics in benchmark_results.items():
        print(
            f"{component:<18} {metrics['P50']:<10.2f} {metrics['P70']:<10.2f} {metrics['P100']:<10.2f}"
        )
    print("-" * 60)
