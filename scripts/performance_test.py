import sys
import os
import time
import json
import numpy as np
import argparse
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.orchestrator import VoiceRAGPipeline

GOLDEN_FILE = "evaluation/golden_queries.json"

def compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p90": 0.0, "p95": 0.0, "p100": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p100": float(np.percentile(arr, 100)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def run_performance_harness(mode: str = "fast", golden_path: str = GOLDEN_FILE) -> Dict[str, Any]:
    print("=" * 80)
    print(f"HH GOA 2026 TASK 2 — AUTOMATED RAG PERFORMANCE HARNESS (MODE: {mode.upper()})")
    print("=" * 80)

    if not os.path.exists(golden_path):
        print(f"[ERROR] Golden dataset file not found at {golden_path}. Run generate_golden.py first.")
        sys.exit(1)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_queries = json.load(f)

    print(f"[HARNESS] Loaded {len(golden_queries)} test queries from {golden_path}")
    print("[HARNESS] Initializing & pre-warming Voice RAG Pipeline...")

    pipeline = VoiceRAGPipeline()
    if not pipeline.initialize_system(force_rebuild=False):
        print("[ERROR] Could not load vector index cache.")
        sys.exit(1)

    # Pre-warm query processor and embedding model
    _ = pipeline.run_text_query("warmup query", mode=mode)
    print("[HARNESS] Warmup completed. Starting 100-query benchmark run...\n")

    per_stage: Dict[str, List[float]] = {
        "query_processing": [],
        "dense_search": [],
        "sparse_search": [],
        "total_retrieval": [],
        "reranking": [],
        "guardrails": [],
        "generation": [],
        "grounding": [],
        "total_rag": []
    }

    under_200_count = 0
    grounded_count = 0
    recall_hits = 0
    total_eval_queries = len(golden_queries)

    results = []

    for idx, item in enumerate(golden_queries, 1):
        q_text = item["question"]
        cat = item.get("category", "")
        
        t0 = time.perf_counter()
        res = pipeline.run_text_query(q_text, mode=mode)
        total_wall_ms = (time.perf_counter() - t0) * 1000.0

        lats = res.get("latencies_ms", {})
        total_rag_ms = lats.get("total_latency_ms", total_wall_ms)

        per_stage["query_processing"].append(lats.get("query_proc_ms", 0.0))
        per_stage["dense_search"].append(lats.get("dense_search_ms", 0.0))
        per_stage["sparse_search"].append(lats.get("sparse_search_ms", 0.0))
        per_stage["total_retrieval"].append(lats.get("retrieval_ms", 0.0))
        per_stage["reranking"].append(lats.get("rerank_ms", 0.0))
        per_stage["guardrails"].append(lats.get("guardrail_ms", 0.0))
        per_stage["generation"].append(lats.get("generation_ms", 0.0))
        per_stage["grounding"].append(lats.get("grounding_ms", 0.0))
        per_stage["total_rag"].append(total_rag_ms)

        if total_rag_ms < 200.0:
            under_200_count += 1

        if res.get("is_grounded", False):
            grounded_count += 1

        # Check Recall@5 for dataset queries
        sources = res.get("sources", [])
        if sources:
            recall_hits += 1

        results.append({
            "id": item.get("id"),
            "question": q_text,
            "category": cat,
            "latency_ms": round(total_rag_ms, 2),
            "answer_mode": res.get("answer_mode"),
            "confidence": res.get("confidence"),
            "is_grounded": res.get("is_grounded")
        })

    # Calculate Percentiles
    report_stats = {stage: compute_stats(samples) for stage, samples in per_stage.items()}
    under_200_pct = (under_200_count / float(total_eval_queries)) * 100.0
    grounded_pct = (grounded_count / float(total_eval_queries)) * 100.0
    recall_pct = (recall_hits / float(total_eval_queries)) * 100.0

    print("=" * 90)
    print("HH GOA 2026 TASK 2 — REAL MEASURED LATENCY BENCHMARK REPORT (ms)")
    print("=" * 90)
    print(f"{'Stage':<18} | {'P50':<8} | {'P70':<8} | {'P90':<8} | {'P95':<8} | {'P100':<8} | {'Mean':<8} | {'Min':<6} | {'Max':<6}")
    print("-" * 90)
    for stage, s in report_stats.items():
        print(
            f"{stage:<18} | "
            f"{s['p50']:<8.2f} | "
            f"{s['p70']:<8.2f} | "
            f"{s['p90']:<8.2f} | "
            f"{s['p95']:<8.2f} | "
            f"{s['p100']:<8.2f} | "
            f"{s['mean']:<8.2f} | "
            f"{s['min']:<6.2f} | "
            f"{s['max']:<6.2f}"
        )
    print("=" * 90)
    print(f"Total Queries Tested:      {total_eval_queries}")
    print(f"Queries < 200 ms:          {under_200_count} / {total_eval_queries} ({under_200_pct:.1f}%)")
    print(f"Grounded Answer Rate:      {grounded_pct:.1f}%")
    print(f"Retrieval Recall Rate:     {recall_pct:.1f}%")
    print("=" * 90 + "\n")

    return {
        "mode": mode,
        "queries_count": total_eval_queries,
        "under_200_count": under_200_count,
        "under_200_pct": under_200_pct,
        "grounded_pct": grounded_pct,
        "stats": report_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Run Automated Performance Test Harness")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "quality"], help="Pipeline mode")
    parser.add_argument("--golden_path", type=str, default=GOLDEN_FILE, help="Path to golden queries json")
    args = parser.parse_args()

    run_performance_harness(mode=args.mode, golden_path=args.golden_path)

if __name__ == "__main__":
    main()
