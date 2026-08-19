import numpy as np
from typing import List, Dict, Any

def compute_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p70": 0.0, "p100": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    
    arr = np.array(values)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p100": float(np.percentile(arr, 100)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def benchmark_pipeline_latency(
    pipeline,
    test_queries: List[str],
    mode: str = "fast"
) -> Dict[str, Dict[str, float]]:
    """
    Run pipeline over test queries and compute P50, P70, P100, mean, min, max per stage.
    """
    stage_samples: Dict[str, List[float]] = {
        "stt": [],
        "query_processing": [],
        "dense_retrieval": [],
        "sparse_retrieval": [],
        "total_retrieval": [],
        "reranking": [],
        "guardrails": [],
        "generation": [],
        "total_pipeline": []
    }

    print(f"[BENCHMARK] Running latency benchmark over {len(test_queries)} queries (mode={mode})...")

    for q in test_queries:
        res = pipeline.run_text_query(q, mode=mode)
        lats = res.get("latencies_ms", {})
        
        stage_samples["stt"].append(lats.get("stt_ms", 0.0))
        stage_samples["query_processing"].append(lats.get("query_proc_ms", 0.0))
        stage_samples["dense_retrieval"].append(lats.get("dense_search_ms", 0.0))
        stage_samples["sparse_retrieval"].append(lats.get("sparse_search_ms", 0.0))
        stage_samples["total_retrieval"].append(lats.get("retrieval_ms", 0.0))
        stage_samples["reranking"].append(lats.get("rerank_ms", 0.0))
        stage_samples["guardrails"].append(lats.get("guardrail_ms", 0.0))
        stage_samples["generation"].append(lats.get("generation_ms", 0.0))
        stage_samples["total_pipeline"].append(lats.get("total_latency_ms", 0.0))

    report = {}
    for stage_name, samples in stage_samples.items():
        stats = compute_percentiles(samples)
        report[stage_name] = {k: round(v, 2) for k, v in stats.items()}

    return report

def print_latency_report(report: Dict[str, Dict[str, float]]):
    print("\n" + "=" * 75)
    print("HH GOA 2026 TASK 2 — PIPELINE LATENCY BENCHMARK REPORT (ms)")
    print("=" * 75)
    print(f"{'Stage':<20} | {'P50':<10} | {'P70':<10} | {'P100':<10} | {'Mean':<10} | {'Min':<8} | {'Max':<8}")
    print("-" * 75)
    for stage, stats in report.items():
        print(
            f"{stage:<20} | "
            f"{stats['p50']:<10.2f} | "
            f"{stats['p70']:<10.2f} | "
            f"{stats['p100']:<10.2f} | "
            f"{stats['mean']:<10.2f} | "
            f"{stats['min']:<8.2f} | "
            f"{stats['max']:<8.2f}"
        )
    print("=" * 75 + "\n")
