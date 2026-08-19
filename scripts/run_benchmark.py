import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.normalize import load_and_normalize_dataset
from src.pipeline.orchestrator import VoiceRAGPipeline
from src.benchmark.latency import benchmark_pipeline_latency, print_latency_report
from src.config.settings import VAL_FILE

def main():
    parser = argparse.ArgumentParser(description="Run pipeline latency benchmark")
    parser.add_argument("--val_path", type=str, default=VAL_FILE, help="Path to validation parquet dataset")
    parser.add_argument("--num_queries", type=int, default=20, help="Number of queries to benchmark")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "quality"], help="Pipeline mode")
    args = parser.parse_args()

    print(f"[BENCHMARK SCRIPT] Loading pipeline and validation dataset (Mode={args.mode.upper()})...")
    pipeline = VoiceRAGPipeline()
    if not pipeline.initialize_system(force_rebuild=False):
        print("[ERROR] Index cache not found. Run 'python scripts/build_index.py' first.")
        sys.exit(1)

    val_records = load_and_normalize_dataset(args.val_path, max_rows=args.num_queries)
    val_queries = [r["query"] for r in val_records if r.get("query")]

    report = benchmark_pipeline_latency(
        pipeline,
        val_queries,
        mode=args.mode
    )
    print_latency_report(report)

if __name__ == "__main__":
    main()
