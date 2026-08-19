import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.normalize import load_and_normalize_dataset
from src.chunking.evaluator import evaluate_chunking_strategies
from src.config.settings import VAL_FILE

def main():
    parser = argparse.ArgumentParser(description="Evaluate & compare 5 chunking strategies")
    parser.add_argument("--val_path", type=str, default=VAL_FILE, help="Path to validation parquet dataset")
    parser.add_argument("--num_records", type=int, default=100, help="Number of records to test")
    args = parser.parse_args()

    print("[EVALUATE CHUNKING] Loading sample dataset records...")
    records = load_and_normalize_dataset(args.val_path, max_rows=args.num_records)
    
    report = evaluate_chunking_strategies(records)

    print("\n" + "=" * 80)
    print("EXPERIMENTAL CHUNKING STRATEGY COMPARISON REPORT")
    print("=" * 80)
    print(f"{'Strategy':<15} | {'Chunks':<8} | {'GroundTruth':<12} | {'Avg Len':<10} | {'Time (s)':<10} | {'Throughput':<12}")
    print("-" * 80)
    for strat, data in report.items():
        print(
            f"{strat:<15} | "
            f"{data['total_chunks_produced']:<8} | "
            f"{data['selected_ground_truth_chunks']:<12} | "
            f"{data['avg_chunk_length_chars']:<10.1f} | "
            f"{data['chunking_time_sec']:<10.4f} | "
            f"{data['throughput_chunks_per_sec']:<12.1f}"
        )
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
