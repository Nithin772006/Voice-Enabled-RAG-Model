import sys
import os
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.normalize import load_and_normalize_dataset
from src.indexing.manager import IndexManager
from src.config.settings import TRAIN_FILE, DEFAULT_N_TRAIN_ROWS, INDEX_CACHE_DIR

def main():
    parser = argparse.ArgumentParser(description="Build vector & BM25 indexes for Voice RAG")
    parser.add_argument("--data_path", type=str, default=TRAIN_FILE, help="Path to parquet dataset file")
    parser.add_argument("--max_rows", type=int, default=DEFAULT_N_TRAIN_ROWS, help="Max parquet records to index")
    parser.add_argument("--strategy", type=str, default="sentence", choices=["fixed", "sentence", "semantic", "metadata", "adaptive"], help="Chunking strategy")
    parser.add_argument("--batch_size", type=int, default=128, help="Embedding batch size")
    args = parser.parse_args()

    print("=" * 70)
    print("HH GOA 2026 TASK 2 — INDEX BUILDING SCRIPT")
    print("=" * 70)
    print(f"Dataset File: {args.data_path}")
    print(f"Max Records: {args.max_rows}")
    print(f"Chunking Strategy: {args.strategy}")
    print(f"Index Cache Directory: {INDEX_CACHE_DIR}")

    t0 = time.perf_counter()
    clean_records = load_and_normalize_dataset(args.data_path, max_rows=args.max_rows)
    
    manager = IndexManager(cache_dir=INDEX_CACHE_DIR)
    total_chunks, build_sec = manager.build_and_save(
        clean_records,
        chunk_strategy=args.strategy,
        batch_size=args.batch_size
    )

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 70)
    print(f"SUCCESS: Built and cached index with {total_chunks} chunks in {total_time:.2f}s!")
    print("=" * 70)

if __name__ == "__main__":
    main()
