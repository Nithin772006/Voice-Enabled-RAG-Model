import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.normalize import load_and_normalize_dataset
from src.pipeline.orchestrator import VoiceRAGPipeline
from src.config.settings import VAL_FILE

def main():
    parser = argparse.ArgumentParser(description="Test Answerable vs Unanswerable Query Suite")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "quality"], help="Pipeline mode")
    args = parser.parse_args()

    print("=" * 80)
    print(f"HH GOA 2026 TASK 2 — ANSWERABLE / UNANSWERABLE SUITE (MODE: {args.mode.upper()})")
    print("=" * 80)

    pipeline = VoiceRAGPipeline()
    if not pipeline.initialize_system(force_rebuild=False):
        print("[ERROR] Index cache not found. Run 'python scripts/build_index.py' first.")
        sys.exit(1)

    val_records = load_and_normalize_dataset(VAL_FILE, max_rows=100)
    
    # 1. Select 5 ANSWERABLE dataset queries
    answerable_queries = []
    for r in val_records:
        if r.get("query") and r.get("answer") and r["answer"].strip() != "எந்த பதிலும் இல்லை.":
            answerable_queries.append((r["query"], r["answer"]))
            if len(answerable_queries) >= 5:
                break

    # 2. Define 5 UNANSWERABLE out-of-domain queries
    unanswerable_queries = [
        "What is the largest planet in our solar system?",
        "Who won the FIFA World Cup in 2026?",
        "How to build a nuclear reactor at home?",
        "What is the current stock price of Apple?",
        "சூரிய குடும்பத்தில் மிகப்பெரிய கோள் எது?"
    ]

    print("\n--- TEST GROUP A: ANSWERABLE QUERIES (FROM DATASET) ---")
    for q_text, true_ans in answerable_queries:
        res = pipeline.run_text_query(q_text, mode=args.mode)
        lats = res.get("latencies_ms", {})
        print(f"\n[QUERY]: {q_text}")
        print(f"  System Answer: {res['answer']}")
        print(f"  Confidence:    {res['confidence']:.4f} | Grounded: {res['is_grounded']} | Mode: {res['mode'].upper()} ({res['answer_mode'].upper()})")
        print(f"  Latency:       {lats.get('total_latency_ms', 0.0):.2f} ms (Dense: {lats.get('dense_search_ms', 0.0):.2f}ms, Gen: {lats.get('generation_ms', 0.0):.2f}ms)")

    print("\n--- TEST GROUP B: UNANSWERABLE / OUT-OF-DOMAIN QUERIES ---")
    for q_text in unanswerable_queries:
        res = pipeline.run_text_query(q_text, mode=args.mode)
        lats = res.get("latencies_ms", {})
        print(f"\n[QUERY]: {q_text}")
        print(f"  System Answer: {res['answer']}")
        print(f"  Status:        {res['status']} | Confidence: {res['confidence']:.4f} | Grounded: {res['is_grounded']}")
        print(f"  Latency:       {lats.get('total_latency_ms', 0.0):.2f} ms")

    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
