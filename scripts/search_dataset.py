import sys
import os
import argparse
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.normalize import load_and_normalize_dataset
from src.config.settings import TRAIN_FILE, VAL_FILE

def search_in_dataset(file_path: str, keywords: list, max_inspect: int = 50000):
    print("=" * 70)
    print(f"SEARCHING DATASET: {file_path}")
    print(f"Keywords: {keywords}")
    print("=" * 70)

    records = load_and_normalize_dataset(file_path, max_rows=max_inspect)
    pattern = re.compile("|".join([re.escape(k.lower()) for k in keywords]))

    matched_records = []
    for rec in records:
        text_to_search = f"{rec['query']} {rec['eng_query']} {rec['answer']} {rec['eng_answer']} {' '.join(rec['translated_passages'])} {' '.join(rec['english_passages'])}"
        if pattern.search(text_to_search.lower()):
            matched_records.append(rec)

    print(f"\nFound {len(matched_records)} matching records out of {len(records)} inspected records.")
    for idx, r in enumerate(matched_records[:5], 1):
        print(f"\n--- MATCH {idx} (ID: {r['query_id']}) ---")
        print(f"  Tamil Query: {r['query']}")
        print(f"  Eng Query:   {r['eng_query']}")
        print(f"  Tamil Answer:{r['answer']}")
        print(f"  Eng Answer:  {r['eng_answer']}")
        print(f"  Passages count: {len(r['translated_passages'])}")
        if r['translated_passages']:
            print(f"  Passage 0 snippet: {r['translated_passages'][0][:200]}...")

    if not matched_records:
        print("\n[RESULT] Information NOT found in dataset for these keywords.")
        print("[METADATA FLAG]: UNANSWERABLE_FROM_DATASET")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search MSMARCO-XI dataset for specific topics")
    parser.add_argument("--keywords", nargs="+", default=["planet", "jupiter", "solar system", "வியாழன்", "சூரிய"], help="Search keywords")
    parser.add_argument("--max_rows", type=int, default=30000, help="Max records to search")
    args = parser.parse_args()

    search_in_dataset(VAL_FILE, args.keywords, max_inspect=args.max_rows)
