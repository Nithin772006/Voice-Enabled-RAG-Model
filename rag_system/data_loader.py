import os
import pyarrow.parquet as pq
from typing import List, Dict, Any, Tuple


def extract_passages(row: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    """Extract translated passages and binary selection labels from MSMARCO-XI row."""
    passages = row.get("passages", None)
    if passages is None:
        return [], []

    translated = []
    selected = []

    if isinstance(passages, dict):
        translated = passages.get("Translated_passages", [])
        selected = passages.get("is_selected", [])
    else:
        try:
            translated = passages["Translated_passages"]
            selected = passages["is_selected"]
        except Exception:
            pass

    # Ensure python primitive lists
    if hasattr(translated, "to_pylist"):
        translated = translated.to_pylist()
    elif not isinstance(translated, list):
        translated = list(translated) if translated is not None else []

    if hasattr(selected, "to_pylist"):
        selected = selected.to_pylist()
    elif not isinstance(selected, list):
        selected = list(selected) if selected is not None else []

    clean_translated = [str(p).strip() for p in translated if p is not None]
    clean_selected = []
    for s in selected:
        try:
            clean_selected.append(int(s))
        except Exception:
            clean_selected.append(0)

    return clean_translated, clean_selected


def load_msmarco_samples(
    parquet_path: str, max_rows: int = 5000, batch_size: int = 500
) -> List[Dict[str, Any]]:
    """Stream MSMARCO-XI rows from parquet in batches to avoid high memory usage."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    examples = []
    rows_loaded = 0

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        data = batch.to_pydict()
        queries = data.get("query", [])
        answers = data.get("Answer", data.get("answers", []))
        passages_list = data.get("passages", [])
        query_ids = data.get("query_id", list(range(rows_loaded, rows_loaded + len(queries))))

        batch_len = len(queries)
        for i in range(batch_len):
            row = {
                "query_id": query_ids[i] if i < len(query_ids) else rows_loaded,
                "query": str(queries[i]) if i < len(queries) else "",
                "answer": str(answers[i]) if i < len(answers) else "",
                "passages": passages_list[i] if i < len(passages_list) else {},
            }
            examples.append(row)
            rows_loaded += 1
            if max_rows is not None and rows_loaded >= max_rows:
                break
        if max_rows is not None and rows_loaded >= max_rows:
            break

    return examples
