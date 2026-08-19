import os
import re
import unicodedata
import pyarrow.parquet as pq
from typing import List, Dict, Any, Tuple, Optional
from src.config.settings import PROCESSED_DATA_DIR, PARQUET_BATCH_SIZE

def normalize_text(text: str) -> str:
    """Perform Unicode normalization (NFKC) and clean excessive whitespace."""
    if not text:
        return ""
    # NFKC normalizes compatibility characters while preserving canonical equivalence
    normalized = unicodedata.normalize("NFKC", str(text))
    # Replace non-breaking spaces and redundant tabs/newlines with single spaces
    cleaned = re.sub(r"\s+", " ", normalized).strip()
    return cleaned

def clean_passage_list(passages: Any) -> Tuple[List[str], List[str], List[int]]:
    """Extract and normalize English passages, Tamil translated passages, and is_selected labels."""
    eng_passages = []
    trans_passages = []
    selected_flags = []
    
    if isinstance(passages, dict):
        eng = passages.get("English_passages", [])
        trans = passages.get("Translated_passages", [])
        sel = passages.get("is_selected", [])
    else:
        eng = getattr(passages, "English_passages", [])
        trans = getattr(passages, "Translated_passages", [])
        sel = getattr(passages, "is_selected", [])
        
    if hasattr(eng, "to_pylist"): eng = eng.to_pylist()
    if hasattr(trans, "to_pylist"): trans = trans.to_pylist()
    if hasattr(sel, "to_pylist"): sel = sel.to_pylist()
    
    eng = list(eng) if eng is not None else []
    trans = list(trans) if trans is not None else []
    sel = list(sel) if sel is not None else []
    
    length = max(len(eng), len(trans))
    for i in range(length):
        e_p = normalize_text(eng[i]) if i < len(eng) and eng[i] is not None else ""
        t_p = normalize_text(trans[i]) if i < len(trans) and trans[i] is not None else ""
        s_flag = 0
        if i < len(sel) and sel[i] is not None:
            try:
                s_flag = int(sel[i])
            except (ValueError, TypeError):
                s_flag = 0
                
        # Retain passage if at least translated text or English text exists
        if e_p or t_p:
            eng_passages.append(e_p)
            trans_passages.append(t_p)
            selected_flags.append(s_flag)
            
    return eng_passages, trans_passages, selected_flags

def process_msmarco_record(raw_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate, clean, normalize, and format a single record from MSMARCO-XI."""
    query_id = raw_record.get("query_id")
    query = normalize_text(raw_record.get("query", ""))
    eng_query = normalize_text(raw_record.get("Eng_Query", ""))
    answer = normalize_text(raw_record.get("Answer", ""))
    eng_answer = normalize_text(raw_record.get("Eng_Answer", ""))
    
    source_lang = str(raw_record.get("source_lang", "eng_Latn"))
    target_lang = str(raw_record.get("target_lang", "tam_Taml"))
    query_type = str(raw_record.get("query_type", "UNKNOWN"))
    
    # Must have valid query and query_id
    if not query or query_id is None:
        return None
        
    passages_raw = raw_record.get("passages", {})
    eng_passages, trans_passages, selected_flags = clean_passage_list(passages_raw)
    
    if not trans_passages and not eng_passages:
        return None
        
    return {
        "query_id": query_id,
        "query": query,
        "eng_query": eng_query,
        "answer": answer,
        "eng_answer": eng_answer,
        "query_type": query_type,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "english_passages": eng_passages,
        "translated_passages": trans_passages,
        "is_selected": selected_flags,
        "num_passages": len(trans_passages)
    }

def load_and_normalize_dataset(
    parquet_path: str,
    max_rows: Optional[int] = None,
    deduplicate: bool = True
) -> List[Dict[str, Any]]:
    """Stream parquet batches, apply normalization, deduplicate queries, and return clean records."""
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Dataset parquet file not found: {parquet_path}")

    parquet_file = pq.ParquetFile(parquet_path)
    clean_records = []
    seen_queries = set()
    total_processed = 0

    for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE):
        batch_dict = batch.to_pydict()
        num_in_batch = len(batch_dict["query"]) if "query" in batch_dict else 0

        for i in range(num_in_batch):
            raw_rec = {col: batch_dict[col][i] for col in batch_dict.keys()}
            processed = process_msmarco_record(raw_rec)

            if processed is None:
                continue

            q_key = (processed["query_id"], processed["query"])
            if deduplicate and q_key in seen_queries:
                continue

            seen_queries.add(q_key)
            clean_records.append(processed)
            total_processed += 1

            if max_rows is not None and total_processed >= max_rows:
                break

        if max_rows is not None and total_processed >= max_rows:
            break

    print(f"[DATA NORMALIZATION] Loaded {total_processed} clean records from {os.path.basename(parquet_path)}")
    return clean_records
