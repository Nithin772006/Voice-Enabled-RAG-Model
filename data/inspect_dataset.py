import os
import pyarrow.parquet as pq
import numpy as np

def inspect_parquet_file(file_path: str, name: str, max_inspect_rows: int = 500):
    print("=" * 70)
    print(f"INSPECTING DATASET: {name}")
    print(f"File Path: {file_path}")
    print("=" * 70)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB ({file_size_mb/1024:.2f} GB)")
    
    parquet_file = pq.ParquetFile(file_path)
    schema = parquet_file.schema
    print("\n--- PARQUET SCHEMA ---")
    print(schema)
    
    total_records = parquet_file.metadata.num_rows
    print(f"\nTotal Records in File: {total_records:,}")
    
    # Stream first batch safely
    first_batch = next(parquet_file.iter_batches(batch_size=max_inspect_rows))
    pydict = first_batch.to_pydict()
    columns = list(pydict.keys())
    print(f"Available Columns: {columns}")
    
    print("\n--- SAMPLE RECORD 0 ---")
    for col in columns:
        val = pydict[col][0] if len(pydict[col]) > 0 else None
        val_str = str(val)
        if len(val_str) > 250:
            val_str = val_str[:250] + f"... [length {len(val_str)}]"
        print(f"  [{col}]: {val_str}")
        
    print(f"\n--- STATISTICAL ANALYSIS ON SAMPLE ({len(pydict[columns[0]])} records) ---")
    for col in columns:
        vals = pydict[col]
        null_cnt = sum(1 for v in vals if v is None or (isinstance(v, float) and np.isnan(v)))
        print(f"Column '{col}': Nulls = {null_cnt} / {len(vals)} ({null_cnt/len(vals)*100:.1f}%)")
        
    if "query" in pydict:
        queries = [str(q) for q in pydict["query"] if q is not None]
        q_lens = [len(q) for q in queries]
        print(f"Query Char Length -> Min: {min(q_lens)}, Max: {max(q_lens)}, Mean: {np.mean(q_lens):.1f}, Median: {np.median(q_lens):.1f}")

    if "passages" in pydict:
        p_counts = []
        p_lens = []
        sel_counts = []
        for p_struct in pydict["passages"]:
            if isinstance(p_struct, dict):
                passages = p_struct.get("Translated_passages", [])
                selected = p_struct.get("is_selected", [])
                p_counts.append(len(passages))
                sel_counts.append(sum(1 for s in selected if int(s) == 1) if selected else 0)
                for p in passages:
                    p_lens.append(len(str(p)))
        if p_counts:
            print(f"Passages per Record -> Min: {min(p_counts)}, Max: {max(p_counts)}, Mean: {np.mean(p_counts):.1f}, Median: {np.median(p_counts):.1f}")
            print(f"Selected Passages per Record -> Min: {min(sel_counts)}, Max: {max(sel_counts)}, Mean: {np.mean(sel_counts):.1f}")
        if p_lens:
            print(f"Passage Char Length -> Min: {min(p_lens)}, Max: {max(p_lens)}, Mean: {np.mean(p_lens):.1f}, Median: {np.median(p_lens):.1f}")

    ans_col = "Answer" if "Answer" in pydict else ("answers" if "answers" in pydict else None)
    if ans_col:
        answers = [str(a) for a in pydict[ans_col] if a is not None]
        a_lens = [len(a) for a in answers]
        if a_lens:
            print(f"Answer Char Length -> Min: {min(a_lens)}, Max: {max(a_lens)}, Mean: {np.mean(a_lens):.1f}")

    print("=" * 70 + "\n")

if __name__ == "__main__":
    train_path = r"D:\New folder\HH Goa\train\tamtrain.parquet"
    val_path = r"D:\New folder\HH Goa\validation\tamval.parquet"
    inspect_parquet_file(train_path, "MSMARCO-XI Tamil Train Set")
    inspect_parquet_file(val_path, "MSMARCO-XI Tamil Validation Set")
