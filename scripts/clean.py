import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Optional

# Reconfigure stdout to support UTF-8 on Windows terminal/console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DataCleaning")

def normalize_text(text: str) -> str:
    """Normalizes Unicode encoding to NFC, standardizes line breaks,
    collapses multiple spaces, and strips leading/trailing whitespace.
    """
    if not text:
        return ""
    
    # 1. Normalize Unicode to Canonical Composition (NFC)
    text = unicodedata.normalize("NFC", text)
    
    # 2. Standardize all types of carriage returns / vertical tabs to a single newline character
    text = re.sub(r"\r\n|\r|\v", "\n", text)
    
    # 3. Standardize other spaces (non-breaking spaces \xa0, tabs \t) to standard spaces
    text = re.sub(r"[ \t\xa0\u200b]+", " ", text)
    
    # 4. Collapse multiple consecutive newlines (more than 2) to prevent excessive whitespace gaps
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # 5. Clean up any trailing space per line
    text = "\n".join(line.strip() for line in text.split("\n"))
    
    # 6. Final strip for the outer boundaries
    return text.strip()

def clean_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Cleans text fields, removes invalid/empty passages, and deduplicates passages."""
    # Clean text fields
    record["query"] = normalize_text(record.get("query", ""))
    record["english_query"] = normalize_text(record.get("english_query", ""))
    record["answer"] = normalize_text(record.get("answer", ""))
    record["english_answer"] = normalize_text(record.get("english_answer", ""))
    
    # Basic schema validation: ensure core IDs and texts are present
    if not record.get("id") or not record["query"] or not record["answer"]:
        return None
        
    cleaned_passages = []
    seen_texts = set()
    
    for p in record.get("passages", []):
        cleaned_text = normalize_text(p.get("text", ""))
        
        # Rule: Remove passages containing only whitespace
        if not cleaned_text:
            continue
            
        # Rule: Remove duplicate passages within a document to preserve token budget
        if cleaned_text in seen_texts:
            continue
        seen_texts.add(cleaned_text)
        
        cleaned_passages.append({
            "text": cleaned_text,
            "selected": bool(p.get("selected", False))
        })
        
    # Rule: If a document has no passages left, discard the record (invalid for RAG)
    if not cleaned_passages:
        return None
        
    record["passages"] = cleaned_passages
    return record

def clean_jsonl_file(input_path: Path, output_path: Path) -> None:
    """Reads a JSONL file, cleans each record, and writes valid records to output_path."""
    logger.info(f"Cleaning {input_path.name} -> {output_path.name}...")
    num_records = 0
    num_skipped = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cleaned = clean_record(record)
                if cleaned:
                    fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    num_records += 1
                else:
                    num_skipped += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line as JSON: {e}")
                num_skipped += 1
                
    logger.info(f"Cleaning completed. Cleaned: {num_records} records, Skipped: {num_skipped} invalid/empty records.")

if __name__ == "__main__":
    input_file = Path("data/raw/ta.jsonl")
    output_file = Path("data/processed/ta_clean.jsonl")
    
    if input_file.exists():
        clean_jsonl_file(input_file, output_file)
    else:
        # Fallback to test case if ta.jsonl doesn't exist yet (keeps original testing logic functional)
        logger.info("Raw ta.jsonl not found. Running mock test case...")
        sample_tamil_nfd = "ப" + "\u0bbf" + "ர" + "ப" + "ஞ" + "\u0bcd" + "ச" + "ம" + "\u0bcd"
        sample_raw_text = f"  Hello \t World!  \n\n\n  இது  ஒரு {sample_tamil_nfd}   \xa0  பரிசோதனை.  \r\n"
        print("--- RAW TEXT ---")
        print(repr(sample_raw_text))
        cleaned = normalize_text(sample_raw_text)
        print("\n--- CLEANED TEXT ---")
        print(repr(cleaned))
        is_nfc = unicodedata.is_normalized("NFC", cleaned)
        print(f"\nIs cleaned text NFC normalized? {is_nfc}")
