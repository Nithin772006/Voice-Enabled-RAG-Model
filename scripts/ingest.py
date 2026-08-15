import logging
import json
import pyarrow.parquet as pq
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

# Configure Logging for production traceability
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IngestionPipeline")

@dataclass(frozen=True)
class Passage:
    text: str
    selected: bool

    def validate(self) -> bool:
        """Validates that the passage text is non-empty."""
        return bool(self.text and self.text.strip())

@dataclass(frozen=True)
class RAGRecord:
    id: str
    language: str
    query: str
    english_query: str
    answer: str
    english_answer: str
    passages: List[Passage]

    def validate(self) -> bool:
        """Validates the record fields to ensure high data quality.
        Returns True if the record meets requirements, False otherwise.
        """
        if not self.id:
            return False
        if not self.language:
            return False
        if not self.query or not self.query.strip():
            return False
        if not self.english_query or not self.english_query.strip():
            return False
        if not self.answer or not self.answer.strip():
            return False
        if not self.english_answer or not self.english_answer.strip():
            return False
        if not self.passages:
            return False
        
        # Ensure all nested passages are valid
        return all(p.validate() for p in self.passages)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the record to the standardized target schema."""
        return {
            "id": self.id,
            "language": self.language,
            "query": self.query,
            "english_query": self.english_query,
            "answer": self.answer,
            "english_answer": self.english_answer,
            "passages": [
                {"text": p.text, "selected": p.selected}
                for p in self.passages
            ]
        }

@dataclass
class IngestionConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    batch_size: int = 1000

    def __post_init__(self) -> None:
        # Convert path variables to Path objects and resolve absolute paths
        self.raw_dir = Path(self.raw_dir).resolve()
        self.processed_dir = Path(self.processed_dir).resolve()

def extract_parquet_to_jsonl(parquet_path: Path, output_jsonl_path: Path) -> None:
    """Reads a nested parquet file in batches, validates, and extracts to JSONL."""
    logger.info(f"Starting extraction from {parquet_path.name} to {output_jsonl_path.name}...")
    num_records = 0
    num_invalid = 0
    
    # Open the parquet file
    pf = pq.ParquetFile(parquet_path)
    
    # Ensure parent directory of output exists
    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        # Read in batches to keep memory consumption constant regardless of dataset size
        for batch in pf.iter_batches(batch_size=1000):
            data = batch.to_pydict()
            num_rows = len(data["query_id"])
            
            for i in range(num_rows):
                # Extract passages fields
                passages_struct = data["passages"][i]
                trans_passages = passages_struct["Translated_passages"]
                is_selected = passages_struct["is_selected"]
                
                # Build Passage objects
                passages_list = []
                for trans, sel in zip(trans_passages, is_selected):
                    passages_list.append(Passage(
                        text=trans,
                        selected=bool(sel)
                    ))
                
                # Build RAGRecord
                record = RAGRecord(
                    id=str(data["query_id"][i]),
                    language=data["target_lang"][i],
                    query=data["query"][i],
                    english_query=data["Eng_Query"][i],
                    answer=data["Answer"][i],
                    english_answer=data["Eng_Answer"][i],
                    passages=passages_list
                )
                
                # Check schema validation
                if record.validate():
                    # Write to file
                    f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                    num_records += 1
                else:
                    num_invalid += 1
                    
    logger.info(f"Extraction completed. Extracted: {num_records} records, Skipped: {num_invalid} invalid records.")

if __name__ == "__main__":
    config = IngestionConfig()
    raw_val_parquet = config.raw_dir / "tamval.parquet"
    raw_ta_jsonl = config.raw_dir / "ta.jsonl"
    
    if raw_val_parquet.exists():
        extract_parquet_to_jsonl(raw_val_parquet, raw_ta_jsonl)
    else:
        logger.error(f"Raw validation parquet file not found at {raw_val_parquet}")
