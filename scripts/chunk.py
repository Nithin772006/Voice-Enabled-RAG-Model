import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

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
logger = logging.getLogger("DataChunking")

def split_text_recursive(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
    """Recursively splits a text into chunks based on a list of priority separators
    without exceeding chunk_size, while maintaining a chunk_overlap.
    """
    if not text:
        return []
        
    separator = separators[0]
    next_separators = separators[1:]
    
    if separator == "":
        splits = list(text)
    else:
        splits = text.split(separator)
        
    chunks = []
    current_chunk = []
    current_len = 0
    
    for split in splits:
        split_len = len(split)
        
        if split_len > chunk_size:
            if current_chunk:
                chunks.append(separator.join(current_chunk))
                current_chunk = []
                current_len = 0
                
            if next_separators:
                chunks.extend(split_text_recursive(split, chunk_size, chunk_overlap, next_separators))
            else:
                i = 0
                while i < split_len:
                    chunks.append(split[i : i + chunk_size])
                    i += chunk_size - chunk_overlap
        else:
            sep_len = len(separator) if current_chunk else 0
            if current_len + sep_len + split_len > chunk_size:
                chunks.append(separator.join(current_chunk))
                
                overlap_chunk = []
                overlap_len = 0
                for prev_split in reversed(current_chunk):
                    prev_sep_len = len(separator) if overlap_chunk else 0
                    if overlap_len + prev_sep_len + len(prev_split) <= chunk_overlap:
                        overlap_chunk.insert(0, prev_split)
                        overlap_len += prev_sep_len + len(prev_split)
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = overlap_len
                
            if current_chunk:
                current_len += len(separator)
            current_chunk.append(split)
            current_len += split_len
            
    if current_chunk:
        chunks.append(separator.join(current_chunk))
        
    return chunks

def chunk_record(record: Dict[str, Any], chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[Dict[str, Any]]:
    """Chunks all passages in a single record and maps them to the standardized chunk schema."""
    chunks_output = []
    doc_id = record["id"]
    lang = record["language"]
    query = record["query"]
    
    for passage_idx, p in enumerate(record.get("passages", [])):
        p_text = p.get("text", "")
        p_selected = p.get("selected", False)
        
        # Split this passage's text
        passage_chunks = split_text_recursive(p_text, chunk_size, chunk_overlap, separators)
        total_chunks = len(passage_chunks)
        
        for chunk_idx, chunk_text in enumerate(passage_chunks):
            # Form unique chunk_id
            chunk_id = f"{doc_id}_p{passage_idx}_c{chunk_idx}"
            chunks_output.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "language": lang,
                "query": query,
                "text": chunk_text,
                "selected": p_selected,
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks
            })
            
    return chunks_output

def chunk_jsonl_file(input_path: Path, output_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """Reads a cleaned JSONL file, chunks each record, and writes chunks to output_path."""
    logger.info(f"Chunking {input_path.name} -> {output_path.name}...")
    num_docs = 0
    num_chunks = 0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seps = ["\n\n", "\n", " ", ""]
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                chunks = chunk_record(record, chunk_size, chunk_overlap, seps)
                for chunk in chunks:
                    fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    num_chunks += 1
                num_docs += 1
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line as JSON: {e}")
                
    logger.info(f"Chunking completed. Processed {num_docs} documents, Generated {num_chunks} chunks.")

if __name__ == "__main__":
    input_file = Path("data/processed/ta_clean.jsonl")
    output_file = Path("data/chunks/ta_chunks.jsonl")
    
    # Configure chunk parameters: size 400 characters, overlap 100 characters
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 100
    
    if input_file.exists():
        chunk_jsonl_file(input_file, output_file, CHUNK_SIZE, CHUNK_OVERLAP)
    else:
        logger.info("Cleaned ta_clean.jsonl not found. Running mock test case...")
        sample_text = (
            "Introduction: RAG stands for Retrieval-Augmented Generation.\n\n"
            "Paragraph 1: In a RAG pipeline, we retrieve relevant context from a database.\n\n"
            "Paragraph 2: If we embed an entire document, its semantic meaning is averaged."
        )
        print("--- SAMPLE CHUNKS ---")
        chunks = split_text_recursive(sample_text, 120, 30, ["\n\n", "\n", " ", ""])
        for idx, chunk in enumerate(chunks):
            print(f"Chunk {idx}: {repr(chunk)}")
