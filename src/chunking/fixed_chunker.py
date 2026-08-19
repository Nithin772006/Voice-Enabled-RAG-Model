import uuid
from typing import List, Dict, Any
from src.chunking.base import BaseChunker, ChunkMetadata

class FixedSizeChunker(BaseChunker):
    """Strategy A — Fixed-size Chunking with overlap."""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        super().__init__("fixed_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[ChunkMetadata]:
        chunks = []
        if not text:
            return chunks
            
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_str = text[start:end].strip()
            
            if chunk_str:
                chunk_id = f"{doc_metadata.get('document_id', 'doc')}_chk_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=doc_metadata.get("document_id", ""),
                    language=doc_metadata.get("language", "tam_Taml"),
                    source=doc_metadata.get("source", "MSMARCO-XI"),
                    chunk_strategy=self.name,
                    chunk_index=chunk_idx,
                    text=chunk_str,
                    original_record_id=doc_metadata.get("original_record_id", ""),
                    passage_idx=doc_metadata.get("passage_idx", 0),
                    is_selected=doc_metadata.get("is_selected", 0)
                )
                chunks.append(metadata)
                chunk_idx += 1

            if end >= text_len:
                break
            start += (self.chunk_size - self.overlap)
            
        return chunks
