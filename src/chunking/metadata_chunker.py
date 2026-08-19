import uuid
from typing import List, Dict, Any
from src.chunking.base import BaseChunker, ChunkMetadata
from src.chunking.sentence_chunker import SentenceAwareChunker

class MetadataAwareChunker(BaseChunker):
    """Strategy D — Metadata-aware Chunking preserving rich document hierarchy & context."""
    
    def __init__(self, max_chars: int = 400):
        super().__init__("metadata_aware")
        self.sentence_chunker = SentenceAwareChunker(max_chars_per_chunk=max_chars)

    def chunk_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[ChunkMetadata]:
        base_chunks = self.sentence_chunker.chunk_text(text, doc_metadata)
        enriched_chunks = []

        query_id = doc_metadata.get("original_record_id", "")
        passage_idx = doc_metadata.get("passage_idx", 0)
        query = doc_metadata.get("query", "")

        for idx, chunk in enumerate(base_chunks):
            # Prepend contextual metadata header if query exists
            if query:
                enriched_text = f"[Context: {query}] {chunk.text}"
            else:
                enriched_text = chunk.text

            c_id = f"doc_{query_id}_p{passage_idx}_chk_{idx}_{uuid.uuid4().hex[:6]}"
            meta = ChunkMetadata(
                chunk_id=c_id,
                document_id=doc_metadata.get("document_id", f"doc_{query_id}"),
                language=doc_metadata.get("language", "tam_Taml"),
                source=doc_metadata.get("source", "MSMARCO-XI"),
                chunk_strategy=self.name,
                chunk_index=idx,
                text=enriched_text,
                original_record_id=str(query_id),
                passage_idx=passage_idx,
                is_selected=doc_metadata.get("is_selected", 0),
                extra={"raw_passage_text": chunk.text}
            )
            enriched_chunks.append(meta)

        return enriched_chunks
