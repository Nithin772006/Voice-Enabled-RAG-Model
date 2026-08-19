import re
import uuid
from typing import List, Dict, Any
from src.chunking.base import BaseChunker, ChunkMetadata

class SentenceAwareChunker(BaseChunker):
    """Strategy B — Sentence-aware Chunking preserving sentence boundaries."""
    
    def __init__(self, max_chars_per_chunk: int = 450, max_sentences: int = 4):
        super().__init__("sentence_aware")
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_sentences = max_sentences
        self.sentence_pattern = re.compile(r'(?<=[.!?|।\n])\s+')

    def split_sentences(self, text: str) -> List[str]:
        raw_sents = self.sentence_pattern.split(text)
        return [s.strip() for s in raw_sents if s and s.strip()]

    def chunk_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[ChunkMetadata]:
        chunks = []
        sentences = self.split_sentences(text)
        if not sentences:
            return chunks

        current_sentences = []
        current_char_count = 0
        chunk_idx = 0

        extra_meta = {
            "query": doc_metadata.get("query", ""),
            "question": doc_metadata.get("query", ""),
            "answer": doc_metadata.get("answer", "")
        }

        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_sentences and (current_char_count + sentence_len > self.max_chars_per_chunk or len(current_sentences) >= self.max_sentences):
                chunk_str = " ".join(current_sentences).strip()
                if chunk_str:
                    c_id = f"{doc_metadata.get('document_id', 'doc')}_sent_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                    chunks.append(ChunkMetadata(
                        chunk_id=c_id,
                        document_id=doc_metadata.get("document_id", ""),
                        language=doc_metadata.get("language", "tam_Taml"),
                        source=doc_metadata.get("source", "MSMARCO-XI"),
                        chunk_strategy=self.name,
                        chunk_index=chunk_idx,
                        text=chunk_str,
                        original_record_id=doc_metadata.get("original_record_id", ""),
                        passage_idx=doc_metadata.get("passage_idx", 0),
                        is_selected=doc_metadata.get("is_selected", 0),
                        extra=extra_meta
                    ))
                    chunk_idx += 1
                current_sentences = []
                current_char_count = 0

            current_sentences.append(sentence)
            current_char_count += sentence_len

        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            if chunk_str:
                c_id = f"{doc_metadata.get('document_id', 'doc')}_sent_{chunk_idx}_{uuid.uuid4().hex[:6]}"
                chunks.append(ChunkMetadata(
                    chunk_id=c_id,
                    document_id=doc_metadata.get("document_id", ""),
                    language=doc_metadata.get("language", "tam_Taml"),
                    source=doc_metadata.get("source", "MSMARCO-XI"),
                    chunk_strategy=self.name,
                    chunk_index=chunk_idx,
                    text=chunk_str,
                    original_record_id=doc_metadata.get("original_record_id", ""),
                    passage_idx=doc_metadata.get("passage_idx", 0),
                    is_selected=doc_metadata.get("is_selected", 0),
                    extra=extra_meta
                ))

        return chunks
