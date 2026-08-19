import re
import uuid
import numpy as np
from typing import List, Dict, Any
from src.chunking.base import BaseChunker, ChunkMetadata

class SemanticChunker(BaseChunker):
    """Strategy C — Semantic Chunking grouping semantically cohesive sentences."""
    
    def __init__(self, similarity_threshold: float = 0.40, max_chunk_chars: int = 500):
        super().__init__("semantic")
        self.similarity_threshold = similarity_threshold
        self.max_chunk_chars = max_chunk_chars
        self.sentence_pattern = re.compile(r'(?<=[.!?|।\n])\s+')

    def _get_sentence_words(self, sent: str) -> set:
        return set(re.findall(r'\w+', sent.lower()))

    def _compute_jaccard_similarity(self, s1: str, s2: str) -> float:
        w1 = self._get_sentence_words(s1)
        w2 = self._get_sentence_words(s2)
        if not w1 or not w2:
            return 0.0
        intersection = w1.intersection(w2)
        union = w1.union(w2)
        return len(intersection) / float(len(union))

    def chunk_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[ChunkMetadata]:
        chunks = []
        sentences = [s.strip() for s in self.sentence_pattern.split(text) if s and s.strip()]
        if not sentences:
            return chunks

        current_group = [sentences[0]]
        current_len = len(sentences[0])
        chunk_idx = 0

        for i in range(1, len(sentences)):
            sent = sentences[i]
            sim = self._compute_jaccard_similarity(current_group[-1], sent)
            
            if (sim >= self.similarity_threshold or len(current_group) == 1) and (current_len + len(sent) <= self.max_chunk_chars):
                current_group.append(sent)
                current_len += len(sent)
            else:
                chunk_str = " ".join(current_group).strip()
                c_id = f"{doc_metadata.get('document_id', 'doc')}_sem_{chunk_idx}_{uuid.uuid4().hex[:6]}"
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
                    is_selected=doc_metadata.get("is_selected", 0)
                ))
                chunk_idx += 1
                current_group = [sent]
                current_len = len(sent)

        if current_group:
            chunk_str = " ".join(current_group).strip()
            c_id = f"{doc_metadata.get('document_id', 'doc')}_sem_{chunk_idx}_{uuid.uuid4().hex[:6]}"
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
                is_selected=doc_metadata.get("is_selected", 0)
            ))

        return chunks
