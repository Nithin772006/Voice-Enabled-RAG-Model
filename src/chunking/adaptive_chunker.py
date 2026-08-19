from typing import List, Dict, Any
from src.chunking.base import BaseChunker, ChunkMetadata
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.sentence_chunker import SentenceAwareChunker
from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.metadata_chunker import MetadataAwareChunker

class AdaptiveChunker(BaseChunker):
    """Strategy E — Adaptive Chunking dynamically selecting strategy based on text length & structure."""
    
    def __init__(self, short_threshold: int = 250, long_threshold: int = 800):
        super().__init__("adaptive")
        self.short_threshold = short_threshold
        self.long_threshold = long_threshold
        
        self.sentence_chunker = SentenceAwareChunker(max_chars_per_chunk=350)
        self.semantic_chunker = SemanticChunker(max_chunk_chars=450)
        self.metadata_chunker = MetadataAwareChunker(max_chars=400)
        self.fixed_chunker = FixedSizeChunker(chunk_size=350, overlap=40)

    def chunk_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[ChunkMetadata]:
        text_len = len(text)
        
        if text_len <= self.short_threshold:
            # Short passage: use sentence chunker to keep intact or split cleanly
            chunks = self.sentence_chunker.chunk_text(text, doc_metadata)
        elif text_len <= self.long_threshold:
            # Medium passage: use metadata-aware chunker
            chunks = self.metadata_chunker.chunk_text(text, doc_metadata)
        else:
            # Long complex passage: use semantic chunker
            chunks = self.semantic_chunker.chunk_text(text, doc_metadata)

        for chunk in chunks:
            chunk.chunk_strategy = f"{self.name}({chunk.chunk_strategy})"
            
        return chunks
