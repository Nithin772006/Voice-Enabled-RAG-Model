import re
from typing import List, Dict, Any
from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Abstract Base Class for Document Chunking Strategies."""

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        pass

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a list of document dicts while maintaining metadata trace."""
        chunked_docs = []
        for doc_idx, doc in enumerate(documents):
            passages = doc.get("passages", [])
            query = doc.get("query", "")
            answer = doc.get("answer", "")
            query_id = doc.get("query_id", doc_idx)

            # If document has raw passages
            if isinstance(passages, list) and passages:
                for passage_idx, p_info in enumerate(passages):
                    p_text = p_info.get("text", "") if isinstance(p_info, dict) else str(p_info)
                    is_selected = p_info.get("is_selected", 0) if isinstance(p_info, dict) else 0

                    chunks = self.chunk_text(p_text)
                    for chunk_idx, chunk in enumerate(chunks):
                        if not chunk.strip():
                            continue
                        chunked_docs.append({
                            "text": chunk,
                            "query": query,
                            "answer": answer,
                            "query_id": query_id,
                            "passage_idx": passage_idx,
                            "chunk_idx": chunk_idx,
                            "is_selected": is_selected,
                        })
            else:
                raw_text = doc.get("text", "")
                is_selected = doc.get("is_selected", 0)
                passage_idx = doc.get("passage_idx", 0)
                chunks = self.chunk_text(raw_text)
                for chunk_idx, chunk in enumerate(chunks):
                    if not chunk.strip():
                        continue
                    chunked_docs.append({
                        "text": chunk,
                        "query": query,
                        "answer": answer,
                        "query_id": query_id,
                        "passage_idx": passage_idx,
                        "chunk_idx": chunk_idx,
                        "is_selected": is_selected,
                    })
        return chunked_docs


class OriginalPassageChunker(BaseChunker):
    """Preserve each MSMARCO passage as a single retrieval unit."""

    def chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        return [text] if text else []


class FixedSizeChunker(BaseChunker):
    """Chunk text into fixed word-count windows with specified overlap."""

    def __init__(self, chunk_size: int = 100, overlap: int = 20):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.strip().split()
        if not words:
            return []
        if len(words) <= self.chunk_size:
            return [text.strip()]

        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
            if i + self.chunk_size >= len(words):
                break
        return chunks


class SentenceAwareChunker(BaseChunker):
    """Chunk text based on sentence boundaries (supporting Tamil and English punctuation)."""

    def __init__(self, max_words: int = 100):
        self.max_words = max_words
        # Matches Tamil sentence bounds (। ॥) and English (. ! ?)
        self.sentence_regex = re.compile(r"(?<=[.!?।॥])\s+")

    def chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        sentences = [s.strip() for s in self.sentence_regex.split(text) if s.strip()]
        if not sentences:
            return [text]

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            word_count = len(sentence_words)

            if current_chunk and (current_word_count + word_count > self.max_words):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sentence)
            current_word_count += word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class ParagraphStructureChunker(BaseChunker):
    """Chunk text by paragraph/structural breaks (\n\n), falling back to sentence bounds."""

    def __init__(self, max_words: int = 120):
        self.max_words = max_words
        self.sentence_chunker = SentenceAwareChunker(max_words=max_words)

    def chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []

        for paragraph in paragraphs:
            words = paragraph.split()
            if len(words) <= self.max_words:
                chunks.append(paragraph)
            else:
                sub_chunks = self.sentence_chunker.chunk_text(paragraph)
                chunks.extend(sub_chunks)

        return chunks
