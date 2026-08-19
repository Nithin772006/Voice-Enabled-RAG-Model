from typing import Dict, Any, List
import uuid

class ChunkMetadata:
    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        language: str,
        source: str,
        chunk_strategy: str,
        chunk_index: int,
        text: str,
        original_record_id: str,
        passage_idx: int = 0,
        is_selected: int = 0,
        extra: Dict[str, Any] = None
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.language = language
        self.source = source
        self.chunk_strategy = chunk_strategy
        self.chunk_index = chunk_index
        self.text = text
        self.original_record_id = str(original_record_id)
        self.passage_idx = passage_idx
        self.is_selected = is_selected
        self.extra = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "language": self.language,
            "source": self.source,
            "chunk_strategy": self.chunk_strategy,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "original_record_id": self.original_record_id,
            "passage_idx": self.passage_idx,
            "is_selected": self.is_selected,
            "question": self.extra.get("query", "") or self.extra.get("question", ""),
            "answer": self.extra.get("answer", "")
        }
        if self.extra:
            res.update(self.extra)
        return res

class BaseChunker:
    """Abstract Base Class for Chunking Strategies."""
    
    def __init__(self, name: str):
        self.name = name

    def chunk_text(self, text: str, doc_metadata: Dict[str, Any]) -> List[ChunkMetadata]:
        raise NotImplementedError("Subclasses must implement chunk_text.")

    def chunk_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk a list of normalized MSMARCO-XI dataset records into structured chunk dicts."""
        all_chunks = []
        for record in records:
            trans_passages = record.get("translated_passages", [])
            sel_flags = record.get("is_selected", [])
            query_id = record.get("query_id", "")
            lang = record.get("target_lang", "tam_Taml")
            query_text = record.get("query", "")
            answer_text = record.get("answer", "")

            for p_idx, passage in enumerate(trans_passages):
                if not passage or not passage.strip():
                    continue
                    
                is_sel = sel_flags[p_idx] if p_idx < len(sel_flags) else 0
                doc_id = f"doc_{query_id}_{p_idx}"
                doc_meta = {
                    "document_id": doc_id,
                    "original_record_id": str(query_id),
                    "language": lang,
                    "source": "MSMARCO-XI",
                    "passage_idx": p_idx,
                    "is_selected": is_sel,
                    "query": query_text,
                    "answer": answer_text
                }
                chunks = self.chunk_text(passage, doc_meta)
                for chunk in chunks:
                    c_dict = chunk.to_dict()
                    c_dict["question"] = query_text
                    c_dict["answer"] = answer_text
                    all_chunks.append(c_dict)
                    
        return all_chunks
