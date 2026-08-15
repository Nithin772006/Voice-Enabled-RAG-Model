import sys
import logging
from typing import List, Dict, Any, Union
from pathlib import Path

# Resolve local imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from guardrails.schemas import RetrievalValidationResult, ValidationStatus
from guardrails.policies import MIN_SIMILARITY_SCORE
from llm.schemas import RetrievedChunk, ChunkMetadata

# Reconfigure stdout to support UTF-8 on Windows terminal/console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

logger = logging.getLogger("RetrievalGuard")

def validate_retrieved_chunks(
    chunks: List[Union[Dict[str, Any], Any]]
) -> RetrievalValidationResult:
    """Validates the retrieved database hits by filtering low similarity scores,
    deduplicating passages, and detecting zero-evidence scenarios.
    """
    if not chunks:
        logger.warning("Retriever returned an empty list.")
        return RetrievalValidationResult(
            is_valid=False,
            status=ValidationStatus.LOW_CONFIDENCE,
            reason="No context chunks were retrieved from the database.",
            filtered_chunks=[]
        )

    validated_chunks = []
    seen_texts = set()

    for idx, raw_chunk in enumerate(chunks):
        # 1. Adapt both dictionary and Pydantic object formats
        if isinstance(raw_chunk, dict):
            chunk_id = raw_chunk.get("chunk_id", f"chk_{idx}")
            text = raw_chunk.get("text", "").strip()
            score = raw_chunk.get("score", 0.0)
            lang = raw_chunk.get("language", "unknown")
            doc_id = raw_chunk.get("document_id", "unknown_doc")
        else:
            chunk_id = getattr(raw_chunk, "chunk_id", f"chk_{idx}")
            text = getattr(raw_chunk, "text", "").strip()
            # If Pydantic model doesn't have score on outer level, search hit metadata or assume 1.0
            score = getattr(raw_chunk, "score", 1.0)
            lang = getattr(raw_chunk, "language", "unknown")
            doc_id = getattr(raw_chunk, "document_id", "unknown_doc")

        # 2. Filter out empty or whitespace-only chunks
        if not text:
            logger.warning(f"Discarding empty chunk {chunk_id}.")
            continue

        # 3. Similarity Score Filtering (Layer 3)
        # Discard any chunk with a similarity score below our policy minimum
        if score < MIN_SIMILARITY_SCORE:
            logger.warning(
                f"Filtering out chunk {chunk_id} due to low similarity score: {score:.4f} "
                f"(Threshold: {MIN_SIMILARITY_SCORE:.2f})"
            )
            continue

        # 4. Deduplication
        # Remove identical passages or repeat chunks to save LLM context window space
        norm_text = " ".join(text.lower().split())
        if norm_text in seen_texts:
            logger.info(f"Discarding duplicate chunk {chunk_id}.")
            continue
        seen_texts.add(norm_text)

        # Convert valid chunk into a standard Pydantic RetrievedChunk model
        validated_chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                language=lang,
                text=text,
                metadata=ChunkMetadata(selected=True)
            )
        )

    # 5. No-Evidence Handling (Layer 4)
    # If no retrieved passages survived the security filters, flag validation failure
    if not validated_chunks:
        logger.warning("Zero chunks survived retrieval filters. System enters low confidence mode.")
        return RetrievalValidationResult(
            is_valid=False,
            status=ValidationStatus.LOW_CONFIDENCE,
            reason="All retrieved chunks were discarded due to low similarity confidence scores.",
            filtered_chunks=[]
        )

    return RetrievalValidationResult(
        is_valid=True,
        status=ValidationStatus.SAFE,
        filtered_chunks=validated_chunks
    )

if __name__ == "__main__":
    # Test suite for retrieval validation
    logging.basicConfig(level=logging.INFO)
    print("=== RUNNING RETRIEVED CHUNKS VALIDATION TESTS ===\n")

    # Mock outputs representing high-quality search
    good_hits = [
        {"chunk_id": "chunk_001", "score": 0.85, "language": "tamil", "text": "வியாழன் சூரிய குடும்பத்தின் மிகப்பெரிய கோள் ஆகும்."},
        {"chunk_id": "chunk_002", "score": 0.82, "language": "tamil", "text": "வியாழன் சூரிய குடும்பத்தின் மிகப்பெரிய கோள் ஆகும்."} # Duplicate
    ]

    # Mock outputs representing low similarity (out-of-domain query search results)
    poor_hits = [
        {"chunk_id": "chunk_003", "score": 0.05, "language": "tamil", "text": "இணைப்பு வெற்றிகரமாக முடிந்தது."}
    ]

    print("Test 1: Validating High-Confidence Hits...")
    res_good = validate_retrieved_chunks(good_hits)
    print(f"Result is_valid: {res_good.is_valid} | Status: {res_good.status}")
    print(f"Filtered Chunks Count: {len(res_good.filtered_chunks)} (Expected: 1 due to deduplication)\n")

    print("Test 2: Validating Low-Confidence Hits (No-Evidence scenario)...")
    res_poor = validate_retrieved_chunks(poor_hits)
    print(f"Result is_valid: {res_poor.is_valid} | Status: {res_poor.status}")
    print(f"Reason: {res_poor.reason}\n")
