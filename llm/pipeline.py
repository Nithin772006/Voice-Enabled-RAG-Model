import sys
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add project root to sys.path to resolve local imports cleanly when loaded as a module
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.retriever import QdrantRetriever
from llm.generator import RAGGenerator
from guardrails.retrieval_guard import validate_retrieved_chunks
from tts.engine import TTSEngine

logger = logging.getLogger("RAGPipeline")

# Pre-translated localized safe fallback responses to guarantee 100% accurate grounding
# on out-of-domain/unanswerable requests without relying on fragile LLM generation.
FALLBACK_RESPONSES = {
    "english": "I couldn't find enough relevant information in the available knowledge base to answer that question.",
    "tamil": "வழங்கப்பட்டுள்ள அறிவுத் தளத்தில் இந்த வினாவிற்குப் பதிலளிக்கத் தேவையான போதுமான தகவல்கள் கிடைக்கவில்லை.",
    "hindi": "मुझे उपलब्ध ज्ञान आधार में उस प्रश्न का उत्तर देने के लिए पर्याप्त प्रासंगिक जानकारी नहीं मिली।",
    "telugu": "అందుబాటులో ఉన్న నాలెడ్జ్ బేస్ లో ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి సరిపడా సమాచారం లభించలేదు."
}

class RAGPipeline:
    """A clean, high-level interface coordinating retrieval, LLM generation, and Text-to-Speech."""
    def __init__(
        self, 
        retriever: Optional[QdrantRetriever] = None, 
        generator: Optional[RAGGenerator] = None,
        tts_engine: Optional[TTSEngine] = None
    ) -> None:
        self.retriever = retriever or QdrantRetriever()
        self.generator = generator or RAGGenerator()
        self.tts_engine = tts_engine or TTSEngine()

    def answer(self, query: str, language: str = "tamil", top_k: int = 2, generate_voice: bool = True) -> Dict[str, Any]:
        """Runs the complete RAG loop and returns a structured response containing the answer, sources, and audio response file path."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
            
        # 1. Retrieve relevant chunks from Qdrant
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        # 2. Run Retrieval Validation (Layer 3 & 4 Guardrails)
        validation = validate_retrieved_chunks(retrieved_chunks)
        
        # Determine the raw text answer first
        if not validation.is_valid:
            # Bypass LLM invocation completely for zero-evidence states
            logger.warning("Retrieval validation failed. Triggering short-circuit fallback response.")
            lang_key = language.lower()
            answer = FALLBACK_RESPONSES.get(
                lang_key, 
                FALLBACK_RESPONSES["english"]
            )
            sources = []
        else:
            # 3. Generate grounded answer using only validated high-confidence chunks
            answer = self.generator.generate(
                query=query,
                retrieved_chunks=validation.filtered_chunks,
                language=language
            )
            # Format structured metadata sources
            sources = [
                {
                    "chunk_id": chunk.chunk_id,
                    "score": next((h["score"] for h in retrieved_chunks if h["chunk_id"] == chunk.chunk_id), 1.0),
                    "language": chunk.language
                }
                for chunk in validation.filtered_chunks
            ]

        # 4. Generate Voice (TTS synthesis) if requested
        audio_path = None
        if generate_voice:
            # Hash query and language to generate deterministic and cacheable file paths
            query_hash = abs(hash(query))
            output_file = f"data/audio_responses/response_{query_hash}_{language.lower()}.wav"
            try:
                audio_path = self.tts_engine.synthesize(
                    text=answer,
                    language=language,
                    output_path=output_file
                )
            except Exception as e:
                logger.error(f"TTS synthesis failed during pipeline execution: {e}")
                audio_path = None
        
        # 5. Return unified payload
        return {
            "query": query,
            "answer": answer,
            "language": language,
            "sources": sources,
            "audio_path": audio_path
        }
