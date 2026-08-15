import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Resolve local imports cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph.state import RAGGraphState
from retrieval.retriever import QdrantRetriever
from guardrails.retrieval_guard import validate_retrieved_chunks
from guardrails.input_validator import validate_query
from guardrails.query_guard import evaluate_query_scope
from guardrails.output_guard import validate_llm_output
from llm.prompt_builder import PromptBuilder
from llm.generator import RAGGenerator
from tts.engine import TTSEngine

logger = logging.getLogger("GraphNodes")

# Pre-translated localized safe fallbacks
FALLBACK_RESPONSES = {
    "english": "I couldn't find enough relevant information in the available knowledge base to answer that question.",
    "tamil": "வழங்கப்பட்டுள்ள அறிவுத் தளத்தில் இந்த வினாவிற்குப் பதிலளிக்கத் தேவையான போதுமான தகவல்கள் கிடைக்கவில்லை.",
    "hindi": "मुझे उपलब्ध ज्ञान आधार में उस प्रश्न का उत्तर देने के लिए पर्याप्त प्रासंगिक जानकारी नहीं मिली।",
    "telugu": "అందుబాటులో ఉన్న నాలెడ్జ్ బేస్ లో ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి సరిపడా సమాచారం లభించలేదు."
}

# Module-level singletons for lazy-loading resource-heavy engines
_retriever = None
_generator = None
_tts_engine = None

def get_retriever() -> QdrantRetriever:
    global _retriever
    if _retriever is None:
        _retriever = QdrantRetriever()
    return _retriever

def get_generator() -> RAGGenerator:
    global _generator
    if _generator is None:
        _generator = RAGGenerator()
    return _generator

def get_tts_engine() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine

# =================================================================
# 1. INPUT PROCESSOR NODE
# =================================================================
def input_processor_node(state: RAGGraphState) -> Dict[str, Any]:
    """Validates the query syntax, length, and scope. Detects or maps the target language."""
    logger.info("--- [NODE] InputProcessorNode ---")
    query = state.get("query")
    language = state.get("language", "tamil").lower().strip()
    
    # 1. Check supported languages
    if language not in FALLBACK_RESPONSES:
        logger.error(f"Unsupported language requested: '{language}'")
        return {
            "error": f"Language '{language}' is not supported.",
            "status": "error",
            "answer_valid": False
        }
        
    # 2. Syntax and Injection Validation (Layer 1 Guardrails)
    input_validation = validate_query(query)
    if not input_validation.is_valid:
        logger.warning(f"Input validation rejected: {input_validation.reason}")
        return {
            "query": query,
            "language": language,
            "error": input_validation.reason,
            "status": "error",
            "answer_valid": False
        }
        
    # 3. Functional Scope Evaluation (Layer 2 Guardrails)
    scope_validation = evaluate_query_scope(input_validation.sanitized_query)
    if not scope_validation.is_valid:
        logger.warning(f"Scope evaluation rejected: {scope_validation.reason}")
        # Mark as fallback, save scope refusal text as the answer, bypass retrieval
        return {
            "query": input_validation.sanitized_query,
            "language": language,
            "answer": scope_validation.reason,
            "answer_valid": True,
            "retrieval_passed": False,
            "status": "fallback"
        }
        
    # 4. Input Success
    return {
        "query": input_validation.sanitized_query,
        "language": language,
        "status": "success",
        "error": None
    }

# =================================================================
# 2. RETRIEVAL NODE
# =================================================================
def retrieval_node(state: RAGGraphState) -> Dict[str, Any]:
    """Retrieves document chunks from the local Qdrant collection using real semantic embeddings."""
    logger.info("--- [NODE] RetrievalNode ---")
    if state.get("status") == "error":
        return {}
        
    try:
        retriever = get_retriever()
        retrieved_chunks = retriever.retrieve(state["query"], top_k=2)
        scores = [chunk["score"] for chunk in retrieved_chunks]
        return {
            "retrieved_chunks": retrieved_chunks,
            "retrieval_scores": scores
        }
    except Exception as e:
        logger.error(f"Failed during Qdrant retrieval: {e}")
        return {
            "error": f"Retrieval failed: {e}",
            "status": "error"
        }

# =================================================================
# 3. GUARDRAIL NODE
# =================================================================
def guardrail_node(state: RAGGraphState) -> Dict[str, Any]:
    """Validates retrieval confidence scores and filters out low confidence or duplicate chunks."""
    logger.info("--- [NODE] GuardrailNode ---")
    if state.get("status") == "error":
        return {}
        
    chunks = state.get("retrieved_chunks", [])
    validation = validate_retrieved_chunks(chunks)
    
    if not validation.is_valid:
        logger.warning(f"Retrieval guardrail rejected context: {validation.reason}")
        return {
            "retrieval_passed": False,
            "retrieved_chunks": [],
            "status": "fallback"
        }
        
    # Successfully passed similarity threshold
    return {
        "retrieval_passed": True,
        "retrieved_chunks": validation.filtered_chunks,
        "status": "success"
    }

# =================================================================
# 4. PROMPT BUILDER NODE
# =================================================================
def prompt_builder_node(state: RAGGraphState) -> Dict[str, Any]:
    """Formulates System & User instruction templates isolated inside XML tags."""
    logger.info("--- [NODE] PromptBuilderNode ---")
    if state.get("status") == "error" or not state.get("retrieval_passed"):
        return {}
        
    messages = PromptBuilder.build_prompt_messages(
        query=state["query"],
        retrieved_chunks=state["retrieved_chunks"],
        language=state["language"]
    )
    return {
        "prompt": messages
    }

# =================================================================
# 5. LLM GENERATION NODE
# =================================================================
def llm_generation_node(state: RAGGraphState) -> Dict[str, Any]:
    """Submits the grounded prompt to the Sarvam-105B LLM wrapper and runs Output Guardrails."""
    logger.info("--- [NODE] LLMGenerationNode ---")
    if state.get("status") == "error" or not state.get("retrieval_passed"):
        return {}
        
    try:
        generator = get_generator()
        answer = generator.model.generate(state["prompt"])
        
        # Integrate Output Guardrail validation immediately
        logger.info("Running Output Guardrail checks on generated answer...")
        output_validation = validate_llm_output(answer)
        
        if not output_validation.is_valid:
            logger.warning(f"Output validation failed: {output_validation.reason}")
            return {
                "error": output_validation.reason,
                "answer_valid": False,
                "status": "fallback"
            }
            
        # Parse sources from context chunks
        sources = [
            {
                "chunk_id": chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk.get("chunk_id"),
                "language": chunk.language if hasattr(chunk, "language") else chunk.get("language")
            }
            for chunk in state["retrieved_chunks"]
        ]
        
        return {
            "answer": answer.strip(),
            "answer_valid": True,
            "sources": sources,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        return {
            "error": f"LLM generation failed: {e}",
            "answer_valid": False,
            "status": "fallback"
        }

# =================================================================
# 6. FALLBACK NODE
# =================================================================
def fallback_node(state: RAGGraphState) -> Dict[str, Any]:
    """Generates localized pre-translated safe responses when retrieval confidence or generation fails."""
    logger.info("--- [NODE] FallbackNode ---")
    
    # If the state already has an answer (e.g. from out-of-scope query guard), preserve it
    answer = state.get("answer")
    language = state.get("language", "tamil").lower().strip()
    
    if not answer or not answer.strip():
        # Load the default pre-translated generic RAG fallback
        answer = FALLBACK_RESPONSES.get(language, FALLBACK_RESPONSES["english"])
        
    return {
        "answer": answer,
        "answer_valid": True,
        "sources": [],
        "status": "fallback"
    }

# =================================================================
# 7. TTS NODE
# =================================================================
def tts_node(state: RAGGraphState) -> Dict[str, Any]:
    """Calls TTSEngine to synthesize the output answer into a cacheable WAV speech file."""
    logger.info("--- [NODE] TTSNode ---")
    # Skip TTS only if status is absolute error or answer is invalid
    if state.get("status") == "error" or not state.get("answer_valid") or not state.get("answer"):
        return {}
        
    try:
        tts_engine = get_tts_engine()
        query_hash = abs(hash(state["query"]))
        output_file = f"data/audio_responses/response_{query_hash}_{state['language']}.wav"
        
        audio_path = tts_engine.synthesize(
            text=state["answer"],
            language=state["language"],
            output_path=output_file
        )
        return {
            "audio_path": audio_path
        }
    except Exception as e:
        logger.error(f"Failed during TTS voice generation: {e}")
        # Graph returns a meaningful state instead of crashing
        return {
            "error": f"TTS synthesis failed: {e}",
            "audio_path": None
        }
