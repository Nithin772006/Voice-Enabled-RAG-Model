import re
import time
import uuid
import logging
from typing import Dict, Any, Optional, Tuple, List
from src.stt.sarvam_stt import SarvamSTTClient
from src.query.processor import QueryProcessor
from src.indexing.manager import IndexManager
from src.retrieval.hybrid import HybridRetriever
from src.guardrails.off_topic import OffTopicGuardrail
from src.guardrails.confidence import ConfidenceGuardrail
from src.guardrails.grounding import GroundingValidator
from src.embeddings.cache import QueryCache
from src.config.settings import (
    DEFAULT_TOP_K,
    DEFAULT_CANDIDATE_K,
    DEFAULT_GUARDRAIL_THRESHOLD
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VoiceRAGPipeline")

class VoiceRAGPipeline:
    """Orchestration layer connecting Voice STT, Query Processing, Fast Semantic QA Retrieval, Quality Fallback & Generation."""
    
    def __init__(self, index_manager: Optional[IndexManager] = None):
        self.index_manager = index_manager or IndexManager()
        self.stt_client = SarvamSTTClient()
        self.query_processor = QueryProcessor()
        self.retriever: Optional[HybridRetriever] = None
        self._reranker = None
        self._llm = None
        self.query_cache = QueryCache(max_capacity=2000)
        
        self.off_topic_guardrail = OffTopicGuardrail()
        self.confidence_guardrail = ConfidenceGuardrail(threshold=DEFAULT_GUARDRAIL_THRESHOLD)
        self.grounding_validator = GroundingValidator()

    @property
    def reranker(self):
        if self._reranker is None:
            from src.reranking.cross_encoder import MultilingualReranker
            self._reranker = MultilingualReranker()
        return self._reranker

    @property
    def llm(self):
        if self._llm is None:
            from src.generation.llm import GroundedLLM
            self._llm = GroundedLLM()
        return self._llm

    def initialize_system(self, force_rebuild: bool = False, records: Optional[list] = None) -> bool:
        """Load index from cache or build new index from records."""
        if not force_rebuild and self.index_manager.load_cache():
            logger.info("[PIPELINE] Loaded existing vector index from disk cache.")
        else:
            if not records:
                logger.warning("[PIPELINE] No records provided to build index.")
                return False
            logger.info("[PIPELINE] Building index from dataset records...")
            self.index_manager.build_and_save(records, chunk_strategy="sentence")

        self.retriever = HybridRetriever(
            self.index_manager.dense_indexer,
            self.index_manager.sparse_indexer,
            self.index_manager.chunks
        )
        return True

    def _extract_best_sentence(self, query: str, text: str) -> str:
        """Extract the sentence from text that best matches query terms."""
        if not text:
            return ""
        
        sentences = [s.strip() for s in re.split(r'[\.\!\?\n]+', text) if len(s.strip()) > 5]
        if not sentences:
            return text.strip()
            
        q_words = set(re.split(r'[^\w]+', query.lower(), flags=re.UNICODE))
        best_sentence = sentences[0]
        max_overlap = -1

        for sent in sentences:
            sent_words = set(re.split(r'[^\w]+', sent.lower(), flags=re.UNICODE))
            overlap = len(q_words.intersection(sent_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = sent

        return best_sentence

    def run_text_query(
        self,
        query: str,
        mode: str = "fast",
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        alpha: float = 0.70
    ) -> Dict[str, Any]:
        """Execute RAG pipeline with Fast Mode (<200ms precomputed QA) and Quality Mode Fallback."""
        t_request_start = time.perf_counter()
        request_id = str(uuid.uuid4())
        mode = mode.lower().strip()
        
        # 0. Instant Query Cache check (< 1ms)
        cached_resp = self.query_cache.get(query, mode)
        if cached_resp:
            res_copy = dict(cached_resp)
            res_copy["request_id"] = request_id
            res_copy["cache_hit"] = True
            return res_copy

        latencies = {
            "stt_ms": 0.0,
            "query_proc_ms": 0.0,
            "embedding_ms": 0.0,
            "dense_search_ms": 0.0,
            "sparse_search_ms": 0.0,
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "guardrail_ms": 0.0,
            "generation_ms": 0.0,
            "grounding_ms": 0.0,
            "total_latency_ms": 0.0
        }

        # 1. Query Processing
        q_info, proc_ms = self.query_processor.process_query(query)
        latencies["query_proc_ms"] = round(proc_ms, 2)
        
        normalized_q = q_info["normalized_query"]
        lang = q_info["language"]
        q_class = q_info["classification"]

        # 2. Safety Guardrail
        passed_safety, safety_msg = self.off_topic_guardrail.evaluate(q_class)
        if not passed_safety:
            total_ms = (time.perf_counter() - t_request_start) * 1000.0
            latencies["total_latency_ms"] = round(total_ms, 2)
            return self._build_response(
                request_id=request_id,
                query=query,
                normalized_query=normalized_q,
                language=lang,
                answer=safety_msg,
                sources=[],
                confidence=0.0,
                status="REJECTED_OFF_TOPIC",
                latencies=latencies,
                mode=mode,
                answer_mode="rejected",
                is_grounded=False
            )

        if self.retriever is None:
            total_ms = (time.perf_counter() - t_request_start) * 1000.0
            latencies["total_latency_ms"] = round(total_ms, 2)
            return self._build_response(
                request_id=request_id,
                query=query,
                normalized_query=normalized_q,
                language=lang,
                answer="Index not initialized.",
                sources=[],
                confidence=0.0,
                status="SYSTEM_ERROR",
                latencies=latencies,
                mode=mode,
                answer_mode="error",
                is_grounded=False
            )

        # 3. FAST MODE vs QUALITY MODE Branching
        if mode == "fast":
            # FAST MODE: Optimized Pre-computed FAISS Vector Search
            candidates, ret_timings, ret_debug = self.retriever.retrieve(
                normalized_q,
                candidate_k=candidate_k,
                top_k=top_k,
                alpha=1.0
            )
            latencies["embedding_ms"] = round(ret_timings.get("embedding_ms", 0.0), 2)
            latencies["dense_search_ms"] = round(ret_timings.get("dense_search_ms", 0.0), 2)
            latencies["retrieval_ms"] = round(ret_timings.get("total_retrieval_ms", 0.0), 2)
            top_chunks = candidates[:top_k]
            rerank_debug = []

            # Check Confidence for Fast Mode Precomputed Answer Selection
            t_guard = time.perf_counter()
            passed_conf, conf_score, guard_signals = self.confidence_guardrail.evaluate(top_chunks, normalized_q)
            latencies["guardrail_ms"] = round((time.perf_counter() - t_guard) * 1000.0, 2)

            if passed_conf and top_chunks:
                top_chunk = top_chunks[0]
                rec_ans = top_chunk.get("answer", "") or top_chunk.get("extra", {}).get("answer", "")
                
                invalid_answers = ["no answer present", "எந்த பதிலும் இல்லை."]
                if rec_ans and rec_ans.strip() and rec_ans.strip().lower() not in invalid_answers:
                    final_answer = rec_ans.strip()
                    answer_mode = "retrieved"
                else:
                    final_answer = self._extract_best_sentence(normalized_q, top_chunk.get("text", ""))
                    answer_mode = "retrieved_passage"

                latencies["generation_ms"] = 0.0
                
                # Grounding validation
                t_ground = time.perf_counter()
                is_grounded, grounding_score, g_case = self.grounding_validator.evaluate(final_answer, top_chunks)
                latencies["grounding_ms"] = round((time.perf_counter() - t_ground) * 1000.0, 2)

                total_ms = (time.perf_counter() - t_request_start) * 1000.0
                latencies["total_latency_ms"] = round(total_ms, 2)

                response = self._build_response(
                    request_id=request_id,
                    query=query,
                    normalized_query=normalized_q,
                    language=lang,
                    answer=final_answer,
                    sources=self._format_sources(top_chunks),
                    confidence=conf_score,
                    status="SUCCESS" if is_grounded else "FAILED_GROUNDING_CHECK",
                    latencies=latencies,
                    guardrail_signals=guard_signals,
                    mode=mode,
                    answer_mode=answer_mode,
                    is_grounded=is_grounded,
                    retrieval_debug=ret_debug,
                    rerank_debug=rerank_debug
                )
                self.query_cache.put(query, mode, response)
                return response
            else:
                # Low confidence in Fast Mode -> FALLBACK TO QUALITY MODE
                logger.info(f"[PIPELINE] Fast Mode confidence {conf_score:.2f} < threshold. Falling back to Quality Mode...")
                mode = "quality_fallback"

        # QUALITY MODE / QUALITY FALLBACK PATH
        candidates, ret_timings, ret_debug = self.retriever.retrieve(
            normalized_q,
            candidate_k=candidate_k,
            top_k=candidate_k,
            alpha=alpha
        )
        latencies["embedding_ms"] = round(ret_timings.get("embedding_ms", 0.0), 2)
        latencies["dense_search_ms"] = round(ret_timings.get("dense_search_ms", 0.0), 2)
        latencies["sparse_search_ms"] = round(ret_timings.get("sparse_search_ms", 0.0), 2)
        latencies["retrieval_ms"] = round(ret_timings.get("total_retrieval_ms", 0.0), 2)

        top_chunks, rerank_ms, rerank_debug = self.reranker.rerank(
            normalized_q,
            candidates,
            top_k=3,
            max_rerank_candidates=5,
            enabled=True
        )
        latencies["rerank_ms"] = round(rerank_ms, 2)

        t_guard = time.perf_counter()
        passed_conf, conf_score, guard_signals = self.confidence_guardrail.evaluate(top_chunks, normalized_q)
        latencies["guardrail_ms"] = round((time.perf_counter() - t_guard) * 1000.0, 2)

        if not passed_conf:
            total_ms = (time.perf_counter() - t_request_start) * 1000.0
            latencies["total_latency_ms"] = round(total_ms, 2)
            refusal_msg = self.confidence_guardrail.get_refusal_message()
            return self._build_response(
                request_id=request_id,
                query=query,
                normalized_query=normalized_q,
                language=lang,
                answer=refusal_msg,
                sources=self._format_sources(top_chunks),
                confidence=conf_score,
                status="INSUFFICIENT_EVIDENCE",
                latencies=latencies,
                guardrail_signals=guard_signals,
                mode=mode,
                answer_mode="refusal",
                is_grounded=False,
                retrieval_debug=ret_debug,
                rerank_debug=rerank_debug
            )

        raw_answer, gen_ms = self.llm.generate_answer(normalized_q, top_chunks, max_new_tokens=40)
        latencies["generation_ms"] = round(gen_ms, 2)

        t_ground = time.perf_counter()
        is_grounded, grounding_score, g_case = self.grounding_validator.evaluate(raw_answer, top_chunks)
        latencies["grounding_ms"] = round((time.perf_counter() - t_ground) * 1000.0, 2)

        final_answer = raw_answer if is_grounded else self.grounding_validator.get_grounding_failure_message()
        status = "SUCCESS" if is_grounded else "FAILED_GROUNDING_CHECK"

        total_ms = (time.perf_counter() - t_request_start) * 1000.0
        latencies["total_latency_ms"] = round(total_ms, 2)

        guard_signals["grounding_case"] = g_case
        guard_signals["grounding_score"] = grounding_score

        answer_mode = "quality_fallback" if mode == "quality_fallback" else "generated"

        response = self._build_response(
            request_id=request_id,
            query=query,
            normalized_query=normalized_q,
            language=lang,
            answer=final_answer,
            sources=self._format_sources(top_chunks),
            confidence=conf_score,
            status=status,
            latencies=latencies,
            guardrail_signals=guard_signals,
            mode=mode,
            answer_mode=answer_mode,
            is_grounded=is_grounded,
            retrieval_debug=ret_debug,
            rerank_debug=rerank_debug
        )

        self.query_cache.put(query, mode, response)
        return response

    def run_voice_query(
        self,
        audio_bytes: bytes,
        filename: str = "input.wav",
        mode: str = "fast",
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        alpha: float = 0.70
    ) -> Dict[str, Any]:
        """Execute STT + RAG pipeline for audio input."""
        transcript, detected_lang, stt_ms = self.stt_client.transcribe_audio(
            audio_bytes,
            filename=filename
        )

        if not transcript:
            return {
                "request_id": str(uuid.uuid4()),
                "transcript": "",
                "answer": "மன்னிக்கவும், உங்கள் குரல் தெளிவாகக் கேட்கவில்லை.",
                "status": "STT_FAILURE",
                "latencies": {"stt_ms": round(stt_ms, 2), "total_latency_ms": round(stt_ms, 2)}
            }

        response = self.run_text_query(
            query=transcript,
            mode=mode,
            top_k=top_k,
            candidate_k=candidate_k,
            alpha=alpha
        )
        response["latencies_ms"]["stt_ms"] = round(stt_ms, 2)
        response["latencies_ms"]["total_latency_ms"] = round(
            response["latencies_ms"]["total_latency_ms"] + stt_ms, 2
        )
        response["transcript"] = transcript
        return response

    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted = []
        for idx, c in enumerate(chunks, 1):
            formatted.append({
                "rank": idx,
                "chunk_id": c.get("chunk_id", ""),
                "text": c.get("text", ""),
                "score": round(float(c.get("reranker_score", c.get("combined_score", c.get("vector_score", 0.0)))), 4),
                "original_record_id": c.get("original_record_id", ""),
                "passage_idx": c.get("passage_idx", 0),
                "is_selected": c.get("is_selected", 0)
            })
        return formatted

    def _build_response(
        self,
        request_id: str,
        query: str,
        normalized_query: str,
        language: str,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence: float,
        status: str,
        latencies: Dict[str, float],
        guardrail_signals: Optional[Dict[str, Any]] = None,
        mode: str = "fast",
        answer_mode: str = "generated",
        is_grounded: bool = True,
        retrieval_debug: Optional[Dict[str, Any]] = None,
        rerank_debug: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        return {
            "request_id": request_id,
            "query": query,
            "normalized_query": normalized_query,
            "language": language,
            "mode": mode,
            "answer_mode": answer_mode,
            "answer": answer,
            "sources": sources,
            "confidence": round(float(confidence), 4),
            "is_grounded": is_grounded,
            "status": status,
            "cache_hit": False,
            "latencies_ms": latencies,
            "guardrail_signals": guardrail_signals or {},
            "retrieval_debug": retrieval_debug or {},
            "rerank_debug": rerank_debug or []
        }
