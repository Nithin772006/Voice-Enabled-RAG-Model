import re
from typing import List, Dict, Any, Tuple
from src.config.settings import DEFAULT_GUARDRAIL_THRESHOLD

class ConfidenceGuardrail:
    """Retrieval confidence guardrail to reject low-similarity / low-evidence queries."""
    
    def __init__(self, threshold: float = DEFAULT_GUARDRAIL_THRESHOLD):
        self.threshold = threshold

    def _calculate_query_coverage(self, query: str, passage: str) -> float:
        q_words = set(re.findall(r'\w+', query.lower()))
        if not q_words:
            return 1.0
        p_words = set(re.findall(r'\w+', passage.lower()))
        matched = q_words.intersection(p_words)
        return len(matched) / float(len(q_words))

    def evaluate(self, top_candidates: List[Dict[str, Any]], query: str) -> Tuple[bool, float, Dict[str, Any]]:
        if not top_candidates:
            return False, 0.0, {"reason": "NO_CANDIDATES"}

        top_doc = top_candidates[0]
        v_score = top_doc.get("vector_score", 0.0)
        rerank_score = top_doc.get("reranker_score", v_score)
        rrf_score = top_doc.get("rrf_score", 0.0)
        
        # Combined confidence metric
        confidence = max(v_score, rerank_score, rrf_score * 30.0)
        coverage = self._calculate_query_coverage(query, top_doc.get("text", ""))

        signals = {
            "confidence_score": round(float(confidence), 4),
            "query_coverage": round(float(coverage), 4),
            "top_vector_score": round(float(v_score), 4),
            "threshold": self.threshold
        }

        is_passed = confidence >= self.threshold or coverage >= 0.40
        return is_passed, confidence, signals

    def get_refusal_message(self) -> str:
        return "மன்னிக்கவும், வழங்கப்பட்ட தரவுத்தளத்தில் இந்த கேள்விக்கு போதிய ஆதாரங்கள் கிடைக்கவில்லை."
