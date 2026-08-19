import re
from typing import List, Dict, Any, Tuple
from rag_system.config import (
    DEFAULT_GUARDRAIL_THRESHOLD,
    GUARDRAIL_MIN_MARGIN,
    GUARDRAIL_MIN_QUERY_COVERAGE,
    GUARDRAIL_MIN_ANCHOR_OVERLAP,
)


def is_tamil_text(text: str) -> bool:
    """Detect if text contains Tamil Unicode range (U+0B80 to U+0BFF)."""
    return bool(re.search(r"[\u0B80-\u0BFF]", text))


class ConfidenceGuardrail:
    """
    Confidence Guardrail enforcing evidence quality.
    Threshold is empirically calibrated on validation dataset scores.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_GUARDRAIL_THRESHOLD,
        min_margin: float = GUARDRAIL_MIN_MARGIN,
        min_query_coverage: float = GUARDRAIL_MIN_QUERY_COVERAGE,
        min_anchor_overlap: int = GUARDRAIL_MIN_ANCHOR_OVERLAP,
    ):
        self.threshold = threshold
        self.min_margin = min_margin
        self.min_query_coverage = min_query_coverage
        self.min_anchor_overlap = min_anchor_overlap

    @staticmethod
    def _normalize_token(token: str) -> str:
        for suffix in (
            "த்தின்", "யின்", "இன்", "க்கு", "க்கான", "களின்", "ங்கள்", "கள்",
            "ஆன", "வது", "த்தில்", "த்திலிருந்து", "மாக", "ம்", "து",
        ):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        return token

    @staticmethod
    def _tokens(text: str) -> List[str]:
        tokens = re.findall(r"[\w\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0B00-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F]+", text.lower())
        stopwords = {
            "what", "who", "when", "where", "why", "how", "is", "are", "the", "a", "an",
            "என்ன", "எது", "யார்", "எப்போது", "எங்கே", "ஏன்", "எப்படி", "ஆகும்",
        }
        return [ConfidenceGuardrail._normalize_token(t) for t in tokens if len(t) >= 3 and t not in stopwords]

    def _coverage(self, query: str, text: str) -> Tuple[float, int]:
        query_tokens = set(self._tokens(query))
        if not query_tokens:
            return 0.0, 0
        evidence_tokens = set(self._tokens(text))
        overlap = query_tokens.intersection(evidence_tokens)
        return len(overlap) / max(1, len(query_tokens)), len(overlap)

    def evaluate(self, top_evidence: List[Dict[str, Any]], query: str = "") -> Tuple[bool, float]:
        """
        Check whether retrieved evidence is strong enough to allow generation.
        Returns (is_passed, max_confidence_score).
        """
        if not top_evidence:
            return False, 0.0

        top_score = top_evidence[0].get("combined_score", top_evidence[0].get("reranker_score", 0.0))
        second_score = (
            top_evidence[1].get("combined_score", top_evidence[1].get("reranker_score", 0.0))
            if len(top_evidence) > 1
            else 0.0
        )
        margin = top_score - second_score
        coverage, anchor_overlap = self._coverage(query, top_evidence[0].get("text", "")) if query else (1.0, self.min_anchor_overlap)

        top_evidence[0]["guardrail_signals"] = {
            "top_score": float(top_score),
            "reranker_score": float(top_evidence[0].get("reranker_score", 0.0)),
            "second_score": float(second_score),
            "score_margin": float(margin),
            "query_coverage": float(coverage),
            "anchor_overlap": int(anchor_overlap),
        }

        is_passed = (
            top_score >= self.threshold
            and coverage >= self.min_query_coverage
            and anchor_overlap >= self.min_anchor_overlap
            and (margin >= self.min_margin or second_score >= self.threshold)
        )
        return is_passed, top_score

    def get_refusal_message(self, query: str) -> str:
        """Return refusal response matched to the language of the user's question."""
        if is_tamil_text(query):
            return "கேட்கப்பட்ட கேள்விக்கு வழங்கப்பட்ட பத்திகளில் போதுமான ஆதாரம் இல்லை."
        return "I don't have enough relevant information in the knowledge base to answer this question."
