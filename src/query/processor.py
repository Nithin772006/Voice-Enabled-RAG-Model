import re
import time
from typing import Dict, Any, Tuple
from src.data.normalize import normalize_text

UNSAFE_KEYWORDS = {
    "hack", "exploit", "attack", "malware", "virus", "bomb", "suicide", "bypass", "illegal", "murder", "kill"
}

OUT_OF_DOMAIN_PATTERNS = [
    r"\b(weather|temperature|stock|price|crypto|bitcoin|currency exchange|sports score)\b",
    r"\b(play song|movie download|torrent|game cheat)\b"
]

class QueryProcessor:
    """Fast non-LLM Query Preprocessor for low latency language detection & classification."""
    
    def detect_language(self, text: str) -> str:
        """Detect language using Indic Unicode character range inspection."""
        if not text:
            return "unknown"
            
        tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        total_chars = len(text.strip())

        if total_chars > 0:
            if (tamil_chars / total_chars) > 0.15:
                return "ta"
            if (hindi_chars / total_chars) > 0.15:
                return "hi"
                
        return "en"

    def classify_query(self, text: str) -> str:
        """Classify query fast without LLM call."""
        text_lower = text.lower()
        
        # 1. Unsafe check
        words = set(re.findall(r'\w+', text_lower))
        if words.intersection(UNSAFE_KEYWORDS):
            return "UNSAFE"
            
        # 2. Ambiguous check (too short or meaningless)
        if len(text_lower.strip()) < 3:
            return "AMBIGUOUS"

        # 3. Out of domain keyword pattern check
        for pattern in OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, text_lower):
                return "OUT_OF_DOMAIN"

        return "IN_DOMAIN"

    def process_query(self, query: str) -> Tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        normalized = normalize_text(query)
        lang = self.detect_language(normalized)
        q_class = self.classify_query(normalized)
        proc_ms = (time.perf_counter() - t0) * 1000.0

        res = {
            "original_query": query,
            "normalized_query": normalized,
            "language": lang,
            "classification": q_class,
            "is_valid": q_class in ["IN_DOMAIN", "AMBIGUOUS"]
        }
        return res, proc_ms
