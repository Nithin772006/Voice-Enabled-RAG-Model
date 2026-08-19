import re
from typing import List, Dict, Any, Tuple

class GroundingValidator:
    """Verifies that the answer is strictly supported by retrieved evidence context."""
    
    def evaluate(self, answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[bool, float, str]:
        """
        Returns: (is_grounded, support_ratio, grounding_case)
        - CASE_A_EXACT_MATCH: Answer directly matched in retrieved context -> grounded = True
        - CASE_A_GROUNDED: Factual answer supported by context -> grounded = True
        - CASE_C_REFUSAL: Answer is a refusal / info unavailable -> grounded = False
        - CASE_B_UNSUPPORTED_FACT: Fact unsupported by context -> grounded = False
        - CASE_C_NO_CONTEXT: No context retrieved -> grounded = False
        """
        if not context_chunks:
            return False, 0.0, "CASE_C_NO_CONTEXT"

        if not answer or not answer.strip():
            return False, 0.0, "CASE_B_EMPTY_ANSWER"

        ans_lower = answer.strip().lower()

        # 1. Refusal responses are NEVER grounded!
        refusal_phrases = [
            "மன்னிக்கவும்", "போதுமான தகவல் இல்லை", "information unavailable", "insufficient information"
        ]
        for phrase in refusal_phrases:
            if phrase in ans_lower:
                return False, 0.0, "CASE_C_REFUSAL"

        combined_context = " ".join([
            f"{c.get('text', '')} {c.get('answer', '')}" for c in context_chunks
        ]).lower()

        # 2. Check exact substring or direct containment match
        if ans_lower in combined_context or any(ans_lower in c.get('text', '').lower() for c in context_chunks):
            return True, 1.0, "CASE_A_EXACT_MATCH"

        # 3. Check sentence-level overlap in context
        for chunk in context_chunks:
            chunk_text = chunk.get("text", "").lower()
            if len(ans_lower) > 10 and ans_lower[:20] in chunk_text:
                return True, 1.0, "CASE_A_EXACT_MATCH"

        # 4. Token overlap calculation using Unicode-aware splitting
        # Splitting on non-alphanumeric unicode characters to properly handle Indic graphemes
        answer_tokens = [w for w in re.split(r'[^\w]+', ans_lower, flags=re.UNICODE) if len(w) >= 2]
        
        if not answer_tokens:
            return True, 1.0, "CASE_A_SHORT_ANSWER"

        context_tokens = set(re.split(r'[^\w]+', combined_context, flags=re.UNICODE))
        supported_count = sum(1 for w in answer_tokens if w in context_tokens)
        support_ratio = supported_count / float(len(answer_tokens))

        if support_ratio >= 0.40:
            return True, round(support_ratio, 4), "CASE_A_GROUNDED"
        else:
            return False, round(support_ratio, 4), "CASE_B_UNSUPPORTED_FACT"

    def get_grounding_failure_message(self) -> str:
        return "Sorry, I couldn't find enough information in the provided knowledge base to answer that."
