import logging
import re
import sys
from typing import List
from guardrails.schemas import InputValidationResult, ValidationStatus

# Reconfigure stdout to support UTF-8 on Windows terminal/console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

logger = logging.getLogger("QueryGuard")

# Regex patterns for unsupported task domains (code generation, math solving, translations)
UNSUPPORTED_TASK_PATTERNS = [
    re.compile(r"\b(write|create|develop|code|program|make)\b.*\b(code|python|game|script|program|app|function|class|html|css|javascript|java|c\+\+|rust|go)\b", re.IGNORECASE),
    re.compile(r"\b(translate|translation)\b", re.IGNORECASE),
    re.compile(r"\b(solve|calculate|compute)\b.*\b(equation|math|formula|algebra|calculus|math problem)\b", re.IGNORECASE)
]

# Regex patterns for real-time data requests
REALTIME_PATTERNS = [
    re.compile(r"\b(weather|temperature|forecast|rain|snow)\b.*\b(tomorrow|today|current|now|in\s+\w+)\b", re.IGNORECASE),
    re.compile(r"\b(stock\s+price|market\s+price|share\s+price)\b", re.IGNORECASE),
    re.compile(r"\b(time|date)\b.*\b(right\s+now|current|today)\b", re.IGNORECASE),
    re.compile(r"\b(latest|current|today's|breaking)\s+(news|headline|headlines)\b", re.IGNORECASE)
]

def evaluate_query_scope(query: str) -> InputValidationResult:
    """Evaluates if the user query is within the functional scope of our QA RAG system.
    This prevents downstream LLM hallucinations or wasted vector lookups for unanswerable topics.
    """
    normalized_query = query.strip()
    
    # 1. Check for programming/coding/task execution requests using regexes
    for pattern in UNSUPPORTED_TASK_PATTERNS:
        if pattern.search(normalized_query):
            logger.warning(f"Flagged out-of-scope task query matching pattern: '{pattern.pattern}'")
            return InputValidationResult(
                is_valid=False,
                status=ValidationStatus.OUT_OF_DOMAIN,
                reason="This system only answers questions based on the reference dataset. Code writing, task automation, and general math problem-solving are not supported."
            )
            
    # 2. Check for real-time / current sensory requests using regexes
    for pattern in REALTIME_PATTERNS:
        if pattern.search(normalized_query):
            logger.warning(f"Flagged out-of-scope real-time query matching pattern: '{pattern.pattern}'")
            return InputValidationResult(
                is_valid=False,
                status=ValidationStatus.OUT_OF_DOMAIN,
                reason="This system does not have access to real-time information or external live APIs."
            )

            
    # 3. Passed validation
    return InputValidationResult(
        is_valid=True,
        status=ValidationStatus.SAFE,
        sanitized_query=query
    )

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test cases
    test_queries = [
        ("What is photosynthesis?", True),
        ("Write me a Python game.", False),
        ("What is the weather in New York tomorrow?", False),
        ("How does a diesel engine work?", True)
    ]
    
    print("--- RUNNING QUERY SCOPE TESTS ---")
    all_passed = True
    for idx, (test_query, expected_valid) in enumerate(test_queries):
        result = evaluate_query_scope(test_query)
        print(f"Test {idx + 1} | Query: {repr(test_query)}")
        print(f"       Result is_valid={result.is_valid} | Status={result.status} | Reason={result.reason}")
        if result.is_valid != expected_valid:
            print("       [FAIL] Mismatch between scope evaluation and expectation.")
            all_passed = False
        else:
            print("       [PASS]")
            
    if all_passed:
        print("\nAll query scope tests passed successfully!")
    else:
        print("\nSome query scope tests failed.")
