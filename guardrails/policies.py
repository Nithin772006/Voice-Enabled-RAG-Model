import re

# Central safety and threshold configurations for the RAG pipeline

# Max character limit for user queries
# Production rationale: Long queries bloat embedding latency and are often vector database denial-of-service vectors.
MAX_QUERY_LENGTH = 300

# Minimum similarity score required to accept a context chunk from Qdrant
# Cosine distance ranges from -1 to 1. 0.60 ensures high semantic alignment.
MIN_SIMILARITY_SCORE = 0.60

# Allowed query and answer languages
ALLOWED_LANGUAGES = ["english", "tamil", "hindi", "telugu"]

# Regex patterns for direct and indirect prompt injection attempts
BLOCKED_PATTERNS = [
    # 1. System Prompt Leak / Instruction Override Attempts
    re.compile(r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?system\s+(?:prompt|instruction)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a\s+different\s+assistant", re.IGNORECASE),
    re.compile(r"forget\s+(?:everything\s+)?you\s+know", re.IGNORECASE),
    
    # 2. Malicious Code/SQL Injection Signatures
    re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"\bSELECT\b.*\bFROM\b", re.IGNORECASE),
    re.compile(r"\bDROP\b\s+\bTABLE\b", re.IGNORECASE),
    re.compile(r"\bUNION\b\s+\bSELECT\b", re.IGNORECASE),
    
    # 3. Hidden File Access/System Commands
    re.compile(r"\bcat\s+/etc/passwd\b", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
]
