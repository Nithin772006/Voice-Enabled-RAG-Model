import sys
import logging
from pathlib import Path

# Add project root to sys.path to resolve local imports cleanly from python scripts/test_rag.py
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.retriever import QdrantRetriever
from llm.generator import RAGGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main() -> None:
    print("=== MILESTONE 6: CONNECT RETRIEVAL + PROMPT BUILDER + SARVAM ===\n")
    
    # 1. Initialize retriever and ensure test data is seeded
    retriever = QdrantRetriever()
    retriever.seed_test_data()
    
    # 2. Initialize generator
    try:
        generator = RAGGenerator()
    except Exception as e:
        print(f"Failed to initialize RAGGenerator: {e}")
        return
        
    # 3. Define the query
    query = "சூரிய குடும்பத்தின் மிகப்பெரிய கோள் எது?"  # "What is the largest planet in the solar system?"
    print(f"User Query: {repr(query)}")
    
    # 4. Step 1: Retrieve context chunks from Qdrant
    retrieved_chunks = retriever.retrieve(query, top_k=2)
    print("\n--- RETRIEVED CHUNKS FROM QDRANT ---")
    for chunk in retrieved_chunks:
        print(f"- [Rank {chunk['rank']}] (ID: {chunk['chunk_id']}, Score: {chunk['score']}): {repr(chunk['text'])}")
        
    print("\nSending context and query to Sarvam AI...")
    
    try:
        # 5. Step 2: Generate answer grounded in context (requesting answer in Tamil)
        answer = generator.generate(
            query=query,
            retrieved_chunks=retrieved_chunks,
            language="tamil"
        )
        print("\nGenerated Grounded Answer (Tamil):")
        print(answer)
    except Exception as e:
        print(f"\nExecution Failed: {e}")

if __name__ == "__main__":
    main()
