import sys
import logging
from pathlib import Path

# Resolve import paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm.pipeline import RAGPipeline
from llm.generator import RAGGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def test_same_language(pipeline: RAGPipeline) -> None:
    print("\n--- TEST: SAME-LANGUAGE RETRIEVAL & GENERATION ---")
    
    # We test 4 languages (English, Tamil, Hindi, Telugu)
    queries = [
        {"query": "What is machine learning?", "lang_code": "english"},
        {"query": "இயந்திர கற்றल என்றால் என்ன?", "lang_code": "tamil"},
        {"query": "मशीन लर्निंग क्या है?", "lang_code": "hindi"},
        {"query": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?", "lang_code": "telugu"}
    ]
    
    for item in queries:
        q = item["query"]
        lang = item["lang_code"]
        print(f"\nQuery ({lang}): {repr(q)}")
        try:
            # We seed data first in pipeline's retriever
            pipeline.retriever.seed_test_data()
            
            response = pipeline.answer(q, language=lang, top_k=1)
            print(f"Answer ({lang}):")
            print(response["answer"])
        except Exception as e:
            print(f"Failed same-language test for {lang}: {e}")

def test_cross_language(generator: RAGGenerator) -> None:
    print("\n--- TEST: CROSS-LANGUAGE RETRIEVAL & GENERATION ---")
    print("Scenario: Tamil query ➔ English evidence chunk ➔ Tamil answer")
    
    tamil_query = "இயந்திர கற்றல் என்றால் என்ன?"
    english_evidence = [
        {
            "chunk_id": "chunk_ml_en",
            "language": "english",
            "text": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."
        }
    ]
    
    print(f"\nTamil Query: {repr(tamil_query)}")
    print(f"English Source: {repr(english_evidence[0]['text'])}")
    print("Waiting for response from Sarvam AI...")
    
    try:
        # We pass English evidence to the generator but request the answer in Tamil
        answer = generator.generate(
            query=tamil_query,
            retrieved_chunks=english_evidence,
            language="tamil"
        )
        print("\nGenerated Answer (Should be Tamil translation):")
        print(answer)
    except Exception as e:
        print(f"Failed cross-language test: {e}")

def main() -> None:
    print("=== MILESTONE 8: MULTILINGUAL & CROSS-LINGUAL RAG TESTS ===\n")
    
    pipeline = RAGPipeline()
    generator = RAGGenerator(model=pipeline.generator.model)
    
    # 1. Test same-language flows
    test_same_language(pipeline)
    
    # 2. Test cross-language flows (English context to Tamil answer)
    test_cross_language(generator)

if __name__ == "__main__":
    main()
