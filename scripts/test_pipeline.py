import sys
import pprint
import logging
from pathlib import Path

# Resolve import paths
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main() -> None:
    print("=== MILESTONE 7: COMPLETE RAG PIPELINE FACADE ===\n")
    
    # 1. Instantiate the pipeline
    pipeline = RAGPipeline()
    
    # 2. Run query
    query = "சூரிய குடும்பத்தின் மிகப்பெரிய கோள் எது?"
    print(f"Calling pipeline.answer(query) with Tamil query...\n")
    
    try:
        response = pipeline.answer(query, language="tamil")
        print("\nStructured Response:")
        pprint.pprint(response)
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
