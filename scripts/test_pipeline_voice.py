import sys
import pprint
import os
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
    print("=== MILESTONE 4: INTEGRATED RAG + VOICE PIPELINE TEST ===\n")
    
    # 1. Initialize the unified pipeline
    pipeline = RAGPipeline()
    
    # 2. Define our Tamil query (exists in the indexed 200 real chunks)
    query = "ஒரு நிறுவனம் என்பது என்ன?"
    print(f"User Query: {repr(query)}")
    print("Executing full voice RAG pipeline...\n")
    
    try:
        # 3. Execute pipeline with voice generation enabled
        response = pipeline.answer(query, language="tamil", generate_voice=True)
        
        print("\nPipeline Structured Response:")
        pprint.pprint(response)
        
        # 4. Verify that the output audio file was successfully generated
        audio_file = response.get("audio_path")
        if audio_file and os.path.exists(audio_file):
            print(f"\n[SUCCESS] Unified RAG Voice Pipeline execution completed!")
            print(f"Generated Audio File: {audio_file} (Size: {os.path.getsize(audio_file)} bytes)")
        else:
            print("\n[FAILED] Audio file was not generated or path is missing.")
            
    except Exception as e:
        print(f"\nPipeline failed during execution: {e}")

if __name__ == "__main__":
    main()
