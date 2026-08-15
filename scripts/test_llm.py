import sys
import logging
from pathlib import Path

# Add project root to sys.path to resolve local imports cleanly from python scripts/test_llm.py
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm.model import SarvamModel

# Configure basic logging to see standard warnings and info logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main() -> None:
    print("--- Initializing SarvamModel Wrapper ---")
    try:
        model = SarvamModel()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return
        
    prompt = "Explain Retrieval-Augmented Generation in simple terms for a beginner."
    print(f"\nPrompt: {repr(prompt)}")
    print("Waiting for response from Sarvam AI...")
    
    try:
        answer = model.generate(prompt)
        print("\nGenerated Answer:")
        print(answer)
    except Exception as e:
        print(f"\nExecution Failed: {e}")

if __name__ == "__main__":
    main()
