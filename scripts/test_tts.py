import sys
import logging
from pathlib import Path

# Resolve local imports cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tts.engine import TTSEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main() -> None:
    print("=== MILESTONE 2: TEST TTS ENGINE WRAPPER ===\n")
    
    # 1. Instantiate TTS Engine
    try:
        engine = TTSEngine()
    except Exception as e:
        print(f"Failed to initialize TTSEngine: {e}")
        return

    # 2. Test English audio generation
    eng_text = "Welcome to the HHGOA-2026 Multilingual RAG Voice Generator."
    eng_path = "data/test_english.wav"
    print(f"\nGenerating English Audio:")
    print(f"Text: {repr(eng_text)}")
    print(f"Saving to: {eng_path}...")
    try:
        saved_eng = engine.synthesize(eng_text, language="english", output_path=eng_path)
        print(f"[SUCCESS] Saved English audio to: {saved_eng}")
    except Exception as e:
        print(f"[FAILED] English synthesis: {e}")

    # 3. Test Tamil audio generation
    tam_text = "வணக்கம்! சூரிய குடும்பத்தின் மிகப்பெரிய கோள் வியாழன் ஆகும்."
    tam_path = "data/test_tamil.wav"
    print(f"\nGenerating Tamil Audio:")
    print(f"Text: {repr(tam_text)}")
    print(f"Saving to: {tam_path}...")
    try:
        saved_tam = engine.synthesize(tam_text, language="tamil", output_path=tam_path)
        print(f"[SUCCESS] Saved Tamil audio to: {saved_tam}")
    except Exception as e:
        print(f"[FAILED] Tamil synthesis: {e}")

if __name__ == "__main__":
    main()
