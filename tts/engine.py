import os
import logging
import base64
import sys
from typing import Optional
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Reconfigure stdout to support UTF-8 on Windows terminal/console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

logger = logging.getLogger("TTSEngine")

# BCP-47 language codes map supported by Sarvam Bulbul:v3
LANGUAGE_MAP = {
    "english": "en-IN",
    "tamil": "ta-IN",
    "hindi": "hi-IN",
    "telugu": "te-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "marathi": "mr-IN",
    "punjabi": "pa-IN",
    "odia": "or-IN"
}

class TTSEngine:
    """Wrapper class for Sarvam AI's Text-to-Speech service.
    Translates language inputs and manages credentials for speech synthesis.
    """
    def __init__(self, api_key: Optional[str] = None) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        
        if not self.api_key:
            logger.error("SARVAM_API_KEY environment variable is missing.")
            raise ValueError("SARVAM_API_KEY was not found in environment or .env file.")
            
        logger.info("Initializing SarvamAI TTS client...")
        self.client = SarvamAI(api_subscription_key=self.api_key, timeout=30.0)

    def synthesize(self, text: str, language: str, output_path: str) -> str:
        """Synthesizes text into spoken audio and writes it as a WAV file to disk.
        Returns the absolute string path of the generated audio file.
        """
        if not text or not text.strip():
            raise ValueError("Text content to synthesize cannot be empty.")
            
        # 1. Translate language query string to BCP-47 code
        lang_key = language.lower().strip()
        lang_code = LANGUAGE_MAP.get(lang_key)
        
        if not lang_code:
            logger.warning(f"Language '{language}' not directly mapped. Falling back to English (en-IN).")
            lang_code = "en-IN"

        logger.info(f"Sending TTS request ({lang_code}): {repr(text[:40])}...")
        
        try:
            # 2. Call Sarvam TTS API
            # Uses default model 'bulbul:v3' and speaker 'shubh' automatically
            response = self.client.text_to_speech.convert(
                text=text,
                language_code=lang_code
            )
            
            # 3. Parse base64 and save to file
            if not getattr(response, "audios", None) or not response.audios:
                raise RuntimeError("Sarvam TTS API returned response without audio data.")
                
            audio_base64 = response.audios[0]
            audio_bytes = base64.b64decode(audio_base64)
            
            # Ensure target directory exists
            output_file = os.path.abspath(output_path)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, "wb") as f:
                f.write(audio_bytes)
                
            logger.info(f"Speech successfully synthesized and saved to {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error during TTS synthesis: {e}")
            raise RuntimeError(f"Speech synthesis failed: {e}") from e
