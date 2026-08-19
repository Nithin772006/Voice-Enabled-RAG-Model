import os
import time
import httpx
from typing import Dict, Any, Tuple
from src.config.settings import SARVAM_API_KEY

class SarvamSTTClient:
    """Sarvam AI Speech-to-Text Client for Indian languages (Tamil, Hindi, English, etc.)."""
    
    def __init__(self, api_key: str = SARVAM_API_KEY):
        self.api_key = api_key
        self.api_url = "https://api.sarvam.ai/speech-to-text"

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language_code: str = "ta-IN",
        model: str = "saarika:v2"
    ) -> Tuple[str, str, float]:
        """
        Send audio bytes to Sarvam STT API.
        Returns: (transcribed_text, detected_language, latency_ms)
        """
        t0 = time.perf_counter()

        if not self.api_key:
            # Fallback mock transcription for testing when API key is missing
            time.sleep(0.05) # simulate fast response
            mock_text = "இந்தியாவின் தலைநகரம் என்ன?"
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print("[SARVAM STT] API key not provided; returning simulated mock transcript.")
            return mock_text, language_code, latency_ms

        headers = {
            "api-subscription-key": self.api_key
        }

        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model": model,
            "language_code": language_code
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self.api_url, headers=headers, files=files, data=data)
                response.raise_for_status()
                result = response.json()

            transcript = result.get("transcript", "").strip()
            detected_lang = result.get("language_code", language_code)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[SARVAM STT] Transcribed successfully in {latency_ms:.2f}ms: '{transcript}'")
            return transcript, detected_lang, latency_ms

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[SARVAM STT ERROR] API call failed ({e}). Returning fallback.")
            return "", language_code, latency_ms
