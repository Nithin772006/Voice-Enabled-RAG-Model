import os
import logging
from typing import Optional, Union, List, Dict
from dotenv import load_dotenv
from sarvamai import SarvamAI

logger = logging.getLogger("SarvamModel")

class SarvamModel:
    """Wrapper class for the Sarvam AI completions API.
    Insulates the application from API-specific payloads and manages credentials securely.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "sarvam-105b") -> None:
        # Load dotenv to read .env file if it hasn't been loaded in the entrypoint
        load_dotenv()
        
        # 1. Fetch API key from arguments or environment
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        
        # 2. Validation
        if not self.api_key:
            logger.error("SARVAM_API_KEY environment variable is missing or empty.")
            raise ValueError(
                "SARVAM_API_KEY was not found. Please ensure it is set in your environment "
                "or provided in your .env file."
            )
            
        # 3. Model configurations
        self.model_name = model_name
        
        # 4. Initialize client
        logger.info(f"Initializing SarvamAI client with model: {self.model_name}")
        self.client = SarvamAI(api_subscription_key=self.api_key, timeout=30.0)

    def generate(self, prompt: Union[str, List[Dict[str, str]]], temperature: float = 0.0) -> str:
        """Sends a prompt (either a raw string or a list of message dicts) to Sarvam AI and returns the text response."""
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
            
        try:
            # If prompt is a string, construct single user message list.
            # If it's a pre-built list of messages (System + User), pass it directly.
            if isinstance(prompt, str):
                if not prompt.strip():
                    raise ValueError("Prompt cannot be empty.")
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt

            logger.info(f"Sending generation request to {self.model_name}...")
            response = self.client.chat.completions(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            
            # Extract and return the answer
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error during Sarvam AI completion: {e}")
            raise RuntimeError(f"Sarvam AI generation failed: {e}") from e

