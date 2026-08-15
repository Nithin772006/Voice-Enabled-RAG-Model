from pydantic import BaseModel, Field
from typing import Optional

class LLMConfig(BaseModel):
    # Model Identifier: Can be an API-based model (e.g. gpt-4o, gemini-1.5-flash) 
    # or a local model (e.g. qwen2.5-7b-instruct, llama3)
    model_name: str = Field(default="gemini-1.5-flash", description="The identifier of the LLM to run")
    
    # Provider: 'openai', 'gemini', 'anthropic', or 'openai_compatible' (Ollama / OpenRouter)
    provider: str = Field(default="gemini", description="LLM provider name")
    
    # Temperature: 0.0 is critical for RAG. It ensures the model is deterministic, 
    # strictly selecting the most likely tokens and avoiding creative additions (hallucinations).
    temperature: float = Field(default=0.0, description="Temperature for generation (0.0 for factual constraint)")
    
    # Maximum output length
    max_output_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    
    # Custom endpoints for local execution or intermediate gateways (e.g. Ollama, OpenRouter)
    api_base_url: Optional[str] = Field(default=None, description="Custom base URL for OpenAI-compatible APIs")
    api_key_env_var: Optional[str] = Field(default="GEMINI_API_KEY", description="Environment variable holding the API key")
