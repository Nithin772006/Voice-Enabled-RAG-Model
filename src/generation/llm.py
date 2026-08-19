import time
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config.settings import LLM_MODEL_NAME, DEVICE

class GroundedLLM:
    """Grounded LLM Generator using Qwen2.5-1.5B-Instruct optimized for low-latency FP16 GPU generation."""
    
    def __init__(self, model_name: str = LLM_MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is None:
            t0 = time.perf_counter()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            except TypeError:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            self.model.to(self.device)
            self.model.eval()
            load_sec = time.perf_counter() - t0
            print(f"[LLM] Loaded {self.model_name} (dtype={'fp16' if self.device == 'cuda' else 'fp32'}) on {self.device} in {load_sec:.2f}s")

    def format_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        # Limit context to top 2 chunks to minimize prompt token count
        top_context = context_chunks[:2]
        context_str = "\n".join([f"[{idx+1}] {c['text']}" for idx, c in enumerate(top_context)])
        
        system_instruction = (
            "Answer the question concisely in 1-2 sentences using ONLY the provided context.\n"
            "If insufficient info, reply: 'Information unavailable in provided context.'\n"
        )
        user_message = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"

        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_message}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
        return f"{system_instruction}\n{user_message}"

    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_new_tokens: int = 40
    ) -> Tuple[str, float]:
        if not context_chunks:
            return "மன்னிக்கவும், வழங்கப்பட்ட தரவுத்தளத்தில் போதுமான தகவல் இல்லை.", 0.0

        t0 = time.perf_counter()
        
        try:
            self.load_model()
            prompt = self.format_prompt(query, context_chunks)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_len:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            gen_ms = (time.perf_counter() - t0) * 1000.0
            return answer, gen_ms

        except Exception as e:
            gen_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[LLM GENERATION FALLBACK] ({e}) Returning context extraction fallback.")
            top_text = context_chunks[0]["text"]
            if len(top_text) > 200:
                top_text = top_text[:200] + "..."
            return top_text, gen_ms

    def diagnostics(self) -> Dict[str, Any]:
        self.load_model()
        param_cnt = sum(p.numel() for p in self.model.parameters())
        return {
            "model": self.model_name,
            "device": self.device,
            "parameter_count": param_cnt
        }
