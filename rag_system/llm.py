import time
import importlib.util
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from rag_system.config import LLM_MODEL_NAME, DEVICE
from rag_system.runtime import first_parameter_info, parameter_count


class GroundedLLM:
    """Grounded LLM Generator using Qwen/Qwen2.5-1.5B-Instruct in 4-bit / FP16."""

    def __init__(self, model_name: str = LLM_MODEL_NAME, device: str = DEVICE):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.precision = "not_loaded"

    def memory_report(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 3),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 3),
        }

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            if self.device == "cuda":
                bnb_available = importlib.util.find_spec("bitsandbytes") is not None
                if bnb_available:
                    try:
                        # Attempt 4-bit NF4 quantization using BitsAndBytes.
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            quantization_config=bnb_config,
                            device_map="auto",
                            trust_remote_code=True,
                        )
                        self.precision = "4-bit"
                    except Exception as e:
                        print(f"[Warning] 4-bit loading failed. Falling back to FP16: {e}")
                        self.model = None

                if self.model is None:
                    if not bnb_available:
                        print("[Warning] bitsandbytes is not installed. Loading Qwen in FP16 instead of 4-bit.")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    self.precision = "FP16"
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                self.precision = "CPU FP32"

            print(f"[INFO] LLM precision: {self.precision}")
            mem = self.memory_report()
            if mem:
                print(f"[INFO] CUDA memory after LLM load: allocated={mem['allocated_gb']}GB reserved={mem['reserved_gb']}GB total={mem['total_gb']}GB")

            # Warmup generation pass
            try:
                target_device = next(self.model.parameters()).device
                dummy_inputs = self.tokenizer("Hello", return_tensors="pt").to(target_device)
                with torch.no_grad():
                    _ = self.model.generate(**dummy_inputs, max_new_tokens=5, do_sample=False)
            except Exception:
                pass

    def load_model(self):
        self._load_model()
        return self.model

    def diagnostics(self) -> Dict[str, Any]:
        self._load_model()
        info = first_parameter_info(self.model)
        return {
            "model": self.model_name,
            "device": info["device"],
            "dtype": self.precision if self.precision == "4-bit" else info["dtype"],
            "parameter_count": parameter_count(self.model),
            "precision": self.precision,
        }

    def generate_answer(
        self, query: str, evidence: List[Dict[str, Any]], max_new_tokens: int = 150
    ) -> Tuple[str, float]:
        """Generate answer strictly grounded in retrieved evidence."""
        if not evidence:
            return "", 0.0

        self._load_model()
        t_start = time.perf_counter()

        context = "\n\n".join(
            [f"[Source {i+1}]\n{item['text']}" for i, item in enumerate(evidence)]
        )

        prompt = f"""You are a grounded multilingual search assistant.

Instructions:
1. Answer the question using ONLY the provided CONTEXT below.
2. Do NOT use external knowledge. Do NOT hallucinate facts not present in the context.
3. Keep the answer concise and direct.
4. You MUST answer in the EXACT SAME LANGUAGE as the question (e.g. Tamil if question is in Tamil).

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = output[0, inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        gen_ms = (time.perf_counter() - t_start) * 1000.0

        return answer, gen_ms
