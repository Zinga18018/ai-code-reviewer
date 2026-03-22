import time
import logging
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .config import ReviewConfig, SYSTEM_PROMPT, FOCUS_PROMPTS

logger = logging.getLogger(__name__)


class CodeReviewer:
    """loads TinyLlama and runs code reviews with optional token streaming.

    keeps the model on GPU if available, falls back to CPU otherwise.
    supports four review modes: general, security, performance, style.
    """

    def __init__(self, config: ReviewConfig | None = None):
        self.config = config or ReviewConfig()
        self.model = None
        self.tokenizer = None
        self.device = None

    def load(self):
        """pull model weights and move to the best available device."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("loading %s on %s", self.config.model_id, self.device)

        dtype = torch.float16 if (self.device == "cuda" and self.config.use_fp16) else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("model ready on %s", self.device)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ---- prompt construction ----

    def _build_prompt(self, code: str, language: str, focus: str) -> str:
        focus_line = FOCUS_PROMPTS.get(focus, FOCUS_PROMPTS["general"])
        return (
            f"<|system|>\n{SYSTEM_PROMPT}\n{focus_line}</s>\n"
            f"<|user|>\nReview this {language} code:\n\n```{language}\n{code}\n```</s>\n"
            f"<|assistant|>\n"
        )

    def _tokenize(self, prompt: str) -> dict:
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=self.config.max_input_tokens,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _gen_kwargs(self, inputs: dict, max_tokens: int, **extra) -> dict:
        return {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "do_sample": True,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            **extra,
        }

    # ---- inference ----

    def review(self, code: str, language: str = "python",
               focus: str = "general", max_tokens: int = 512) -> dict:
        """run a full review and return the text + timing info."""
        prompt = self._build_prompt(code, language, focus)
        inputs = self._tokenize(prompt)

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**self._gen_kwargs(inputs, max_tokens))
        elapsed_ms = (time.perf_counter() - start) * 1000

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        review_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "review": review_text,
            "language": language,
            "focus": focus,
            "inference_ms": round(elapsed_ms, 1),
            "model": self.config.model_id,
            "device": self.device,
        }

    def stream(self, code: str, language: str = "python",
               focus: str = "general", max_tokens: int = 512):
        """yield review tokens one at a time for SSE streaming."""
        prompt = self._build_prompt(code, language, focus)
        inputs = self._tokenize(prompt)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        kwargs = self._gen_kwargs(inputs, max_tokens, streamer=streamer)

        thread = Thread(target=lambda: self.model.generate(**kwargs))
        thread.start()
        for chunk in streamer:
            yield chunk

    # ---- health check ----

    def health(self) -> dict:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        return {
            "status": "healthy" if self.is_loaded else "loading",
            "model": self.config.model_id,
            "device": str(self.device),
            "gpu": gpu,
        }
