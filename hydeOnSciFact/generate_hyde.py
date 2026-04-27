from __future__ import annotations

import os

from huggingface_hub import InferenceClient
from openai import OpenAI

from config import (
    HYDE_FALLBACK_MODELS,
    HF_TOKEN,
    HF_INFERENCE_PROVIDER,
    HYDE_PROVIDER,
    OPENAI_API_KEY,
    TOGETHER_API_KEY,
    HYDE_DO_SAMPLE,
    HYDE_MAX_NEW_TOKENS,
    HYDE_PROMPT_TEMPLATE,
    HYDE_TEMPERATURE,
    LLM_MODEL,
)


class HyDEGenerator:
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        hf_token: str | None = HF_TOKEN,
        provider: str = HYDE_PROVIDER,
        max_new_tokens: int = HYDE_MAX_NEW_TOKENS,
        temperature: float = HYDE_TEMPERATURE,
        do_sample: bool = HYDE_DO_SAMPLE,
    ) -> None:
        # Read env at runtime to support notebook-side key input after import time.
        if provider == "openai":
            resolved_token = hf_token or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            if not resolved_token:
                raise ValueError(
                    "OpenAI API key is missing. Set OPENAI_API_KEY before HyDE generation."
                )
        elif provider == "together":
            resolved_token = hf_token or TOGETHER_API_KEY or os.getenv("TOGETHER_API_KEY")
            if not resolved_token:
                raise ValueError(
                    "Together API key is missing. Set TOGETHER_API_KEY before HyDE generation."
                )
        else:
            resolved_token = hf_token or os.getenv("HF_TOKEN")
            if not resolved_token:
                raise ValueError(
                    "HF token is missing. Set environment variable HF_TOKEN "
                    "or run `hf auth login` before HyDE generation."
                )

        self._token = resolved_token
        self._provider = provider
        self._candidate_models = [model_name] + [
            m for m in HYDE_FALLBACK_MODELS if m != model_name
        ]
        self._active_model_idx = 0
        self.client = None
        self._openai_client = None
        if self._provider == "openai":
            self._openai_client = OpenAI(api_key=self._token)
        else:
            self.client = self._build_client(self._candidate_models[self._active_model_idx])
        self.model_name = self._candidate_models[self._active_model_idx]
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    def _build_client(self, model_name: str) -> InferenceClient:
        return InferenceClient(
            model=model_name,
            token=self._token,
            provider=self._provider,
            timeout=120,
        )

    def _is_model_not_found_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return ("404" in msg and "not found" in msg) or "router.huggingface.co" in msg

    def _try_generate_with_client(self, prompt: str) -> str:
        if self._provider == "openai":
            resp = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("OpenAI returned empty HyDE response.")
            return text

        # Preferred path: chat-completions API.
        try:
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception:
            pass

        generated = self.client.text_generation(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            return_full_text=False,
        )
        return str(generated).strip()

    def generate(self, query: str) -> str:
        prompt = HYDE_PROMPT_TEMPLATE.format(query=query)

        last_error: Exception | None = None
        start_idx = self._active_model_idx

        for offset in range(len(self._candidate_models)):
            idx = (start_idx + offset) % len(self._candidate_models)
            model_name = self._candidate_models[idx]

            # Switch client only when needed.
            if self._provider != "openai" and idx != self._active_model_idx:
                self.client = self._build_client(model_name)
                self._active_model_idx = idx
                self.model_name = model_name

            try:
                return self._try_generate_with_client(prompt)
            except Exception as e:
                last_error = e
                # Only auto-switch on likely model availability/routing errors.
                if not self._is_model_not_found_error(e):
                    break
                continue

        raise RuntimeError(
            f"HyDE generation failed for model candidates: {self._candidate_models}. "
            f"Last error: {last_error}"
        )
