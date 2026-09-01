"""Rater for any OpenAI-compatible ``/chat/completions`` endpoint.

Covers the free / cheap options:

* **Ollama** (local, genuinely free) — ``base_url: http://localhost:11434/v1``,
  ``api_key: ollama``, ``name: llama3.1:70b`` (or a smaller model).
* **Groq** free tier — ``base_url: https://api.groq.com/openai/v1``,
  ``api_key`` from ``GROQ_API_KEY``.
* **OpenRouter** (has free models) — ``base_url: https://openrouter.ai/api/v1``.
* **Together, DeepInfra, vLLM, LM Studio, …** — same shape.

Uses raw ``requests`` so no extra SDK. Set ``model.backend: openai_compatible``
and ``model.base_url`` / ``model.api_key_env`` / ``model.name`` in the config.
"""

from __future__ import annotations

import time
from typing import Any

from .base import Rater
from .keys import resolve_key

_SYSTEM = (
    "You are a careful research assistant assisting with a personality-"
    "measurement study. Follow the response-format instruction exactly. "
    "Output only what is asked for — no preamble, no code fences."
)


class OpenAICompatibleRater(Rater):
    name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        request_pause_s: float = 0.0,
        force_json: bool = True,
        **kw: Any,
    ):
        super().__init__(**kw)
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install requests") from e
        self._requests = requests
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._key = resolve_key(api_key, api_key_env) or "not-needed"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._pause = request_pause_s
        self._force_json = force_json

    @property
    def model_id(self) -> str:
        return f"{self._model}@{self._url}"

    def _generate(self, prompt: str, call_parts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
        }
        if self._force_json and getattr(self, "_active_expect_json", True):
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(6):
            resp = self._requests.post(
                self._url,
                headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                json=payload,
                timeout=180,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("retry-after", 2**attempt))
                time.sleep(min(retry_after, 60))
                continue
            if resp.status_code == 400 and self._force_json:
                # some servers reject response_format — drop it and retry once
                payload.pop("response_format", None)
                self._force_json = False
                continue
            resp.raise_for_status()
            body = resp.json()
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {}) or {}
            if self._pause:
                time.sleep(self._pause)
            return text, {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            }
        raise RuntimeError(f"{self.name}: giving up after retries ({self._url})")
