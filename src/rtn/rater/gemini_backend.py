"""Google Gemini backend via the AI Studio REST API.

The AI Studio free tier (key from https://aistudio.google.com/apikey) is rate-
limited per minute and per day rather than billed, which is enough to run
Milestone 1.2 over a day or two. Check the exact model name available to your
key with ``GET /v1beta/models?key=...``; ``gemini-2.0-flash`` and
``gemini-2.5-flash`` are the usual free-tier options. Paid Flash is cheap if you
want it to finish in one sitting.

Config: ``model.backend: gemini``, ``model.name: gemini-2.0-flash``,
key from ``GEMINI_API_KEY`` (or ``model.api_key``). Raw ``requests``, no SDK.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .base import Rater

_SYSTEM = (
    "You are a careful research assistant assisting with a personality-"
    "measurement study. Follow the response-format instruction exactly. "
    "Output only what is asked for — no preamble, no code fences."
)


class GeminiRater(Rater):
    name = "gemini"

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash",
        api_key_env: str = "GEMINI_API_KEY",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        request_pause_s: float = 0.0,
        **kw: Any,
    ):
        super().__init__(**kw)
        try:
            import requests
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install requests") from e
        self._requests = requests
        self._model = model
        self._key = api_key or os.environ.get(api_key_env)
        if not self._key:
            raise RuntimeError(f"set ${api_key_env} (free key at aistudio.google.com/apikey)")
        self._base = "https://generativelanguage.googleapis.com/v1beta/models"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._pause = request_pause_s

    @property
    def model_id(self) -> str:
        return f"gemini:{self._model}"

    def _generate(self, prompt: str, call_parts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        url = f"{self._base}/{self._model}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": _SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self._temperature,
                "maxOutputTokens": self._max_tokens,
                "responseMimeType": "application/json",
            },
        }
        for attempt in range(6):
            resp = self._requests.post(
                url,
                params={"key": self._key},
                json=payload,
                timeout=180,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(min(2**attempt * 5, 90))
                continue
            resp.raise_for_status()
            body = resp.json()
            try:
                text = body["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                # blocked / empty candidate — surface as a JSON error the base
                # class will retry once, then fail loudly
                text = json.dumps({"_gemini_error": body.get("promptFeedback", body)})
            usage = body.get("usageMetadata", {})
            if self._pause:
                time.sleep(self._pause)
            return text, {
                "input_tokens": usage.get("promptTokenCount"),
                "output_tokens": usage.get("candidatesTokenCount"),
            }
        raise RuntimeError(f"gemini: giving up after retries ({self._model})")
