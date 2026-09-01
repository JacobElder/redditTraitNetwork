"""Anthropic Messages API backend.

``temperature`` is forced low (default 0.0) for rating stability; replicate
variance is whatever the API still produces and is reported as ``weight_sd``.
JSON is requested in the prompt; the base class handles one reformat retry.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import Rater

_SYSTEM = (
    "You are a careful research assistant assisting with a personality-"
    "measurement study. Follow the response-format instruction exactly. "
    "Output only what is asked for."
)


class AnthropicRater(Rater):
    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        seed: int | None = None,
        api_key: str | None = None,
        request_pause_s: float = 0.0,
        **kw: Any,
    ):
        super().__init__(**kw)
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic") from e
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seed = seed
        self._pause = request_pause_s

    @property
    def model_id(self) -> str:
        return self._model

    def _generate(
        self, prompt: str, call_parts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        import anthropic

        for attempt in range(5):
            try:
                msg = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    system=_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except (anthropic.RateLimitError, anthropic.APIStatusError):
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        text = "".join(
            block.text for block in msg.content if block.type == "text"
        )
        usage = {
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        if self._pause:
            time.sleep(self._pause)
        return text, usage
