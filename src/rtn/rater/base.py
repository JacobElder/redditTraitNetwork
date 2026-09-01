"""``Rater`` — the model-call interface used by every estimator.

The base class owns caching, ret/validation bookkeeping, and JSON parsing.
Subclasses implement ``_generate(prompt) -> (text, usage)`` only.

``call_parts`` identifies the cache cell: ``account_hash, prompt_version, task,
trait, replicate, config_hash``. ``trait`` is whatever the cell is keyed on
(a single trait for E1 ablation / E3 chunk id, a pair string for E2, ``None``
for the E1 baseline).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .cache import CallCache, make_key

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class RaterResponse:
    text: str
    data: Any  # parsed JSON (dict/list) or None
    cached: bool = False
    usage: dict[str, Any] = field(default_factory=dict)


def parse_json_obj(text: str) -> Any:
    """Best-effort extraction of a JSON object from a model response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"no JSON object in response: {text[:200]!r}")


class Rater(ABC):
    name: str = "base"

    def __init__(
        self,
        *,
        cache: CallCache | None = None,
        cache_enabled: bool = True,
        max_retries: int = 1,
    ):
        self.cache = cache
        self.cache_enabled = cache_enabled and cache is not None
        self.max_retries = max_retries
        self._active_parts: dict[str, Any] = {}

    # -- subclass hook ------------------------------------------------
    @abstractmethod
    def _generate(
        self, prompt: str, call_parts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Return (response_text, usage_dict). ``call_parts`` is passed through
        for backends that answer from structure rather than the prompt string
        (the synthetic-recovery mock); real backends ignore it.
        """

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    # -- public API -------------------------------------------------
    def complete(
        self, prompt: str, *, call_parts: dict[str, Any], expect_json: bool = True
    ) -> RaterResponse:
        call_parts = {**call_parts, "model": self.model_id}
        key = make_key(call_parts)
        if self.cache_enabled:
            hit = self.cache.get(key)
            if hit is not None:
                data = parse_json_obj(hit["response"]) if expect_json else None
                return RaterResponse(
                    text=hit["response"], data=data, cached=True, usage=hit["usage"]
                )

        self._active_parts = call_parts
        text, usage, data = self._generate_validated(prompt, expect_json)

        if self.cache_enabled:
            self.cache.put(
                key,
                parts=call_parts,
                request=prompt,
                response=text,
                model=self.model_id,
                usage=usage,
            )
        return RaterResponse(text=text, data=data, cached=False, usage=usage)

    def _generate_validated(
        self, prompt: str, expect_json: bool
    ) -> tuple[str, dict[str, Any], Any]:
        last_err: Exception | None = None
        cur_prompt = prompt
        for attempt in range(self.max_retries + 1):
            text, usage = self._generate(cur_prompt, self._active_parts)
            if not expect_json:
                return text, usage, None
            try:
                return text, usage, parse_json_obj(text)
            except (ValueError, json.JSONDecodeError) as err:
                last_err = err
                cur_prompt = (
                    prompt
                    + "\n\nYour previous reply was not valid JSON. "
                    "Return ONLY the JSON object, nothing else."
                )
        raise ValueError(f"model did not return valid JSON after retries: {last_err}")
