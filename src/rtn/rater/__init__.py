"""Rater backends. Use :func:`build_rater` to construct from a ``Config``."""

from __future__ import annotations

from ..traits import TraitVocab
from .base import Rater, RaterResponse, parse_json_obj
from .cache import CallCache, make_key
from .mock_backend import MockRater, Oracle

__all__ = [
    "CallCache",
    "MockRater",
    "Oracle",
    "Rater",
    "RaterResponse",
    "build_rater",
    "make_key",
    "parse_json_obj",
]


def build_rater(config, *, vocab: TraitVocab, oracle: Oracle | None = None) -> Rater:
    """Construct the rater named by ``config.model['backend']``.

    ``oracle`` is only meaningful for the ``mock`` backend (synthetic recovery).
    """
    cache = CallCache(config.cache_path) if config.cache_enabled else None
    common = dict(cache=cache, cache_enabled=config.cache_enabled)
    m = config.model
    backend = m["backend"]

    if backend == "mock":
        return MockRater(
            vocab=vocab, oracle=oracle, seed=m.get("seed", 7), **common
        )
    if backend == "anthropic":
        from .anthropic_backend import AnthropicRater

        return AnthropicRater(
            model=m["name"],
            max_tokens=m.get("max_tokens", 2048),
            temperature=m.get("temperature", 0.0),
            seed=m.get("seed"),
            **common,
        )
    if backend == "gemini":
        from .gemini_backend import GeminiRater

        return GeminiRater(
            model=m.get("name", "gemini-2.0-flash"),
            api_key_env=m.get("api_key_env", "GEMINI_API_KEY"),
            api_key=m.get("api_key"),
            temperature=m.get("temperature", 0.0),
            max_tokens=m.get("max_tokens", 2048),
            request_pause_s=m.get("request_pause_s", 0.0),
            **common,
        )
    if backend in ("openai_compatible", "ollama", "openai"):
        from .openai_compatible import OpenAICompatibleRater

        defaults = {
            "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY"),
            "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
        }
        base_default, key_default = defaults.get(backend, (None, "OPENAI_API_KEY"))
        return OpenAICompatibleRater(
            base_url=m.get("base_url", base_default),
            model=m["name"],
            api_key_env=m.get("api_key_env", key_default),
            api_key=m.get("api_key"),
            temperature=m.get("temperature", 0.0),
            max_tokens=m.get("max_tokens", 2048),
            request_pause_s=m.get("request_pause_s", 0.0),
            force_json=m.get("force_json", True),
            **common,
        )
    raise ValueError(f"unknown model backend {backend!r}")
