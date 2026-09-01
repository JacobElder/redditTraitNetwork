"""API-key resolution: constructor arg > environment variable > gitignored file.

The file fallback (``secrets/<name>_key``) lets a key be pasted into an editor
rather than a shell command, so it never lands in a terminal transcript.
"""

from __future__ import annotations

import os

from ..config import REPO_ROOT

_KEY_FILES = {
    "GEMINI_API_KEY": REPO_ROOT / "secrets" / "gemini_key",
    "OPENROUTER_API_KEY": REPO_ROOT / "secrets" / "openrouter_key",
    "GROQ_API_KEY": REPO_ROOT / "secrets" / "groq_key",
    "ANTHROPIC_API_KEY": REPO_ROOT / "secrets" / "anthropic_key",
}


def key_file_for(env_var: str):
    return _KEY_FILES.get(env_var, REPO_ROOT / "secrets" / f"{env_var.lower()}")


def resolve_key(api_key: str | None, env_var: str) -> str | None:
    if api_key:
        return api_key
    val = os.environ.get(env_var)
    if val:
        return val.strip()
    f = key_file_for(env_var)
    if f and f.exists():
        return f.read_text().strip()
    return None
