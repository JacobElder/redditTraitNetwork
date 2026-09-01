"""Versioned prompt templates.

Each version is a module (``p1``, ``p1a``, ...) exposing ``PROMPTS`` : a
``Prompts`` instance. Bump the version on ANY wording change — the version
string is part of the cache key and the config hash.
"""

from __future__ import annotations

import importlib

from .base import Prompts


def get_prompts(version: str) -> Prompts:
    mod = importlib.import_module(f"rtn.prompts.{version}")
    return mod.PROMPTS
