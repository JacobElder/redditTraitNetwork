"""Load, merge, and hash run configuration.

Every network artifact records ``Config.hash8`` and the prompt version so a run
is reproducible and reruns hit the cache. The hash covers the fully-resolved
config *including the inlined trait vocabulary*, so editing ``traits.yaml``
correctly invalidates downstream caches.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class Config:
    """Resolved configuration. Access attributes, not the raw dict.

    ``raw`` holds the merged mapping; typed accessors below cover the fields the
    pipeline reads. Unlisted fields are still reachable via ``get()``.
    """

    raw: dict[str, Any]
    sources: tuple[str, ...] = ()
    _traits_inlined: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- generic access -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # -- typed shortcuts (only the hot paths) --------------------------
    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def prompt_version(self) -> str:
        return self.raw["prompts"]["version"]

    @property
    def cache_path(self) -> Path:
        return _resolve_path(self.raw["cache"]["path"])

    @property
    def cache_enabled(self) -> bool:
        return bool(self.raw["cache"].get("enabled", True))

    @property
    def replicates(self) -> int:
        return int(self.raw["estimate"]["replicates"])

    @property
    def trait_vocab_path(self) -> Path:
        return _resolve_path(self.raw["trait_vocab"])

    # -- hashing ------------------------------------------------------
    @property
    def canonical_json(self) -> str:
        """Deterministic JSON of the resolved config + inlined trait vocab.

        Excludes ``cache`` (path / enabled toggle must not change results) but
        keeps everything that could change a number.
        """
        payload = {
            k: v for k, v in self.raw.items() if k != "cache"
        }
        payload["_traits"] = self._traits_inlined
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @property
    def hash_full(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def hash8(self) -> str:
        return self.hash_full[:8]


def load_config(path: str | Path, *overrides: str | Path) -> Config:
    """Deep-merge ``overrides`` (later wins) onto the base YAML at ``path``.

    The trait vocab file named by ``trait_vocab`` is read and inlined into the
    hash input so vocab edits invalidate caches.
    """
    base_path = _resolve_path(str(path))
    merged: dict[str, Any] = yaml.safe_load(base_path.read_text()) or {}
    seen = [str(base_path)]
    for ov in overrides:
        ov_path = _resolve_path(str(ov))
        merged = _deep_merge(merged, yaml.safe_load(ov_path.read_text()) or {})
        seen.append(str(ov_path))

    traits_path = _resolve_path(merged["trait_vocab"])
    traits_inlined = yaml.safe_load(traits_path.read_text()) or {}

    return Config(raw=merged, sources=tuple(seen), _traits_inlined=traits_inlined)
