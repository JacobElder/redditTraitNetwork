"""Username hashing. Called exactly once per account, at fetch time.

No raw username may exist past this module. The raw->hash map is the single
gitignored file ``secrets/usermap.json``; :class:`UserMap` refuses to operate
unless ``secrets/`` is gitignored.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..config import REPO_ROOT

SALT_ENV = "RTN_HASH_SALT"


def _salt() -> str:
    salt = os.environ.get(SALT_ENV)
    if not salt:
        raise RuntimeError(
            f"set {SALT_ENV} (a fixed project secret) before hashing usernames"
        )
    return salt


def hash_username(name: str, salt: str | None = None) -> str:
    salt = salt or _salt()
    digest = hashlib.sha256(f"{salt}:{name.strip().lower()}".encode()).hexdigest()
    return digest[:16]


def _gitignore_covers_secrets() -> bool:
    gi = REPO_ROOT / ".gitignore"
    if not gi.exists():
        return False
    return any(
        line.strip().rstrip("/") == "secrets"
        for line in gi.read_text().splitlines()
    )


class UserMap:
    """Append-only raw-username -> hash store. Lives only under ``secrets/``."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else (REPO_ROOT / "secrets" / "usermap.json")
        if not _gitignore_covers_secrets():
            raise RuntimeError(
                "refusing to run: 'secrets/' is not in .gitignore"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._map: dict[str, str] = {}
        if self.path.exists():
            self._map = json.loads(self.path.read_text())

    def add(self, name: str, salt: str | None = None) -> str:
        h = hash_username(name, salt)
        key = name.strip().lower()
        if key not in self._map:
            self._map[key] = h
            self.path.write_text(json.dumps(self._map, indent=2, sort_keys=True))
        return h

    def get(self, name: str) -> str | None:
        return self._map.get(name.strip().lower())

    def __len__(self) -> int:
        return len(self._map)
