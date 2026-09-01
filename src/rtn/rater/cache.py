"""SQLite cache for model calls.

Key = sha256 of the canonical ``(account_hash, prompt_version, task, trait,
replicate, config_hash)`` tuple. A config or prompt change changes the key, so a
stale result is never served.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    key            TEXT PRIMARY KEY,
    account_hash   TEXT,
    prompt_version TEXT,
    task           TEXT,
    trait          TEXT,
    replicate      INTEGER,
    config_hash    TEXT,
    request        TEXT,
    response       TEXT,
    model          TEXT,
    usage          TEXT,
    created_utc    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_calls_account ON calls(account_hash);
CREATE INDEX IF NOT EXISTS idx_calls_task ON calls(task);
"""


def make_key(parts: dict[str, Any]) -> str:
    ordered = {
        k: parts.get(k)
        for k in (
            "account_hash",
            "prompt_version",
            "task",
            "trait",
            "replicate",
            "config_hash",
            "model",
        )
    }
    blob = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class CallCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT response, model, usage FROM calls WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "response": row[0],
            "model": row[1],
            "usage": json.loads(row[2]) if row[2] else {},
        }

    def put(
        self,
        key: str,
        *,
        parts: dict[str, Any],
        request: str,
        response: str,
        model: str,
        usage: dict[str, Any],
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO calls "
            "(key, account_hash, prompt_version, task, trait, replicate, "
            " config_hash, request, response, model, usage, created_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                key,
                parts.get("account_hash"),
                parts.get("prompt_version"),
                parts.get("task"),
                parts.get("trait"),
                parts.get("replicate"),
                parts.get("config_hash"),
                request,
                response,
                model,
                json.dumps(usage),
                int(time.time()),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
