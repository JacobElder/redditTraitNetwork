"""Chunk an account's items into rater-sized units.

``thread`` strategy groups items sharing a ``link_id`` (a Reddit submission),
ordered by time, splitting when ``max_chars`` is exceeded. ``window`` strategy
takes fixed-size time-ordered windows of items. Every chunk gets an ``epoch``
index (equal-width time bins) for the temporal split-half test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Chunk:
    chunk_id: str
    account_hash: str
    strategy: str
    item_ids: list[str]
    t_start: int
    t_end: int
    n_chars: int
    text: str
    epoch: int
    meta: dict[str, Any] = field(default_factory=dict)


def _assemble(items: pd.DataFrame, max_chars: int) -> list[tuple[list[str], str, int, int]]:
    """Greedy pack time-ordered items into <= max_chars blobs."""
    out = []
    cur_ids: list[str] = []
    cur_parts: list[str] = []
    cur_len = 0
    t0 = t1 = None
    for _, row in items.iterrows():
        body = str(row["body"]).strip()
        piece = f"({row['subreddit']}) {body}"
        if cur_ids and cur_len + len(piece) > max_chars:
            out.append((cur_ids, "\n\n".join(cur_parts), t0, t1))
            cur_ids, cur_parts, cur_len = [], [], 0
            t0 = None
        cur_ids.append(row["item_id"])
        cur_parts.append(piece)
        cur_len += len(piece) + 2
        t0 = row["created_utc"] if t0 is None else t0
        t1 = row["created_utc"]
    if cur_ids:
        out.append((cur_ids, "\n\n".join(cur_parts), t0, t1))
    return out


def chunk_items(items: pd.DataFrame, cfg: dict, account_hash: str) -> list[Chunk]:
    strategy = cfg.get("strategy", "thread")
    max_chars = int(cfg.get("max_chars", 6000))
    min_chars = int(cfg.get("min_chars", 200))
    window_size = int(cfg.get("window_size", 15))
    target_chunks = cfg.get("target_chunks")

    items = items.sort_values("created_utc").reset_index(drop=True)
    if items.empty:
        return []

    # A prolific account can yield thousands of thread-chunks, which is unusable
    # for per-chunk LLM rating and for graphical VAR. When target_chunks is set,
    # coarsen to ~that many time-ordered windows (still capped by max_chars).
    if target_chunks:
        strategy = "window"
        window_size = max(window_size, -(-len(items) // int(target_chunks)))

    t_lo, t_hi = int(items["created_utc"].min()), int(items["created_utc"].max())
    n_epochs = max(2, min(6, len(items) // 40))
    edges = np.linspace(t_lo, t_hi + 1, n_epochs + 1)

    def epoch_of(ts: int) -> int:
        return int(np.clip(np.searchsorted(edges, ts, side="right") - 1, 0, n_epochs - 1))

    blocks: list[tuple[list[str], str, int, int]] = []
    if strategy == "thread":
        for _, grp in items.groupby("link_id", sort=False):
            blocks.extend(_assemble(grp.sort_values("created_utc"), max_chars))
    elif strategy == "window":
        for start in range(0, len(items), window_size):
            grp = items.iloc[start : start + window_size]
            blocks.extend(_assemble(grp, max_chars))
    else:
        raise ValueError(f"unknown chunk strategy {strategy!r}")

    chunks: list[Chunk] = []
    k = 0
    for ids, text, t0, t1 in blocks:
        if len(text) < min_chars:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{account_hash}:{k:05d}",
                account_hash=account_hash,
                strategy=strategy,
                item_ids=ids,
                t_start=int(t0),
                t_end=int(t1),
                n_chars=len(text),
                text=text,
                epoch=epoch_of(int(t0)),
            )
        )
        k += 1
    return chunks


def chunks_to_df(chunks: list[Chunk]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "account_hash": [c.account_hash for c in chunks],
            "strategy": [c.strategy for c in chunks],
            "item_ids": [c.item_ids for c in chunks],
            "t_start": [c.t_start for c in chunks],
            "t_end": [c.t_end for c in chunks],
            "n_chars": [c.n_chars for c in chunks],
            "text": [c.text for c in chunks],
            "epoch": [c.epoch for c in chunks],
        }
    )
