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


def _sampled_windows(
    items: pd.DataFrame, target: int, max_chars: int, min_chars: int, account_hash: str
) -> list[Chunk]:
    n = len(items)
    bounds = np.linspace(0, n, target + 1, dtype=int)
    n_epochs = max(2, min(6, target // 8 or 2))
    chunks: list[Chunk] = []
    for k in range(target):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        win = items.iloc[lo:hi]
        pieces_all = [
            f"({r.subreddit}) {str(r.body).strip()}"
            for r in win.itertuples(index=False)
        ]
        avg_len = max(1, sum(len(p) for p in pieces_all) // len(pieces_all) + 2)
        keep_n = min(len(win), max(1, max_chars // avg_len))
        sel = np.linspace(0, len(win) - 1, keep_n, dtype=int)
        sel = sorted({int(s) for s in sel})
        truncated = keep_n < len(win)
        parts, ids, used = [], [], 0
        for pos in sel:
            piece = pieces_all[pos]
            if used and used + len(piece) > max_chars:
                truncated = True
                break
            parts.append(piece)
            ids.append(win.iloc[pos]["item_id"])
            used += len(piece) + 2
        text = "\n\n".join(parts)
        if len(text) < min_chars:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{account_hash}:{len(chunks):05d}",
                account_hash=account_hash,
                strategy="sampled_window",
                item_ids=ids,
                t_start=int(win["created_utc"].iloc[0]),
                t_end=int(win["created_utc"].iloc[-1]),
                n_chars=len(text),
                text=text,
                epoch=int(k * n_epochs // target),
                meta={"window_items": int(hi - lo), "sampled": len(ids), "truncated": truncated},
            )
        )
    return chunks


def chunk_items(items: pd.DataFrame, cfg: dict, account_hash: str) -> list[Chunk]:
    strategy = cfg.get("strategy", "thread")
    max_chars = int(cfg.get("max_chars", 6000))
    min_chars = int(cfg.get("min_chars", 200))
    window_size = int(cfg.get("window_size", 15))
    target_chunks = cfg.get("target_chunks")

    items = items.sort_values("created_utc").reset_index(drop=True)
    if items.empty:
        return []

    # A prolific account can yield thousands of thread-chunks, unusable for
    # per-chunk LLM rating and graphical VAR. When target_chunks is set, cut the
    # history into exactly that many contiguous time windows and take an
    # evenly-spaced <= max_chars SAMPLE of each window (rather than splitting a
    # dense window into more chunks). Each chunk is then a snapshot of one
    # period — the right unit for E3's time series.
    if target_chunks and len(items) > int(target_chunks):
        return _sampled_windows(
            items, int(target_chunks), max_chars, min_chars, account_hash
        )

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
