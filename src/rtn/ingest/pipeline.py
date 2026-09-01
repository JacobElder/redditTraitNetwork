"""Turn a username into a cached, hashed, filtered, chunked corpus.

The one place raw usernames are handled. Everything downstream keys on
``account_hash``. Writes:

    data/accounts/{account_hash}/items.parquet
    data/accounts/{account_hash}/chunks.parquet
    data/accounts/index.parquet          (append; no raw usernames)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ..config import REPO_ROOT, Config
from .chunking import chunk_items, chunks_to_df
from .clients import ArcticShiftClient, PrawClient
from .exclusion import ExcludedHistoryError, ExclusionFilter
from .hashing import UserMap


@dataclass
class IngestResult:
    account_hash: str
    n_items: int
    n_comments: int
    n_posts: int
    n_subreddits: int
    span_days: float
    excluded_fraction: float
    n_chunks: int
    eligible: bool
    reason: str


def _accounts_dir() -> Path:
    return REPO_ROOT / "data" / "accounts"


def _check_inclusion(items: pd.DataFrame, cfg: Config) -> tuple[bool, str]:
    ing = cfg.raw["ingest"]
    n_comments = int((items["kind"] == "comment").sum())
    n_subs = items["subreddit"].nunique()
    span_days = (items["created_utc"].max() - items["created_utc"].min()) / 86400
    if n_comments < ing["min_comments"]:
        return False, f"only {n_comments} comments (< {ing['min_comments']})"
    if n_subs < ing["min_subreddits"]:
        return False, f"only {n_subs} subreddits (< {ing['min_subreddits']})"
    if span_days < ing["min_years_span"] * 365:
        return False, f"span {span_days:.0f}d (< {ing['min_years_span']}y)"
    return True, "eligible"


def ingest_username(
    username: str,
    cfg: Config,
    *,
    client: ArcticShiftClient | None = None,
    praw_client: PrawClient | None = None,
    gap_fill: bool = False,
    overwrite: bool = False,
) -> IngestResult:
    ing = cfg.raw["ingest"]
    usermap = UserMap()
    account_hash = usermap.add(username)
    out_dir = _accounts_dir() / account_hash
    items_path = out_dir / "items.parquet"

    if items_path.exists() and not overwrite:
        items = pd.read_parquet(items_path)
    else:
        client = client or ArcticShiftClient()
        max_items = ing.get("max_items")
        items = client.fetch_history(
            username, max_items=int(max_items) if max_items else None
        )
        if gap_fill and praw_client is not None and not items.empty:
            recent = praw_client.fetch_recent(username)
            items = (
                pd.concat([items, recent])
                .drop_duplicates("item_id")
                .sort_values("created_utc")
                .reset_index(drop=True)
            )
        items["account_hash"] = account_hash

    if items.empty:
        return IngestResult(account_hash, 0, 0, 0, 0, 0.0, 0.0, 0, False, "no history returned")

    excl_path = Path(cfg.get("ingest.exclusion_list"))
    if not excl_path.is_absolute():
        excl_path = REPO_ROOT / excl_path
    exclusion = ExclusionFilter(excl_path)
    excluded_fraction = exclusion.excluded_fraction(items)
    try:
        items = exclusion.filter_items(items, ing["max_excluded_fraction"])
    except ExcludedHistoryError as e:
        return IngestResult(
            account_hash, len(items), 0, 0, 0, 0.0, excluded_fraction, 0, False, str(e)
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    items.to_parquet(items_path, index=False)

    eligible, reason = _check_inclusion(items, cfg)

    chunks = chunk_items(items, ing["chunk"], account_hash)
    chunks_to_df(chunks).to_parquet(out_dir / "chunks.parquet", index=False)

    n_comments = int((items["kind"] == "comment").sum())
    result = IngestResult(
        account_hash=account_hash,
        n_items=len(items),
        n_comments=n_comments,
        n_posts=int((items["kind"] == "post").sum()),
        n_subreddits=int(items["subreddit"].nunique()),
        span_days=float((items["created_utc"].max() - items["created_utc"].min()) / 86400),
        excluded_fraction=round(excluded_fraction, 4),
        n_chunks=len(chunks),
        eligible=eligible,
        reason=reason,
    )
    _append_index(result)
    return result


def _append_index(result: IngestResult) -> None:
    idx_path = _accounts_dir() / "index.parquet"
    row = {**asdict(result), "ingested_utc": int(time.time())}
    df = pd.DataFrame([row])
    if idx_path.exists():
        prev = pd.read_parquet(idx_path)
        prev = prev[prev["account_hash"] != result.account_hash]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_parquet(idx_path, index=False)
