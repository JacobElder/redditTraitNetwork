"""Reddit history clients. STUBS — real implementations are a Milestone 1.2 task.

Milestone 1.1 (synthetic recovery) does not touch these; the synthetic path
generates chunks directly. Keep the return schema equal to
``data/accounts/{hash}/items.parquet`` in docs/PLAN.md §5.

Arctic Shift:
    Monthly Parquet dumps on Hugging Face. Query with DuckDB + httpfs without
    downloading the whole month, e.g.::

        duckdb> SELECT * FROM read_parquet('hf://datasets/.../RC_2024-01.parquet')
                WHERE author = ?  -- or scan for the sample frame

PRAW:
    Official API, ~1000-item listing cap, no date-range search. Freshness only.
"""

from __future__ import annotations

import pandas as pd

ITEMS_COLUMNS = [
    "item_id",
    "account_hash",
    "kind",
    "subreddit",
    "created_utc",
    "body",
    "score",
    "parent_id",
    "link_id",
    "permalink",
    "fetched_utc",
    "source",
]


def empty_items_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ITEMS_COLUMNS})


class ArcticShiftClient:
    def __init__(self, dataset_glob: str | None = None):
        self.dataset_glob = dataset_glob

    def sample_frame(self, criteria: dict) -> list[str]:  # pragma: no cover - stub
        raise NotImplementedError("Milestone 1.2: implement DuckDB sample-frame query")

    def fetch_history(self, username: str) -> pd.DataFrame:  # pragma: no cover - stub
        raise NotImplementedError("Milestone 1.2: implement Arctic Shift fetch")


class PrawClient:
    def __init__(self, **praw_kwargs: object):
        self.praw_kwargs = praw_kwargs

    def fetch_recent(self, username: str) -> pd.DataFrame:  # pragma: no cover - stub
        raise NotImplementedError("Milestone 1.2: implement PRAW gap-fill fetch")
