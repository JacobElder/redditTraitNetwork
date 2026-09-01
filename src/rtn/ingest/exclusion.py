"""Subreddit exclusion filter, applied at ingest.

Inferring trait structure from disclosures in support communities is the primary
harm case. An account whose history is >= ``max_excluded_fraction`` excluded
hard-fails ingest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


class ExcludedHistoryError(RuntimeError):
    """Raised when too much of an account's history is in excluded subreddits."""


class ExclusionFilter:
    def __init__(self, path: str | Path):
        doc = yaml.safe_load(Path(path).read_text())
        self.exact = {s.strip().lower() for s in doc.get("exact", [])}
        self.prefixes = tuple(s.strip().lower() for s in doc.get("prefix", []))

    def is_excluded(self, subreddit: str) -> bool:
        s = subreddit.strip().lower()
        if s in self.exact:
            return True
        return any(s.startswith(p) for p in self.prefixes)

    def excluded_mask(self, subreddits: pd.Series) -> pd.Series:
        return subreddits.map(self.is_excluded)

    def excluded_fraction(self, items: pd.DataFrame) -> float:
        if len(items) == 0:
            return 0.0
        return float(self.excluded_mask(items["subreddit"]).mean())

    def filter_items(
        self, items: pd.DataFrame, max_fraction: float
    ) -> pd.DataFrame:
        """Drop excluded items; raise if the excluded fraction >= max_fraction."""
        frac = self.excluded_fraction(items)
        if frac >= max_fraction:
            raise ExcludedHistoryError(
                f"account history is {frac:.0%} excluded subreddits "
                f"(threshold {max_fraction:.0%}) — not eligible"
            )
        keep = ~self.excluded_mask(items["subreddit"])
        return items.loc[keep].reset_index(drop=True)
