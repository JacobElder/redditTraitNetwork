import pandas as pd
import pytest

from rtn.ingest.chunking import chunk_items
from rtn.ingest.exclusion import ExcludedHistoryError, ExclusionFilter


def _items(subs, n_each=3):
    rows = []
    t = 1_600_000_000
    for s in subs:
        for k in range(n_each):
            t += 3600
            rows.append(
                dict(
                    item_id=f"t1_{s}{k}", account_hash="h", kind="comment",
                    subreddit=s, created_utc=t, body=f"a comment in {s} " * 20,
                    score=1, parent_id="t3_x", link_id=f"t3_{s}", permalink="/p",
                    fetched_utc=t, source="test",
                )
            )
    return pd.DataFrame(rows)


def test_exclusion_filter_flags_and_raises():
    f = ExclusionFilter("config/excluded_subreddits.yaml")
    assert f.is_excluded("depression")
    assert f.is_excluded("EatingDisorderHelp")  # prefix match
    assert not f.is_excluded("AskHistorians")

    mostly_excluded = _items(["depression", "anxiety", "AskReddit"])
    with pytest.raises(ExcludedHistoryError):
        f.filter_items(mostly_excluded, max_fraction=0.5)

    ok = _items(["AskReddit", "books", "cooking", "depression"])
    kept = f.filter_items(ok, max_fraction=0.5)
    assert "depression" not in set(kept["subreddit"])


def test_chunking_respects_caps_and_assigns_epochs():
    items = _items(["books", "cooking", "hiking"], n_each=20)
    chunks = chunk_items(
        items, {"strategy": "window", "window_size": 10, "max_chars": 2000, "min_chars": 100}, "h"
    )
    assert chunks
    assert all(c.n_chars <= 2000 for c in chunks)
    assert all(c.chunk_id.startswith("h:") for c in chunks)
    assert {c.epoch for c in chunks}  # epochs assigned
