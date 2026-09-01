"""Ingest pipeline with a fake client — no network."""

import pandas as pd

from rtn.config import load_config
from rtn.ingest.clients import ITEMS_COLUMNS
from rtn.ingest.pipeline import ingest_username


class FakeClient:
    def __init__(self, df):
        self._df = df

    def fetch_history(self, username, **kw):
        return self._df.copy()


def _synthetic_history(n_comments=400, n_subs=8, excluded_frac=0.0):
    rows = []
    t = 1_500_000_000
    subs = [f"sub{i}" for i in range(n_subs)]
    n_excl = int(n_comments * excluded_frac)
    for k in range(n_comments):
        t += int(2.6 * 86400)  # ~2.6 days apart -> >2.8 years for 400 items
        sub = "depression" if k < n_excl else subs[k % n_subs]
        rows.append(
            dict(
                item_id=f"t1_c{k}", account_hash="", kind="comment", subreddit=sub,
                created_utc=t, body=f"a reasonably long comment number {k} " * 8,
                score=1, parent_id="t3_x", link_id=f"t3_{k // 3}", permalink="/p",
                fetched_utc=t, source="test",
            )
        )
    return pd.DataFrame(rows, columns=ITEMS_COLUMNS)


def _patch_dirs(pipeline, monkeypatch, tmp_path, acct_hash):
    monkeypatch.setattr(pipeline, "_accounts_dir", lambda: tmp_path / "data" / "accounts")

    def fake_init(self, p=None):
        self._map = {}
        self.path = tmp_path / "um.json"

    monkeypatch.setattr(pipeline.UserMap, "__init__", fake_init)
    monkeypatch.setattr(pipeline.UserMap, "add", lambda self, name, salt=None: acct_hash)


def test_ingest_eligible_account(tmp_path, monkeypatch):
    monkeypatch.setenv("RTN_HASH_SALT", "test-salt")
    from rtn.ingest import pipeline

    _patch_dirs(pipeline, monkeypatch, tmp_path, "deadbeefcafe0001")
    cfg = load_config("config/default.yaml")
    r = ingest_username("whoever", cfg, client=FakeClient(_synthetic_history()))
    assert r.eligible, r.reason
    assert r.n_comments == 400
    assert r.n_chunks > 0
    assert (tmp_path / "data" / "accounts" / r.account_hash / "chunks.parquet").exists()
    assert (tmp_path / "data" / "accounts" / "index.parquet").exists()


def test_ingest_rejects_majority_excluded(tmp_path, monkeypatch):
    monkeypatch.setenv("RTN_HASH_SALT", "test-salt")
    from rtn.ingest import pipeline

    _patch_dirs(pipeline, monkeypatch, tmp_path, "deadbeefcafe0002")
    cfg = load_config("config/default.yaml")
    r = ingest_username(
        "whoever", cfg, client=FakeClient(_synthetic_history(excluded_frac=0.7))
    )
    assert not r.eligible
    assert "excluded" in r.reason
