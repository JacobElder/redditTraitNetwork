"""Reddit history clients.

``ArcticShiftClient`` — the primary route. Uses the Arctic Shift HTTP API
(``https://arctic-shift.photon-reddit.com/api``, unauthenticated, no key), which
serves the Pushshift-successor archive. Per-author history is fetched by
time-paginating ``/comments/search`` and ``/posts/search`` (``sort=asc``,
advancing ``after`` past the last item each page). This avoids scanning the
261 GB monthly Parquet dumps for one user.

``ArcticShiftDumpClient`` — a thin DuckDB-over-Hugging-Face helper for building a
*sample frame* (a big scan you do once), kept separate because it is slow and
optional.

``PrawClient`` — official Reddit API, for freshness / gap-filling only
(~1000-item listing cap, no date-range search). Needs app credentials.

All three return a DataFrame matching ``data/accounts/{hash}/items.parquet``
(docs/PLAN.md §5). ``account_hash`` is left blank here; the ingest script fills
it so raw usernames never leave that boundary.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

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

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"


def empty_items_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ITEMS_COLUMNS})


def _rows_to_frame(rows: list[dict[str, Any]], source: str) -> pd.DataFrame:
    if not rows:
        return empty_items_frame()
    df = pd.DataFrame(rows, columns=ITEMS_COLUMNS)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce").astype("Int64")
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype("int32")
    df = df.dropna(subset=["created_utc"]).reset_index(drop=True)
    df["created_utc"] = df["created_utc"].astype("int64")
    return df


class ArcticShiftClient:
    def __init__(
        self,
        base_url: str = ARCTIC_SHIFT_BASE,
        *,
        pause_s: float = 0.6,
        page_limit: int = 100,
        max_retries: int = 6,
        session: Any | None = None,
    ):
        import requests

        self.base_url = base_url.rstrip("/")
        self.pause_s = pause_s
        self.page_limit = page_limit
        self.max_retries = max_retries
        self._s = session or requests.Session()
        self._s.headers.setdefault("User-Agent", "reddit-trait-network/0.1 (research)")

    class BadPage(RuntimeError):
        """A 4xx (not 429) for a specific query — usually a query-timeout on a
        dense window. Caller skips the window forward rather than aborting."""

    # -- low level -------------------------------------------------
    def _get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(self.max_retries):
            resp = self._s.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                wait = float(resp.headers.get("X-RateLimit-Reset", 2**attempt))
                time.sleep(min(wait, 60))
                continue
            if resp.status_code >= 500:
                time.sleep(2**attempt)
                continue
            if resp.status_code in (400, 408, 422):
                raise self.BadPage(f"{resp.status_code} for {resp.url}")
            resp.raise_for_status()
            body = resp.json()
            data = body.get("data", body) if isinstance(body, dict) else body
            return data or []
        raise RuntimeError(f"Arctic Shift: giving up on {url} after {self.max_retries} tries")

    def _paginate(self, path: str, base_params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        after = int(base_params.pop("after", 0) or 0)
        seen: set[str] = set()
        skips = 0
        while True:
            params = {**base_params, "sort": "asc", "limit": self.page_limit}
            if after > 0:
                params["after"] = after
            try:
                page = self._get(path, params)
            except self.BadPage:
                # dense window / server hiccup — jump forward a day and retry,
                # up to a bound, so one bad slice doesn't lose the whole account
                skips += 1
                if skips > 400 or after == 0:
                    return
                after += 86400
                if self.pause_s:
                    time.sleep(self.pause_s)
                continue
            fresh = [d for d in page if d.get("id") and d["id"] not in seen]
            for d in fresh:
                seen.add(d["id"])
                yield d
            if not page:
                return
            last = max(int(d["created_utc"]) for d in page if d.get("created_utc"))
            if len(page) < self.page_limit or last <= after:
                return
            after = last + 1
            if self.pause_s:
                time.sleep(self.pause_s)

    # -- normalisation ------------------------------------------
    @staticmethod
    def _norm_comment(d: dict[str, Any]) -> dict[str, Any]:
        cid = d.get("id", "")
        link = d.get("link_id", "")
        link_bare = link.split("_", 1)[-1] if link else ""
        return {
            "item_id": f"t1_{cid}",
            "account_hash": "",
            "kind": "comment",
            "subreddit": str(d.get("subreddit", "")).lower(),
            "created_utc": d.get("created_utc"),
            "body": d.get("body", ""),
            "score": d.get("score", 0),
            "parent_id": d.get("parent_id", ""),
            "link_id": link,
            "permalink": d.get("permalink")
            or (f"https://www.reddit.com/comments/{link_bare}/_/{cid}/" if link_bare else ""),
            "fetched_utc": int(time.time()),
            "source": "arctic_shift",
        }

    @staticmethod
    def _norm_post(d: dict[str, Any]) -> dict[str, Any]:
        pid = d.get("id", "")
        title = (d.get("title") or "").strip()
        selftext = (d.get("selftext") or "").strip()
        body = f"{title}\n\n{selftext}".strip() if selftext else title
        return {
            "item_id": f"t3_{pid}",
            "account_hash": "",
            "kind": "post",
            "subreddit": str(d.get("subreddit", "")).lower(),
            "created_utc": d.get("created_utc"),
            "body": body,
            "score": d.get("score", 0),
            "parent_id": "",
            "link_id": f"t3_{pid}",
            "permalink": d.get("permalink") or f"https://www.reddit.com/comments/{pid}/",
            "fetched_utc": int(time.time()),
            "source": "arctic_shift",
        }

    # -- public -------------------------------------------------
    def fetch_history(
        self,
        username: str,
        *,
        after: int = 0,
        include_posts: bool = True,
        max_items: int | None = None,
    ) -> pd.DataFrame:
        def _take(path: str) -> Iterator[dict[str, Any]]:
            for n, d in enumerate(
                self._paginate(path, {"author": username, "after": after}), start=1
            ):
                yield d
                if max_items is not None and n >= max_items:
                    return

        rows = [self._norm_comment(d) for d in _take("comments/search")]
        if include_posts:
            rows += [self._norm_post(d) for d in _take("posts/search")]
        df = _rows_to_frame(rows, "arctic_shift")
        if not df.empty:
            df = df.drop_duplicates("item_id").sort_values("created_utc").reset_index(drop=True)
        return df

    def active_authors(
        self, subreddit: str, *, after: int, before: int, cap: int = 5000
    ) -> list[str]:
        """Collect distinct comment authors in a subreddit/time window — a cheap
        way to seed a candidate pool for the sample frame.
        """
        authors: dict[str, int] = {}
        for d in self._paginate(
            "comments/search", {"subreddit": subreddit, "after": after, "before": before}
        ):
            a = d.get("author", "")
            if a and a not in ("[deleted]", "AutoModerator"):
                authors[a] = authors.get(a, 0) + 1
            if len(authors) >= cap:
                break
        return sorted(authors, key=authors.get, reverse=True)


class ArcticShiftDumpClient:
    """DuckDB over the Hugging Face Parquet dumps (``open-index/arctic``).

    For one-off big scans — building a sample frame across many accounts. Slow
    per-author (no predicate pushdown on ``author`` across 261 GB); prefer
    :class:`ArcticShiftClient` for targeted history.
    """

    HF_GLOB = "hf://datasets/open-index/arctic/data"

    def __init__(self, hf_glob: str | None = None):
        self.hf_glob = (hf_glob or self.HF_GLOB).rstrip("/")

    def _con(self):
        import duckdb

        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        return con

    def sample_frame_query(self, years: list[int], min_comments: int, min_subreddits: int) -> str:
        yrs = ",".join(f"'{self.hf_glob}/comments/{y}/**/*.parquet'" for y in years)
        return f"""
        SELECT author,
               count(*)                        AS n_comments,
               count(DISTINCT subreddit)       AS n_subreddits,
               min(created_utc)                AS first_utc,
               max(created_utc)                AS last_utc
        FROM read_parquet([{yrs}])
        WHERE author NOT IN ('[deleted]', 'AutoModerator')
        GROUP BY author
        HAVING n_comments >= {min_comments} AND n_subreddits >= {min_subreddits}
        """

    def sample_frame(self, years: list[int], min_comments: int, min_subreddits: int) -> pd.DataFrame:
        con = self._con()
        return con.execute(self.sample_frame_query(years, min_comments, min_subreddits)).df()


class PrawClient:
    """Official Reddit API (PRAW) — freshness / gap-filling only."""

    def __init__(self, **praw_kwargs: object):
        self.praw_kwargs = praw_kwargs

    def _reddit(self):
        import praw

        return praw.Reddit(**self.praw_kwargs)

    def fetch_recent(self, username: str, limit: int = 1000) -> pd.DataFrame:
        r = self._reddit()
        redditor = r.redditor(username)
        rows: list[dict[str, Any]] = []
        for c in redditor.comments.new(limit=limit):
            rows.append(
                {
                    "item_id": f"t1_{c.id}",
                    "account_hash": "",
                    "kind": "comment",
                    "subreddit": str(c.subreddit).lower(),
                    "created_utc": int(c.created_utc),
                    "body": c.body,
                    "score": int(c.score),
                    "parent_id": c.parent_id,
                    "link_id": c.link_id,
                    "permalink": f"https://www.reddit.com{c.permalink}",
                    "fetched_utc": int(time.time()),
                    "source": "praw",
                }
            )
        for s in redditor.submissions.new(limit=limit):
            body = f"{s.title}\n\n{s.selftext}".strip() if s.selftext else s.title
            rows.append(
                {
                    "item_id": f"t3_{s.id}",
                    "account_hash": "",
                    "kind": "post",
                    "subreddit": str(s.subreddit).lower(),
                    "created_utc": int(s.created_utc),
                    "body": body,
                    "score": int(s.score),
                    "parent_id": "",
                    "link_id": f"t3_{s.id}",
                    "permalink": f"https://www.reddit.com{s.permalink}",
                    "fetched_utc": int(time.time()),
                    "source": "praw",
                }
            )
        return _rows_to_frame(rows, "praw")
