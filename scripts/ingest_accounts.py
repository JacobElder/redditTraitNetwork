"""Fetch, hash, filter, and chunk Reddit histories into cached corpora.

    # explicit usernames
    RTN_HASH_SALT=... python -m scripts.ingest_accounts \
        --users alice,bob,carol --config config/default.yaml

    # from a file (one username per line, # comments allowed)
    RTN_HASH_SALT=... python -m scripts.ingest_accounts --users-file accounts.txt

    # discover a candidate pool from a subreddit's recent commenters
    RTN_HASH_SALT=... python -m scripts.ingest_accounts \
        --from-subreddit AskHistorians --since 2023-01-01 --sample 60

Writes data/accounts/{hash}/{items,chunks}.parquet and appends
data/accounts/index.parquet (hashes only, no raw usernames). Raw username -> hash
map lives only in secrets/usermap.json (gitignored).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from rtn.config import load_config
from rtn.ingest.clients import ArcticShiftClient, PrawClient
from rtn.ingest.pipeline import ingest_username


def _parse_date(s: str) -> int:
    return int(dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=dt.UTC).timestamp())


def _collect_usernames(args) -> list[str]:
    users: list[str] = []
    if args.users:
        users += [u.strip() for u in args.users.split(",") if u.strip()]
    if args.users_file:
        for line in Path(args.users_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                users.append(line)
    if args.from_subreddit:
        client = ArcticShiftClient()
        since = _parse_date(args.since) if args.since else _parse_date("2020-01-01")
        until = _parse_date(args.until) if args.until else int(dt.datetime.now(dt.UTC).timestamp())
        subs = [s.strip() for s in args.from_subreddit.split(",") if s.strip()]
        per_sub = max(1, -(-args.sample // len(subs)))  # spread the quota across subs
        for si, sub in enumerate(subs):
            pool = client.active_authors(
                sub, after=since, before=until, cap=per_sub * 6, seed=si
            )
            users += pool[:per_sub]
            print(f"  seed r/{sub}: {min(len(pool), per_sub)} candidates")
    # dedupe, preserve order
    seen: set[str] = set()
    return [u for u in users if not (u in seen or seen.add(u))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--users", help="comma-separated usernames")
    ap.add_argument("--users-file", help="file with one username per line")
    ap.add_argument(
        "--from-subreddit",
        help="seed candidates from these subreddits' commenters (comma-separated; "
        "--sample is split evenly across them)",
    )
    ap.add_argument("--since", help="YYYY-MM-DD (with --from-subreddit)")
    ap.add_argument("--until", help="YYYY-MM-DD (with --from-subreddit)")
    ap.add_argument("--sample", type=int, default=50, help="candidates to take from --from-subreddit")
    ap.add_argument("--gap-fill", action="store_true", help="also pull recent items via PRAW")
    ap.add_argument("--overwrite", action="store_true", help="re-fetch even if cached")
    args = ap.parse_args()

    cfg = load_config(args.config, *args.override)
    usernames = _collect_usernames(args)
    if not usernames:
        sys.exit("no usernames — pass --users, --users-file, or --from-subreddit")

    client = ArcticShiftClient(
        base_url=cfg.get("ingest.arctic_shift_base", "https://arctic-shift.photon-reddit.com/api"),
        pause_s=float(cfg.get("ingest.request_pause_s", 0.35)),
    )
    praw_client = PrawClient(**cfg.get("ingest.praw", {})) if args.gap_fill else None

    print(f"ingesting {len(usernames)} account(s) · config {cfg.hash8}")
    rows = []
    for i, name in enumerate(usernames, 1):
        try:
            r = ingest_username(
                name, cfg, client=client, praw_client=praw_client,
                gap_fill=args.gap_fill, overwrite=args.overwrite,
            )
        except Exception as e:
            print(f"  [{i}/{len(usernames)}] {name!r}: ERROR {e}")
            continue
        flag = "ok " if r.eligible else "skip"
        print(
            f"  [{i}/{len(usernames)}] {r.account_hash} {flag} "
            f"{r.n_comments}c/{r.n_posts}p · {r.n_subreddits} subs · "
            f"{r.span_days:.0f}d · excl {r.excluded_fraction:.0%} · {r.n_chunks} chunks · {r.reason}"
        )
        rows.append(vars(r))

    if rows:
        summ = pd.DataFrame(rows)
        n_ok = int(summ["eligible"].sum())
        print(f"\n{n_ok}/{len(summ)} eligible. index -> data/accounts/index.parquet")


if __name__ == "__main__":
    main()
