"""Print a comma-separated account-hash list for `build_networks --only`.

The ingest index has no batch column, so batches are identified by
`ingested_utc`. This picks a balanced sample across time-of-ingest cohorts
(≈ seed batches) so the network build doesn't just favour the earliest,
most-prolific accounts.

    python -m scripts.select_accounts --per-cohort 12 [--eligible-only] [--gap-hours 6]
    python -m scripts.build_networks ... --only "$(python -m scripts.select_accounts --per-cohort 12)"
"""

from __future__ import annotations

import argparse

import pandas as pd

from rtn.config import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cohort", type=int, default=12, help="accounts to take from each ingest cohort")
    ap.add_argument("--gap-hours", type=float, default=6.0, help="ingest-time gap that separates cohorts")
    ap.add_argument("--eligible-only", action="store_true", default=True)
    ap.add_argument("--sort", choices=["comments", "span", "subreddits"], default="comments")
    args = ap.parse_args()

    df = pd.read_parquet(REPO_ROOT / "data" / "accounts" / "index.parquet")
    if args.eligible_only:
        df = df[df["eligible"]].copy()
    df = df.sort_values("ingested_utc").reset_index(drop=True)

    # split into cohorts wherever there's a > gap-hours jump in ingested_utc
    gap = args.gap_hours * 3600
    df["cohort"] = (df["ingested_utc"].diff().fillna(0) > gap).cumsum()

    sort_col = {"comments": "n_comments", "span": "span_days", "subreddits": "n_subreddits"}[args.sort]
    picks: list[str] = []
    for c, g in df.groupby("cohort"):
        picks += g.sort_values(sort_col, ascending=False)["account_hash"].head(args.per_cohort).tolist()

    import sys

    print(",".join(picks))
    print(
        f"# {len(picks)} accounts from {df['cohort'].nunique()} cohorts",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
