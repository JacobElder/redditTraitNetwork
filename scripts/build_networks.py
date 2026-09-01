"""Build E1/E2/E3 networks for every eligible account in the index.

    python -m scripts.build_networks --config config/default.yaml \
        --override config/milestone1_2.yaml [--limit N] [--only <hash,hash>]

Resumable: skips accounts whose E1 artifact already exists for the current
config hash + prompt version. The rater cache makes a re-run after a crash cost
nothing. Writes a progress line per account and a run summary to
reports/build_networks.log.
"""

from __future__ import annotations

import argparse
import time
import traceback

import pandas as pd

from rtn.config import REPO_ROOT, load_config
from rtn.estimate import build_brief, estimate_e1, estimate_e2, estimate_generic
from rtn.estimate.e3_covariation import estimate_e3
from rtn.network import Network
from rtn.prompts import get_prompts
from rtn.rater import build_rater
from rtn.traits import load_traits


def _eligible_hashes(only: str | None) -> list[str]:
    idx = REPO_ROOT / "data" / "accounts" / "index.parquet"
    if not idx.exists():
        raise SystemExit(f"no {idx} — run scripts.ingest_accounts first")
    df = pd.read_parquet(idx)
    df = df[df["eligible"]]
    if only:
        want = {h.strip() for h in only.split(",")}
        df = df[df["account_hash"].isin(want)]
    return df.sort_values("n_comments", ascending=False)["account_hash"].tolist()


def _already_built(account: str, ch8: str, pv: str) -> bool:
    return (
        REPO_ROOT / "data" / "networks" / account / Network.artifact_name("e1", ch8, pv)
    ).exists()


def build_one(account: str, cfg, vocab, prompts, rater) -> dict:
    ch8, pv = cfg.hash8, cfg.prompt_version
    chunks_path = REPO_ROOT / "data" / "accounts" / account / "chunks.parquet"
    chunks = pd.read_parquet(chunks_path)
    chunk_ids = chunks["chunk_id"].tolist()
    chunk_texts = chunks["text"].tolist()
    out_dir = REPO_ROOT / "data" / "networks" / account

    brief = build_brief(
        account, chunk_texts, vocab, prompts, rater,
        quotes_per_trait=int(cfg.get("estimate.e1.evidence_quotes_per_trait", 8)),
        config_hash=ch8,
    )
    e1 = estimate_e1(
        brief, vocab, prompts, rater, n_replicates=cfg.replicates,
        ablation_strength=cfg.get("estimate.e1.ablation_strength", "strong"),
        config_hash=ch8,
    )
    e1.save(out_dir / Network.artifact_name("e1", ch8, pv))

    if cfg.get("estimate.e2.enabled", True) and cfg.get("estimate.e2.per_account", True):
        estimate_e2(
            brief, vocab, prompts, rater, n_replicates=cfg.replicates,
            account_hash=account, config_hash=ch8,
        ).save(out_dir / Network.artifact_name("e2", ch8, pv))

    e3 = estimate_e3(
        account, chunk_ids, chunk_texts, vocab, prompts, rater,
        n_replicates=cfg.replicates,
        min_chunks=int(cfg.get("estimate.e3.min_chunks", 20)),
        ebic_gamma=float(cfg.get("estimate.e3.ebic_gamma", 0.5)),
        config_hash=ch8,
    )
    e3["undirected"].save(out_dir / Network.artifact_name("e3_undirected", ch8, pv))
    if e3["directed"] is not None:
        e3["directed"].save(out_dir / Network.artifact_name("e3_directed", ch8, pv))

    return {"n_chunks": e3["n_chunks"], "e3_directed": e3["directed"] is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", help="comma-separated account hashes")
    ap.add_argument("--rebuild", action="store_true", help="ignore existing artifacts")
    args = ap.parse_args()

    cfg = load_config(args.config, *args.override)
    vocab = load_traits(cfg.trait_vocab_path)
    prompts = get_prompts(cfg.prompt_version)
    rater = build_rater(cfg, vocab=vocab)
    ch8, pv = cfg.hash8, cfg.prompt_version

    hashes = _eligible_hashes(args.only)
    if args.limit:
        hashes = hashes[: args.limit]

    logf = (REPO_ROOT / "reports" / "build_networks.log").open("a")
    print(f"config {ch8} · prompt {pv} · backend {cfg.model['backend']} · {len(hashes)} accounts")

    # study-wide generic prior, once
    if cfg.get("estimate.e2.fit_generic_prior", True):
        gpath = REPO_ROOT / "data" / "networks" / "_generic" / Network.artifact_name("e2_generic", ch8, pv)
        if args.rebuild or not gpath.exists():
            print("  building generic prior d_generic ...")
            estimate_generic(vocab, prompts, rater, n_replicates=cfg.replicates, config_hash=ch8).save(gpath)

    done = skipped = failed = 0
    for i, account in enumerate(hashes, 1):
        if not args.rebuild and _already_built(account, ch8, pv):
            skipped += 1
            continue
        t0 = time.time()
        try:
            info = build_one(account, cfg, vocab, prompts, rater)
            done += 1
            msg = f"[{i}/{len(hashes)}] {account} ok · {info['n_chunks']} chunks · {time.time()-t0:.0f}s"
        except Exception as e:
            failed += 1
            msg = f"[{i}/{len(hashes)}] {account} FAILED · {e}"
            logf.write(traceback.format_exc() + "\n")
        print("  " + msg)
        logf.write(f"{time.strftime('%Y-%m-%d %H:%M')} {msg}\n")
        logf.flush()

    print(f"\ndone {done} · skipped {skipped} · failed {failed}")
    logf.close()


if __name__ == "__main__":
    main()
