"""Build the E1/E2/E3 networks for one cached account corpus.

    python -m scripts.build_network --account <account_hash> \
        --config config/default.yaml

Milestone 1.2+ : reads ``data/accounts/{hash}/chunks.parquet``, runs the
estimators with the configured (real) rater, writes network artifacts to
``data/networks/{hash}/``. STUB until ingest lands — see docs/PLAN.md §10 step 11.
"""

from __future__ import annotations

import argparse

import pandas as pd

from rtn.config import REPO_ROOT, load_config
from rtn.estimate import build_brief, estimate_e1, estimate_e2, estimate_generic
from rtn.estimate.e3_covariation import estimate_e3
from rtn.network import Network
from rtn.prompts import get_prompts
from rtn.rater import build_rater
from rtn.traits import load_traits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, help="account_hash")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, *args.override)
    vocab = load_traits(cfg.trait_vocab_path)
    prompts = get_prompts(cfg.prompt_version)

    chunks_path = REPO_ROOT / "data" / "accounts" / args.account / "chunks.parquet"
    if not chunks_path.exists():
        raise SystemExit(
            f"no corpus at {chunks_path}. Run ingest first "
            "(Milestone 1.2 — ingest/clients.py is still a stub)."
        )
    chunks = pd.read_parquet(chunks_path)
    chunk_ids = chunks["chunk_id"].tolist()
    chunk_texts = chunks["text"].tolist()

    rater = build_rater(cfg, vocab=vocab)
    out_dir = REPO_ROOT / "data" / "networks" / args.account
    ch8, pv = cfg.hash8, cfg.prompt_version

    brief = build_brief(
        args.account, chunk_texts, vocab, prompts, rater,
        quotes_per_trait=int(cfg.get("estimate.e1.evidence_quotes_per_trait", 8)),
        strategy=cfg.get("estimate.e1.brief_strategy", "per_chunk"),
        config_hash=ch8,
    )
    e1 = estimate_e1(
        brief, vocab, prompts, rater, n_replicates=cfg.replicates,
        ablation_strength=cfg.get("estimate.e1.ablation_strength", "strong"),
        config_hash=ch8,
    )
    e1.save(out_dir / Network.artifact_name("e1", ch8, pv))

    if cfg.get("estimate.e2.enabled", True):
        if cfg.get("estimate.e2.per_account", True):
            e2 = estimate_e2(
                brief, vocab, prompts, rater, n_replicates=cfg.replicates,
                account_hash=args.account, config_hash=ch8,
            )
            e2.save(out_dir / Network.artifact_name("e2", ch8, pv))
        if cfg.get("estimate.e2.fit_generic_prior", True):
            g = estimate_generic(
                vocab, prompts, rater, n_replicates=cfg.replicates, config_hash=ch8
            )
            g.save(REPO_ROOT / "data" / "networks" / "_generic" / Network.artifact_name("e2_generic", ch8, pv))

    if cfg.get("estimate.e3.enabled", True):
        e3 = estimate_e3(
            args.account, chunk_ids, chunk_texts, vocab, prompts, rater,
            n_replicates=cfg.replicates,
            min_chunks=int(cfg.get("estimate.e3.min_chunks", 20)),
            ebic_gamma=float(cfg.get("estimate.e3.ebic_gamma", 0.5)),
            config_hash=ch8,
        )
        e3["undirected"].save(out_dir / Network.artifact_name("e3_undirected", ch8, pv))
        if e3["directed"] is not None:
            e3["directed"].save(out_dir / Network.artifact_name("e3_directed", ch8, pv))

    print(f"wrote networks for {args.account} to {out_dir}")


if __name__ == "__main__":
    main()
