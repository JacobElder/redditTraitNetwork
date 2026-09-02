"""Recompute E3 networks from the cached per-chunk trait ratings.

The expensive part of E3 is the per-chunk LLM ratings (~45 calls/account). The
network estimation on top of them (partial correlations, graphical VAR) is free
and can be re-run with a better method without re-eliciting.

    python -m scripts.recompute_e3 --config config/default.yaml \
        --override config/milestone1_2.yaml --override config/milestone1_2_free.yaml

Reads task='e3_chunk' rows from the rater cache, rebuilds each account's
chunk x trait matrix, and overwrites e3_undirected / e3_directed artifacts
(same config hash + prompt version — nothing upstream changes).
"""

from __future__ import annotations

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd

from rtn.config import REPO_ROOT, load_config
from rtn.estimate.e3_covariation import graphical_var, partial_correlation
from rtn.network import Network
from rtn.traits import load_traits


def _x_matrices_from_cache(cache_path, names, model_id: str | None):
    con = sqlite3.connect(cache_path)
    q = "SELECT trait, replicate, response, model FROM calls WHERE task='e3_chunk'"
    rows = con.execute(q).fetchall()
    per_acct: dict[str, dict[str, list]] = {}
    for chunk_id, rep, resp, model in rows:
        if model_id and model != model_id:
            continue
        acct = chunk_id.split(":")[0]
        per_acct.setdefault(acct, {}).setdefault(chunk_id, []).append(resp)
    out = {}
    for acct, chunks in per_acct.items():
        cids = sorted(chunks)
        X = np.full((len(cids), len(names)), np.nan)
        for k, cid in enumerate(cids):
            vals = np.full((len(chunks[cid]), len(names)), np.nan)
            for r, resp in enumerate(chunks[cid]):
                try:
                    d = json.loads(resp)
                except (json.JSONDecodeError, TypeError):
                    continue
                for j, n in enumerate(names):
                    v = d.get(n)
                    if isinstance(v, (int, float)):
                        vals[r, j] = v
            X[k] = np.nanmean(vals, axis=0)
        out[acct] = (cids, X)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--method", default=None, help="override estimate.e3.undirected_method")
    args = ap.parse_args()

    cfg = load_config(args.config, *args.override)
    vocab = load_traits(cfg.trait_vocab_path)
    names = list(vocab.names)
    ch8, pv = cfg.hash8, cfg.prompt_version
    method = args.method or cfg.get("estimate.e3.undirected_method", "ledoitwolf")
    min_chunks = int(cfg.get("estimate.e3.min_chunks", 20))

    mats = _x_matrices_from_cache(cfg.cache_path, names, None)
    if not mats:
        raise SystemExit("no e3_chunk rows in the cache")

    for acct, (cids, X) in sorted(mats.items()):
        out_dir = REPO_ROOT / "data" / "networks" / acct
        if not out_dir.exists():
            continue
        common = dict(traits=vocab.names, account_hash=acct, config_hash=ch8, prompt_version=pv)
        und = partial_correlation(X, method=method) if X.shape[0] >= 5 else np.zeros((40, 40))
        Network(estimator="e3_undirected", d=und, **common).save(
            out_dir / Network.artifact_name("e3_undirected", ch8, pv)
        )
        tag = f"{acct[:8]}: {X.shape[0]} chunks, undirected[{method}]"
        if X.shape[0] >= min_chunks:
            beta, _ = graphical_var(X, with_contemporaneous=False)
            Network(estimator="e3_directed", d=beta, **common).save(
                out_dir / Network.artifact_name("e3_directed", ch8, pv)
            )
            tag += " + directed"
        # save X for future re-analysis
        pd.DataFrame(X, index=cids, columns=names).to_parquet(
            out_dir / f"e3_X__{ch8}__{pv}.parquet"
        )
        print(f"  {tag}")

    print(f"recomputed E3 for {len(mats)} accounts")


if __name__ == "__main__":
    main()
