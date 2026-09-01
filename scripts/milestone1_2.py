"""Milestone 1.2 — nomothetic / idiographic variance partition.

Run after scripts.build_networks has produced per-account E1 (and E3) artifacts.

    python -m scripts.milestone1_2 --config config/default.yaml \
        --override config/milestone1_2.yaml

Loads the per-account networks, pools them into D-bar + per-account deviations,
decomposes the edge-weight variance, checks D-bar against the generic prior and
pooled E3, and writes reports/milestone1.md §1.2. Aggregate only.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rtn.centrality import all_centralities
from rtn.config import REPO_ROOT, load_config
from rtn.estimate import load_account_networks, nomothetic_network, stack_long
from rtn.network import Network
from rtn.traits import load_traits
from rtn.validate.report import write_variance_partition
from rtn.validate.variance import fit_variance_partition


def _offdiag(m):
    return m[~np.eye(m.shape[0], dtype=bool)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--estimator", default="e1", help="per-account estimator to pool")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config, *args.override)
    vocab = load_traits(cfg.trait_vocab_path)
    ch8, pv = cfg.hash8, cfg.prompt_version

    nets = load_account_networks(args.estimator, ch8, pv)
    if len(nets) < 5:
        raise SystemExit(
            f"only {len(nets)} account networks for {args.estimator}/{ch8}/{pv} — "
            "run scripts.build_networks first"
        )
    print(f"pooling {len(nets)} accounts ({args.estimator})")

    d_bar, b = nomothetic_network(nets, vocab, estimator=f"{args.estimator}_pooled")
    long_df = stack_long(nets)
    vc = fit_variance_partition(long_df, replicates=cfg.replicates, n_boot=args.boot)

    # convergence of the three nomothetic estimates
    convergence: dict[str, float] = {}
    gpath = REPO_ROOT / "data" / "networks" / "_generic" / Network.artifact_name("e2_generic", ch8, pv)
    if gpath.exists():
        gen = Network.load(gpath)
        convergence["D̄ vs generic prior (E2)"] = float(
            pearsonr(_offdiag(d_bar.d), _offdiag(gen.d))[0]
        )
    e3 = load_account_networks("e3_directed", ch8, pv)
    if e3:
        e3_bar, _ = nomothetic_network(e3, vocab, estimator="e3_pooled")
        convergence["D̄ vs pooled E3 (directed)"] = float(
            pearsonr(_offdiag(d_bar.d), _offdiag(e3_bar.d))[0]
        )

    cent = all_centralities(d_bar)

    out_dir = REPO_ROOT / "data" / "networks" / "_pooled"
    d_bar.save(out_dir / Network.artifact_name(f"{args.estimator}_pooled", ch8, pv))
    pd.concat(
        [n.to_long_df().assign(account_hash=a) for a, n in
         ((a, Network(estimator="deviation", traits=vocab.names, d=b[a])) for a in b)],
        ignore_index=True,
    ).to_parquet(out_dir / f"deviations__{ch8}__{pv}.parquet", index=False)
    (out_dir / f"variance__{ch8}__{pv}.json").write_text(json.dumps(vc.as_row(), indent=2, default=str))

    report_path = write_variance_partition(vc, cent, convergence, cfg.get("report.dir", "reports"))

    print(json.dumps(vc.as_row(), indent=2, default=str))
    for k, v in convergence.items():
        print(f"  {k}: r={v:+.3f}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
