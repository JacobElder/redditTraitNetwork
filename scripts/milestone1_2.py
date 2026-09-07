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
from scipy.stats import pearsonr, spearmanr

from rtn.centrality import all_centralities, personalized_pagerank, sla_centrality
from rtn.config import REPO_ROOT, load_config
from rtn.estimate import load_account_networks, nomothetic_network, stack_long
from rtn.estimate.node_weights import node_weights
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
    ap.add_argument("--config-hash", default=None, help="disambiguate if multiple builds on disk")
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config, *args.override)
    vocab = load_traits(cfg.trait_vocab_path)
    pv = cfg.prompt_version
    # auto-detect the config hash from the artifacts build_networks wrote, so
    # milestone1_2 doesn't have to be passed the exact same --override chain
    ch8 = args.config_hash

    nets = load_account_networks(args.estimator, ch8, pv)
    if len(nets) < 5:
        raise SystemExit(
            f"only {len(nets)} account networks for {args.estimator}/*/{pv} — "
            "run scripts.build_networks first"
        )
    first = nets[next(iter(nets))]
    ch8 = first.config_hash
    replicates = first.n_replicates or cfg.replicates
    print(f"pooling {len(nets)} accounts ({args.estimator}) · config {ch8} · {replicates} replicate(s)")

    d_bar, b = nomothetic_network(nets, vocab, estimator=f"{args.estimator}_pooled")
    long_df = stack_long(nets)
    vc = fit_variance_partition(long_df, replicates=replicates, n_boot=args.boot)

    # convergence of the three nomothetic estimates
    convergence: dict[str, float] = {}
    gpath = REPO_ROOT / "data" / "networks" / "_generic" / Network.artifact_name("e2_generic", ch8, pv)
    if gpath.exists():
        gen = Network.load(gpath)
        convergence["D̄ vs generic prior (E2)"] = float(
            pearsonr(_offdiag(d_bar.d), _offdiag(gen.d))[0]
        )
    # E1 is directed; E3-undirected is symmetric — compare against symmetrized E1
    d_bar_sym = (d_bar.d + d_bar.d.T) / 2
    for est, label in [("e3_undirected", "undirected"), ("e3_directed", "directed VAR")]:
        e3 = load_account_networks(est, ch8, pv)
        if not e3:
            continue
        e3_bar, _ = nomothetic_network(e3, vocab, estimator=f"{est}_pooled")
        ref = _offdiag(d_bar_sym) if est == "e3_undirected" else _offdiag(d_bar.d)
        convergence[f"D̄ vs pooled E3 ({label})"] = float(
            pearsonr(ref, _offdiag(e3_bar.d))[0]
        )
        # per-account convergence (not just the pooled means)
        per = []
        for a, net in nets.items():
            if a not in e3:
                continue
            e1m = (net.d + net.d.T) / 2 if est == "e3_undirected" else net.d
            per.append(pearsonr(_offdiag(e1m), _offdiag(e3[a].d))[0])
        if per:
            convergence[f"E1↔E3 ({label}), mean per-account"] = float(np.mean(per))

    # KEY IDIOGRAPHIC TEST: does E1's *residual* (person-specific) structure show
    # up in E3? If Bᵢ is real, (Dᵢ_E1 − D̄_E1) correlates with (E3ᵢ − D̄_E3).
    # If Bᵢ is elicitation noise, this is ~0. E3 shares no failure mode with E1.
    e3u = load_account_networks("e3_undirected", ch8, pv)
    if e3u:
        e3u_bar, _ = nomothetic_network(e3u, vocab, estimator="e3u_pooled")
        resid_r = []
        for a, net in nets.items():
            if a not in e3u:
                continue
            e1_resid = _offdiag((net.d + net.d.T) / 2 - d_bar_sym)
            e3_resid = _offdiag(e3u[a].d - e3u_bar.d)
            resid_r.append(pearsonr(e1_resid, e3_resid)[0])
        if resid_r:
            convergence["E1 residual ↔ E3 residual (idiographic corroboration)"] = float(
                np.mean(resid_r)
            )
            convergence["  — range across accounts"] = (
                round(float(min(resid_r)), 3),
                round(float(max(resid_r)), 3),
            )

    cent = all_centralities(d_bar)

    # -- theory-aligned idiographic operationalisation (docs/PLAN.md §2b) --
    # shared network D̄, node weighting per account (Sloman/Love/Ahn, Elder).
    # Centrality = personalised_pagerank(D̄, teleport = the account's node weights).
    import itertools

    base_c = sla_centrality(d_bar.d)
    pw_density, consist_ds, consist_de3 = {}, [], []
    for a in nets:
        wd = node_weights(cfg.cache_path, a, vocab, "density")
        ws = node_weights(cfg.cache_path, a, vocab, "self_relevance")
        we3 = node_weights(cfg.cache_path, a, vocab, "e3_salience")
        pw_density[a] = personalized_pagerank(d_bar.d, wd)
        consist_ds.append(spearmanr(wd, ws)[0])
        consist_de3.append(spearmanr(wd, we3)[0])  # brief vs E3 — independent estimators
    between = [
        spearmanr(pw_density[a], pw_density[b])[0]
        for a, b in itertools.combinations(nets, 2)
    ]
    move = [spearmanr(pw_density[a], base_c)[0] for a in nets]
    node_weighting = {
        "weight consistency: density ↔ self-relevance, mean ρ": round(float(np.mean(consist_ds)), 3),
        "weight CORROBORATION: brief-density ↔ E3-salience, mean ρ": round(float(np.mean(consist_de3)), 3),
        "  — E3-salience range across accounts": (
            round(float(min(consist_de3)), 2), round(float(max(consist_de3)), 2)),
        "between-account personalised-centrality, mean ρ": round(float(np.mean(between)), 3),
        "personalised vs unweighted D̄ centrality, mean ρ": round(float(np.mean(move)), 3),
    }
    convergence.update({f"[node weighting] {k}": v for k, v in node_weighting.items()})

    out_dir = REPO_ROOT / "data" / "networks" / "_pooled"
    d_bar.save(out_dir / Network.artifact_name(f"{args.estimator}_pooled", ch8, pv))
    dev_frames = []
    for acct, mat in b.items():
        f = Network(estimator="deviation", traits=vocab.names, d=mat).to_long_df()
        f["account_hash"] = acct
        dev_frames.append(f)
    pd.concat(dev_frames, ignore_index=True).to_parquet(
        out_dir / f"deviations__{ch8}__{pv}.parquet", index=False
    )
    (out_dir / f"variance__{ch8}__{pv}.json").write_text(json.dumps(vc.as_row(), indent=2, default=str))

    report_path = write_variance_partition(vc, cent, convergence, cfg.get("report.dir", "reports"))

    print(json.dumps(vc.as_row(), indent=2, default=str))
    for k, v in convergence.items():
        print(f"  {k}: {v if isinstance(v, tuple) else f'{v:+.3f}'}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
