"""Milestone 1.1 — synthetic recovery harness.

Runs the full estimator pipeline against :mod:`rtn.validate.synthetic` ground
truth over a grid of (ground-truth DAG) x (evidence volume) x (noise) x
(replicate) and reports:

* edge-weight recovery: correlation of estimated off-diagonal weights with the
  true dependency matrix, for E1, E1-minus-generic, E2, E3-directed,
  E3-undirected.
* centrality rank recovery: Spearman rho between centrality on the estimated
  network and on the true network, per measure.
* E1<->E3 convergence: agreement between the two estimators that do NOT share a
  failure mode.

Use the results to choose ``k`` (ablation count is fixed at the vocab size here,
but the noise/volume curves tell you the minimum viable corpus), chunk count,
and replicate count before touching real accounts.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning

from ..centrality import all_centralities
from ..estimate import build_brief, estimate_e1, estimate_e2, estimate_generic
from ..estimate.e3_covariation import estimate_e3
from ..prompts import get_prompts
from ..rater import build_rater
from ..traits import load_traits
from .synthetic import (
    GroundTruth,
    SyntheticOracle,
    make_generic_prior,
    make_ground_truth,
    simulate_history,
)


def _offdiag(m: np.ndarray) -> np.ndarray:
    mask = ~np.eye(m.shape[0], dtype=bool)
    return m[mask]


def _corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    f = pearsonr if method == "pearson" else spearmanr
    return float(f(a[ok], b[ok])[0])


def _centrality_wide(net) -> pd.DataFrame:
    return all_centralities(net).pivot(index="trait", columns="measure", values="value")


def _centrality_rank_recovery(est_net, true_d, traits) -> dict[str, float]:
    from ..network import Network

    true_net = Network(estimator="synthetic", traits=traits, d=true_d)
    est = _centrality_wide(est_net)
    tru = _centrality_wide(true_net)
    out = {}
    for measure in est.columns:
        out[measure] = _corr(
            est.loc[list(traits), measure].to_numpy(),
            tru.loc[list(traits), measure].to_numpy(),
            method="spearman",
        )
    return out


def run_recovery(config) -> pd.DataFrame:
    vocab = load_traits(config.trait_vocab_path)
    prompts = get_prompts(config.prompt_version)
    if config.model["backend"] != "mock":
        raise ValueError("synthetic recovery requires model.backend: mock")

    sy = config.get("validate.synthetic", {})
    n_dags = int(sy.get("n_ground_truth_dags", 12))
    volumes = list(sy.get("evidence_volume", ["thin", "medium", "thick"]))
    noises = list(sy.get("noise_levels", [0.0, 0.15, 0.30]))
    reps = int(sy.get("replicates_per_cell", 10))
    seed0 = int(sy.get("seed", 20260831))
    n_replicates = config.replicates
    e1_strength = config.get("estimate.e1.ablation_strength", "strong")
    e3_min_chunks = int(config.get("estimate.e3.min_chunks", 20))
    e3_gamma = float(config.get("estimate.e3.ebic_gamma", 0.5))
    quotes_per_trait = int(config.get("estimate.e1.evidence_quotes_per_trait", 8))

    d_generic = make_generic_prior(vocab, seed0)
    rows = []

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="An input array is constant")

    # E2 generic is account-independent: compute once, reuse everywhere. It still
    # depends on the oracle's noise_sd, so use a mid-volume oracle as reference.
    _ref_gt = make_ground_truth(vocab, d_generic, seed=seed0 + 777)
    _ref_hist = simulate_history(_ref_gt, "medium", 0.0, seed0 + 778, "ref")
    generic = estimate_generic(
        vocab, prompts,
        build_rater(config, vocab=vocab, oracle=SyntheticOracle(_ref_hist, seed=seed0 + 778)),
        n_replicates=n_replicates, config_hash=config.hash8,
    )

    for dag_ix in range(n_dags):
        gt: GroundTruth = make_ground_truth(vocab, d_generic, seed=seed0 + 1000 + dag_ix)
        true_off = _offdiag(gt.d_true)
        true_abs_off = _offdiag(np.abs(gt.d_true))

        for vol, noise, rep in itertools.product(volumes, noises, range(reps)):
            cell_seed = seed0 + 10_000 * dag_ix + 100 * volumes.index(vol) + 10 * noises.index(noise) + rep
            account = f"syn_{dag_ix:02d}_{vol}_{noise:g}_{rep:02d}"
            hist = simulate_history(gt, vol, noise, cell_seed, account)
            oracle = SyntheticOracle(hist, seed=cell_seed)
            rater = build_rater(config, vocab=vocab, oracle=oracle)

            brief = build_brief(
                account, hist.chunk_texts, vocab, prompts, rater,
                quotes_per_trait=quotes_per_trait, strategy="single",
                config_hash=config.hash8,
            )
            e1 = estimate_e1(
                brief, vocab, prompts, rater,
                n_replicates=n_replicates, ablation_strength=e1_strength,
                config_hash=config.hash8,
            )
            e2 = estimate_e2(
                brief, vocab, prompts, rater,
                n_replicates=n_replicates, account_hash=account, config_hash=config.hash8,
            )
            e3 = estimate_e3(
                account, hist.chunk_ids, hist.chunk_texts, vocab, prompts, rater,
                n_replicates=n_replicates, min_chunks=e3_min_chunks,
                ebic_gamma=e3_gamma, with_contemporaneous=False,
                config_hash=config.hash8,
            )

            e1_off = _offdiag(e1.d)
            e1mg_off = _offdiag(e1.d - generic.d)
            e2_off = _offdiag(e2.d)
            e3u_off = _offdiag(e3["undirected"].d)
            e3d = e3["directed"]

            rec = {
                "dag": dag_ix, "volume": vol, "noise": noise, "replicate": rep,
                "n_chunks": e3["n_chunks"],
                "edge_r_e1": _corr(e1_off, true_off),
                "edge_r_e1_minus_generic": _corr(e1mg_off, true_off),
                "edge_r_e2": _corr(e2_off, true_abs_off),
                "edge_r_e3_undirected": _corr(e3u_off, np.abs(true_off)),
                "edge_r_e3_directed": (
                    _corr(_offdiag(e3d.d), true_off) if e3d is not None else np.nan
                ),
            }
            for measure, rho in _centrality_rank_recovery(e1, gt.d_true, vocab.names).items():
                rec[f"cent_rho_e1_{measure}"] = rho
            # E1<->E3 convergence (only when E3 directed exists)
            if e3d is not None:
                c_e1 = _centrality_wide(e1)["sla"].loc[list(vocab.names)].to_numpy()
                c_e3 = _centrality_wide(e3d)["sla"].loc[list(vocab.names)].to_numpy()
                rec["e1_e3_sla_rho"] = _corr(c_e1, c_e3, method="spearman")
                rec["e1_e3_edge_r"] = _corr(e1_off, _offdiag(e3d.d))
            rows.append(rec)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean recovery by (volume, noise)."""
    metric_cols = [c for c in df.columns if c.startswith(("edge_r_", "cent_rho_", "e1_e3_"))]
    return (
        df.groupby(["volume", "noise"])[metric_cols]
        .mean()
        .reset_index()
        .sort_values(["volume", "noise"])
    )
