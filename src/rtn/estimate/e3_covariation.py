"""E3 — behavioural covariation. INDEPENDENT convergent evidence.

The rater only scores each chunk on every trait (it never states a dependency).
Structure is then estimated from the chunk x trait matrix:

* undirected: EBICglasso (graphical lasso + EBIC model selection) -> partial
  correlation network.
* directed: lag-1 graphical VAR over time-ordered chunks -> temporal network
  ``beta[i, j]`` = effect of trait i at t-1 on trait j at t (matches the
  "j depends on i" convention), plus a contemporaneous partial-correlation net.

E3 estimates covariation, not conceptual dependency. Convergence with E1 is
evidence; identity is not claimed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
from sklearn.linear_model import LassoCV

from ..network import Network
from ..prompts.base import Prompts
from ..rater import Rater
from ..traits import TraitVocab
from .common import elicit_trait_vector


def rate_chunks(
    account_hash: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    vocab: TraitVocab,
    prompts: Prompts,
    rater: Rater,
    *,
    n_replicates: int,
    config_hash: str | None = None,
) -> pd.DataFrame:
    rows = []
    for cid, text in zip(chunk_ids, chunk_texts):
        mean, _, _ = elicit_trait_vector(
            rater,
            prompts.e3_chunk(text, vocab),
            vocab,
            {
                "account_hash": account_hash,
                "prompt_version": prompts.version,
                "task": "e3_chunk",
                "trait": cid,
                "config_hash": config_hash,
            },
            n_replicates,
        )
        rows.append({"chunk_id": cid, **mean})
    return pd.DataFrame(rows).set_index("chunk_id")


def _standardize(x: np.ndarray) -> np.ndarray:
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x = x.copy()
    x[inds] = np.take(col_mean, inds[1])
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def _ebic(emp_cov: np.ndarray, precision: np.ndarray, n: int, gamma: float) -> float:
    p = emp_cov.shape[0]
    sign, logdet = np.linalg.slogdet(precision)
    ll = (n / 2.0) * (logdet - np.trace(emp_cov @ precision))
    e = int((np.abs(precision) > 1e-8).sum() - p) // 2  # off-diag nonzeros
    return -2.0 * ll + e * np.log(n) + 4.0 * e * gamma * np.log(p)


def partial_correlation(x: np.ndarray, method: str = "ledoitwolf") -> np.ndarray:
    """Partial-correlation network from the chunk x trait matrix.

    ``ledoitwolf`` (default): Ledoit-Wolf shrinkage covariance, then invert. No
    sparsity selection — stable in the p ≈ n regime this project lives in
    (~45 chunks, 40 traits), where graphical-lasso EBIC collapses to an empty
    graph. Not sparse, but a valid weighted network for the E1↔E3 convergence
    check and for centrality.
    ``ebicglasso``: the sparse alternative (see :func:`ebicglasso`).
    """
    if method == "ebicglasso":
        return ebicglasso(x, gamma=0.0)
    from sklearn.covariance import LedoitWolf

    xs = _standardize(x)
    cov = LedoitWolf().fit(xs).covariance_
    prec = np.linalg.pinv(cov)
    dinv = np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
    pcor = -prec / dinv
    np.fill_diagonal(pcor, 0.0)
    return pcor


def ebicglasso(x: np.ndarray, gamma: float = 0.0, n_alphas: int = 10) -> np.ndarray:
    """Return the partial-correlation matrix chosen by EBIC over an alpha grid.
    ``gamma=0`` = plain BIC (less conservative); higher = sparser. At p ≈ n this
    still tends to over-sparsify — prefer :func:`partial_correlation`."""
    xs = _standardize(x)
    n = xs.shape[0]
    emp_cov = np.cov(xs, rowvar=False)
    emp_cov += 1e-6 * np.eye(emp_cov.shape[0])
    best = None
    densest = None  # fallback: fitted precision with the most off-diagonal edges
    for alpha in np.geomspace(0.003, 0.6, n_alphas):
        try:
            _, prec = graphical_lasso(emp_cov, alpha=alpha, max_iter=100)
        except (FloatingPointError, ValueError):
            continue
        n_edges = int((np.abs(prec) > 1e-8).sum())
        if densest is None or n_edges > densest[0]:
            densest = (n_edges, prec)
        score = _ebic(emp_cov, prec, n, gamma)
        if best is None or score < best[0]:
            best = (score, prec)
    if best is None:
        return np.zeros_like(emp_cov)
    prec = best[1]
    # if EBIC picked an (almost) empty graph, fall back to the densest fit so the
    # estimator still returns a usable structure (weak recovery is then a finding)
    if (np.abs(prec) > 1e-8).sum() <= prec.shape[0] + 2 and densest is not None:
        prec = densest[1]
    dinv = np.sqrt(np.outer(np.diag(prec), np.diag(prec)))
    pcor = -prec / dinv
    np.fill_diagonal(pcor, 0.0)
    return pcor


def graphical_var(
    x: np.ndarray, with_contemporaneous: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Lag-1 VAR by per-target LassoCV. Returns (beta, contemporaneous_pcor).

    ``beta[i, j]`` = standardized effect of trait i at t-1 on trait j at t.
    """
    xs = _standardize(x)
    n, p = xs.shape
    x_lag, x_now = xs[:-1], xs[1:]
    beta = np.zeros((p, p))
    resid = np.zeros_like(x_now)
    for j in range(p):
        if n - 1 < 5:
            continue
        model = LassoCV(cv=min(3, n - 2), n_alphas=12, max_iter=2000)
        model.fit(x_lag, x_now[:, j])
        beta[:, j] = model.coef_
        resid[:, j] = x_now[:, j] - model.predict(x_lag)
    if not with_contemporaneous:
        return beta, np.zeros((p, p))
    try:
        contemp = ebicglasso(resid)
    except Exception:
        contemp = np.zeros((p, p))
    return beta, contemp


def estimate_e3(
    account_hash: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    vocab: TraitVocab,
    prompts: Prompts,
    rater: Rater,
    *,
    n_replicates: int,
    min_chunks: int = 20,
    ebic_gamma: float = 0.5,
    undirected_method: str = "ledoitwolf",
    with_contemporaneous: bool = True,
    config_hash: str | None = None,
) -> dict[str, object]:
    X = rate_chunks(
        account_hash,
        chunk_ids,
        chunk_texts,
        vocab,
        prompts,
        rater,
        n_replicates=n_replicates,
        config_hash=config_hash,
    )
    X = X[list(vocab.names)]
    xmat = X.to_numpy(dtype=float)

    common = dict(
        traits=vocab.names,
        account_hash=account_hash,
        config_hash=config_hash,
        prompt_version=prompts.version,
        n_replicates=n_replicates,
    )

    if xmat.shape[0] >= 5:
        undirected = (
            partial_correlation(xmat, method=undirected_method)
            if undirected_method != "ebicglasso_gamma"
            else ebicglasso(xmat, ebic_gamma)
        )
    else:
        undirected = np.zeros((vocab.k, vocab.k))
    nets: dict[str, object] = {
        "X": X,
        "undirected": Network(estimator="e3_undirected", d=undirected, **common),
        "n_chunks": xmat.shape[0],
        "enough_chunks": xmat.shape[0] >= min_chunks,
    }
    if xmat.shape[0] >= min_chunks:
        beta, contemp = graphical_var(xmat, with_contemporaneous=with_contemporaneous)
        nets["directed"] = Network(estimator="e3_directed", d=beta, **common)
        nets["contemporaneous"] = Network(estimator="e3_contemp", d=contemp, **common)
    else:
        nets["directed"] = None
        nets["contemporaneous"] = None
    return nets
