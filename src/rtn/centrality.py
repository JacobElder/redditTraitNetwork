"""Conceptual centrality over a trait-dependency matrix.

All measures take ``d`` with the project convention ``d[i, j]`` = "j depends on
i", so a node's centrality is driven by its **row** (outgoing) weights: a trait
many others depend on is central (Sloman, Love & Ahn, 1998).

Negative-weight handling is per-measure and documented on each function. The
SLA / eigenvector measures use the signed matrix; PageRank-family and
betweenness use magnitudes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import spearmanr


def _as_matrix(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    d = d.copy()
    np.fill_diagonal(d, 0.0)
    return d


def sla_centrality(
    d: np.ndarray, max_iter: int = 500, tol: float = 1e-10, damping: float = 0.85
) -> np.ndarray:
    """Sloman-Love-Ahn iteration on dependency **magnitudes** ``|d|``:

        c <- normalize( damping * (|d| @ c) + (1 - damping) * c )

    A trait is central when other traits depend strongly on it, regardless of
    coupling sign (use ``out_strength(signed=True)`` for the signed view).

    The lazy ``(1 - damping) * c`` term shifts every eigenvalue by
    ``(1 - damping)`` without changing the eigenvectors, so this still converges
    to the dominant eigenvector of ``|d|`` (matching :func:`eigenvector_centrality`)
    but stays numerically well-defined when ``|d|`` has a small or zero spectral
    radius (near-acyclic networks). Real trait-dependency networks contain
    cycles, so this is not usually load-bearing — but it keeps the measure from
    silently collapsing to a constant.
    """
    m = np.abs(_as_matrix(d))
    k = m.shape[0]
    c = np.ones(k) / np.sqrt(k)
    for _ in range(max_iter):
        c_new = damping * (m @ c) + (1.0 - damping) * c
        norm = np.linalg.norm(c_new)
        if norm < 1e-300:
            return np.zeros(k)
        c_new /= norm
        if np.linalg.norm(c_new - c) < tol or np.linalg.norm(c_new + c) < tol:
            c = c_new
            break
        c = c_new
    if c[np.argmax(np.abs(c))] < 0:
        c = -c
    return c


def eigenvector_centrality(d: np.ndarray) -> np.ndarray:
    """Dominant eigenvector of ``|d|`` (by ``|eigenvalue|``), computed directly.
    Same matrix as :func:`sla_centrality`, so the two should rank-correlate
    ~1.0; a gap indicates a bug.
    """
    d = np.abs(_as_matrix(d))
    vals, vecs = linalg.eig(d)
    idx = int(np.argmax(np.abs(vals)))
    v = np.real(vecs[:, idx])
    if v[np.argmax(np.abs(v))] < 0:
        v = -v
    return v


def _column_stochastic(mat: np.ndarray) -> np.ndarray:
    col_sums = mat.sum(axis=0)
    out = np.divide(
        mat, col_sums, out=np.zeros_like(mat), where=col_sums > 0
    )
    # dangling columns -> uniform
    dangling = col_sums == 0
    if dangling.any():
        out[:, dangling] = 1.0 / mat.shape[0]
    return out


def pagerank(
    d: np.ndarray, alpha: float = 0.85, teleport: np.ndarray | None = None,
    max_iter: int = 500, tol: float = 1e-12,
) -> np.ndarray:
    """PageRank on ``|d|``. An edge i->j (j depends on i) is a "vote" from j to i,
    so random-surfer mass flows along columns toward influential source traits.

    ``teleport`` (personalisation) defaults to uniform; pass an evidence-density
    vector for personalised PageRank.
    """
    m = np.abs(_as_matrix(d))
    k = m.shape[0]
    if teleport is None:
        tv = np.ones(k) / k
    else:
        tv = np.asarray(teleport, dtype=float)
        s = tv.sum()
        tv = np.ones(k) / k if s <= 0 else tv / s
    p = _column_stochastic(m)
    r = np.ones(k) / k
    for _ in range(max_iter):
        r_new = alpha * (p @ r) + (1 - alpha) * tv
        r_new /= r_new.sum()
        if np.abs(r_new - r).sum() < tol:
            r = r_new
            break
        r = r_new
    return r


def personalized_pagerank(
    d: np.ndarray, teleport: np.ndarray, alpha: float = 0.85
) -> np.ndarray:
    return pagerank(d, alpha=alpha, teleport=teleport)


def out_strength(d: np.ndarray, signed: bool = False) -> np.ndarray:
    d = _as_matrix(d)
    return d.sum(axis=1) if signed else np.abs(d).sum(axis=1)


def in_strength(d: np.ndarray, signed: bool = False) -> np.ndarray:
    d = _as_matrix(d)
    return d.sum(axis=0) if signed else np.abs(d).sum(axis=0)


def betweenness(d: np.ndarray) -> np.ndarray:
    """Betweenness on a distance graph ``1/|d|`` over present edges. Comparison
    measure only. Uses networkx if available, else a Floyd-Warshall count.
    """
    m = np.abs(_as_matrix(d))
    k = m.shape[0]
    try:
        import networkx as nx

        g = nx.DiGraph()
        g.add_nodes_from(range(k))
        for i in range(k):
            for j in range(k):
                if i != j and m[i, j] > 0:
                    g.add_edge(i, j, weight=1.0 / m[i, j])
        bc = nx.betweenness_centrality(g, weight="weight", normalized=True)
        return np.array([bc[i] for i in range(k)])
    except ImportError:  # pragma: no cover - networkx is a hard dep
        return np.full(k, np.nan)


def all_centralities(
    net,
    density: np.ndarray | None = None,
    sla_max_iter: int = 500,
    sla_tol: float = 1e-10,
    pagerank_alpha: float = 0.85,
) -> pd.DataFrame:
    """Tidy long DataFrame: one row per (trait, measure, value)."""
    d = net.d if hasattr(net, "d") else np.asarray(net)
    traits = list(net.traits) if hasattr(net, "traits") else list(range(d.shape[0]))

    measures = {
        "sla": sla_centrality(d, sla_max_iter, sla_tol),
        "eigenvector": eigenvector_centrality(d),
        "pagerank": pagerank(d, pagerank_alpha),
        "out_strength": out_strength(d),
        "in_strength": in_strength(d),
        "out_strength_signed": out_strength(d, signed=True),
        "betweenness": betweenness(d),
    }
    if density is not None:
        measures["ppagerank"] = personalized_pagerank(d, density, pagerank_alpha)

    rows = []
    for measure, vec in measures.items():
        for t, val in zip(traits, vec):
            rows.append({"trait": t, "measure": measure, "value": float(val)})
    return pd.DataFrame(rows)


def rank_correlation_matrix(
    cent_df: pd.DataFrame, method: str = "spearman"
) -> pd.DataFrame:
    """Spearman rank-correlation among centrality measures (wide over traits)."""
    wide = cent_df.pivot(index="trait", columns="measure", values="value")
    if method != "spearman":
        raise ValueError("only spearman supported")
    cols = list(wide.columns)
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            rho, _ = spearmanr(wide[a], wide[b])
            out.loc[a, b] = rho
    return out
