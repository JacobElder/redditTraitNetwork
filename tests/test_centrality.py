import numpy as np
from scipy.stats import spearmanr

from rtn.centrality import (
    all_centralities,
    eigenvector_centrality,
    pagerank,
    personalized_pagerank,
    sla_centrality,
)
from rtn.network import Network


def _random_dep_matrix(k=12, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 1, size=(k, k))
    d[rng.random((k, k)) < 0.6] = 0.0
    np.fill_diagonal(d, 0.0)
    return d


def test_sla_matches_eigenvector():
    d = np.abs(_random_dep_matrix(15, 1))  # non-negative -> clean dominant eig
    rho, _ = spearmanr(sla_centrality(d), eigenvector_centrality(d))
    assert rho > 0.98


def test_pagerank_is_a_distribution():
    d = _random_dep_matrix()
    r = pagerank(d, alpha=0.85)
    assert abs(r.sum() - 1.0) < 1e-8
    assert (r >= 0).all()


def test_personalized_pagerank_follows_teleport():
    d = np.zeros((5, 5))  # no edges -> pure teleport
    tv = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
    r = personalized_pagerank(d, tv, alpha=0.85)
    assert np.argmax(r) == 0


def test_all_centralities_shape():
    net = Network(estimator="t", traits=tuple(f"x{i}" for i in range(8)), d=_random_dep_matrix(8, 3))
    df = all_centralities(net, density=np.ones(8) / 8)
    assert set(df["measure"]) >= {"sla", "eigenvector", "pagerank", "ppagerank"}
    assert len(df[df.measure == "sla"]) == 8
