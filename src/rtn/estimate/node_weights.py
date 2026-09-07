"""Per-account node weightings for personalised centrality on the shared network.

The Sloman/Love/Ahn and Elder framework: the *semantic structure* of trait
dependencies is largely shared (nomothetic `D̄`); individuals differ in how much
each node matters to their self-concept. Centrality is then
``personalised_pagerank(D̄, teleport = node_weight_i)`` — the network is shared,
the weighting is idiographic.

Two independent-ish weight sources, both reconstructed from the rater cache so no
new model calls are needed:

* **evidence density** — the share of the account's extracted quotes that bear on
  each trait (how much of their writing is *about* that trait).
* **self-relevance / extremity** — ``|self_rating − midpoint|`` from the account's
  own E1 baseline ratings (how strongly, either way, the trait defines them).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from ..traits import TraitVocab


def _laplace(counts: np.ndarray) -> np.ndarray:
    x = counts.astype(float) + 1.0
    return x / x.sum()


def evidence_density(cache_path: str | Path, account: str, vocab: TraitVocab) -> np.ndarray:
    con = sqlite3.connect(str(cache_path))
    rows = con.execute(
        "SELECT response FROM calls WHERE task='evidence_chunk' AND account_hash=?",
        (account,),
    ).fetchall()
    counts = np.zeros(vocab.k)
    for (resp,) in rows:
        try:
            d = json.loads(resp)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(d, dict):
            continue
        for i, name in enumerate(vocab.names):
            q = d.get(name)
            if isinstance(q, list):
                counts[i] += len(q)
    return _laplace(counts)


def self_relevance(cache_path: str | Path, account: str, vocab: TraitVocab) -> np.ndarray:
    con = sqlite3.connect(str(cache_path))
    row = con.execute(
        "SELECT response FROM calls WHERE task='e1_baseline' AND account_hash=? LIMIT 1",
        (account,),
    ).fetchone()
    mid = (vocab.scale_min + vocab.scale_max) / 2
    span = (vocab.scale_max - vocab.scale_min) / 2
    if not row:
        return np.ones(vocab.k) / vocab.k
    try:
        d = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        return np.ones(vocab.k) / vocab.k
    ext = np.array(
        [abs(float(d.get(n, mid)) - mid) / span if isinstance(d.get(n), (int, float)) else 0.0
         for n in vocab.names]
    )
    return _laplace(ext * 100)  # scale so laplace smoothing is gentle


def e3_salience(cache_path: str | Path, account: str, vocab: TraitVocab) -> np.ndarray:
    """Independent node-salience estimate from the E3 chunk ratings.

    For each trait: mean over chunks of ``|score − midpoint|`` — how strongly and
    consistently the person's writing signals that trait, either direction. The
    E3 chunk rater never sees the evidence brief, so agreement between this and
    the brief-derived weights corroborates the node-weighting signal across
    estimators that don't share a failure mode.
    """
    con = sqlite3.connect(str(cache_path))
    rows = con.execute(
        "SELECT response FROM calls WHERE task='e3_chunk' AND trait LIKE ?",
        (f"{account}:%",),
    ).fetchall()
    mid = (vocab.scale_min + vocab.scale_max) / 2
    acc = np.zeros(vocab.k)
    n = 0
    for (resp,) in rows:
        try:
            d = json.loads(resp)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(d, dict):
            continue
        n += 1
        for i, name in enumerate(vocab.names):
            v = d.get(name)
            if isinstance(v, (int, float)):
                acc[i] += abs(float(v) - mid)
    if n == 0:
        return np.ones(vocab.k) / vocab.k
    return _laplace(acc / n)


def node_weights(
    cache_path: str | Path, account: str, vocab: TraitVocab, kind: str = "density"
) -> np.ndarray:
    if kind == "density":
        return evidence_density(cache_path, account, vocab)
    if kind == "self_relevance":
        return self_relevance(cache_path, account, vocab)
    if kind == "e3_salience":
        return e3_salience(cache_path, account, vocab)
    if kind == "combined":
        a = evidence_density(cache_path, account, vocab)
        b = self_relevance(cache_path, account, vocab)
        w = a * b
        return w / w.sum()
    raise ValueError(f"unknown node-weight kind {kind!r}")
