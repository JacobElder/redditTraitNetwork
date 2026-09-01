"""Synthetic ground truth + history simulation for Milestone 1.1.

We define ground-truth dependency DAGs, generate synthetic Reddit-style
histories consistent with them (crossing evidence volume x noise), and provide a
:class:`SyntheticOracle` that answers every rater task from that structure plus
seeded measurement noise. Running the real pipeline against the oracle tells us
how well E1/E2/E3 recover known structure, and lets us tune k, chunk count, and
replicate count before spending on real accounts.

Measurement model (kept deliberately simple and transparent):

* ``trait_levels`` : the account's true standing on each trait, 0-100.
* E1 baseline rating of j  ~  trait_levels[j] + noise
* E1 ablate-i rating of j  ~  trait_levels[j] - d_true[i, j] + noise
      => full - ablate recovers d_true[i, j]
* E2 pair (i->j)           ~  w*d_generic[i, j] + (1-w)*|d_true[i, j]| + noise
      (w = generic_contamination; this is the folk-theory-leakage knob)
* E2 generic (i->j)        ~  d_generic[i, j] + noise
* E3 chunk t, trait j      ~  trait_levels[j] + latent_z[t, j]*scale + noise
      where latent_z follows a lag-1 VAR whose transition matrix is derived
      from d_true, so across-chunk covariation carries the structure.

Noise scale is inflated for thin evidence, so recovery degrades with volume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..traits import TraitVocab

_VOLUME_NCHUNKS = {"thin": 14, "medium": 36, "thick": 72}
_VOLUME_NOISE_MULT = {"thin": 3.0, "medium": 1.6, "thick": 1.0}
_BASE_NOISE = 4.0  # rating points, sd
_GENERIC_CONTAMINATION = 0.7  # how much E2 leans on the folk prior


# --------------------------------------------------------------------------
# Ground-truth construction
# --------------------------------------------------------------------------
def make_generic_prior(vocab: TraitVocab, seed: int) -> np.ndarray:
    """A single folk-theory dependency matrix, shared across all synthetic
    accounts: antonyms depend strongly on each other, same-domain traits
    moderately, plus light noise. Non-negative, scaled to ~0-60.
    """
    rng = np.random.default_rng(seed)
    k = vocab.k
    g = rng.uniform(2, 10, size=(k, k))
    for i, ti in enumerate(vocab.names):
        for j, tj in enumerate(vocab.names):
            if i == j:
                continue
            if vocab.antonym[ti] == tj:
                g[i, j] += rng.uniform(35, 55)
            elif vocab.big_five[ti].domain == vocab.big_five[tj].domain:
                g[i, j] += rng.uniform(10, 25)
    np.fill_diagonal(g, 0.0)
    return np.clip(g, 0, 100)


@dataclass
class GroundTruth:
    vocab: TraitVocab
    d_true: np.ndarray  # (k, k) signed DAG weights (rating points)
    trait_levels: np.ndarray  # (k,) 0-100
    d_generic: np.ndarray  # (k, k) shared folk prior
    topo_order: np.ndarray
    seed: int


def make_ground_truth(
    vocab: TraitVocab,
    d_generic: np.ndarray,
    seed: int,
    edge_density: float = 0.22,
    feedback_fraction: float = 0.12,
) -> GroundTruth:
    """Near-DAG dependency structure: mostly forward edges in a random topo
    order, plus a minority (``feedback_fraction``) of weaker back-edges so the
    graph contains cycles. A *strict* DAG is nilpotent, which makes
    eigenvector-family centrality undefined; real trait-dependency networks
    (Elder et al.) contain feedback loops, so the near-DAG is the faithful
    ground truth.
    """
    rng = np.random.default_rng(seed)
    k = vocab.k
    order = rng.permutation(k)
    pos = np.empty(k, dtype=int)
    pos[order] = np.arange(k)

    d = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            forward = pos[i] < pos[j]
            p = edge_density if forward else edge_density * feedback_fraction
            if rng.random() < p:
                mag = rng.uniform(6, 26) * (1.0 if forward else 0.5)
                sign = 1.0 if rng.random() < 0.75 else -1.0
                d[i, j] = sign * mag

    trait_levels = rng.uniform(15, 85, size=k)
    return GroundTruth(
        vocab=vocab,
        d_true=d,
        trait_levels=trait_levels,
        d_generic=d_generic,
        topo_order=order,
        seed=seed,
    )


# --------------------------------------------------------------------------
# History simulation
# --------------------------------------------------------------------------
@dataclass
class SyntheticHistory:
    gt: GroundTruth
    volume: str
    noise: float
    chunk_ids: list[str]
    chunk_texts: list[str]
    latent_z: np.ndarray  # (n_chunks, k)
    noise_sd: float
    account_hash: str
    meta: dict = field(default_factory=dict)


def _var_transition(d_true: np.ndarray) -> np.ndarray:
    """Stable lag-1 transition matrix from the signed DAG.

    ``A[i, j]`` = effect of trait i at t-1 on trait j at t, so E3's directed
    estimator (which also uses the i->j = "j depends on i" convention) targets
    something correlated with ``d_true``.
    """
    a = d_true.copy()
    # Scale globally (not per row) so the relative out-influence structure of
    # d_true is preserved — row-normalising would flatten exactly the
    # out-strength signal that centrality keys on. Target spectral radius ~0.7
    # for a stable VAR.
    eig_max = np.max(np.abs(np.linalg.eigvals(a)))
    if eig_max < 1e-9:
        eig_max = np.max(np.abs(a)) or 1.0
    return 0.7 * a / eig_max


def simulate_history(
    gt: GroundTruth,
    volume: str,
    noise: float,
    seed: int,
    account_hash: str,
) -> SyntheticHistory:
    rng = np.random.default_rng(seed)
    k = gt.vocab.k
    n_chunks = _VOLUME_NCHUNKS[volume]
    noise_sd = _BASE_NOISE * _VOLUME_NOISE_MULT[volume] * (1.0 + noise)

    a = _var_transition(gt.d_true)
    z = np.zeros((n_chunks, k))
    innov_sd = 4.0 * (1.0 + noise)
    for t in range(1, n_chunks):
        z[t] = z[t - 1] @ a + rng.normal(0, innov_sd, size=k)

    chunk_ids, chunk_texts = [], []
    for t in range(n_chunks):
        cid = f"{account_hash}:{t:05d}"
        chunk_ids.append(cid)
        # text is not consumed by the oracle; keep a compact deterministic stub
        chunk_texts.append(
            f"[synthetic chunk {t} vol={volume} noise={noise:g}] "
            + " ".join(rng.choice(list(gt.vocab.names), size=6))
        )
    return SyntheticHistory(
        gt=gt,
        volume=volume,
        noise=noise,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        latent_z=z,
        noise_sd=noise_sd,
        account_hash=account_hash,
        meta={"n_chunks": n_chunks, "innov_sd": innov_sd},
    )


# --------------------------------------------------------------------------
# Oracle
# --------------------------------------------------------------------------
class SyntheticOracle:
    """Answers every rater task from a :class:`SyntheticHistory`."""

    def __init__(self, hist: SyntheticHistory, seed: int = 0):
        self.hist = hist
        self.gt = hist.gt
        self.vocab = hist.gt.vocab
        self._rng_seed = seed
        self._levels = hist.gt.trait_levels
        self._z_scale = 3.0

    def _noise(self, *tag) -> np.ndarray:
        rng = np.random.default_rng((self._rng_seed, hash(tag) % (2**32)))
        return rng.normal(0, self.hist.noise_sd, size=self.vocab.k)

    def _vec(self, arr: np.ndarray) -> dict[str, float]:
        return {n: float(np.clip(arr[i], 0, 100)) for i, n in enumerate(self.vocab.names)}

    # -- Oracle protocol -------------------------------------------
    def e1_baseline(self, replicate: int) -> dict[str, float]:
        return self._vec(self._levels + self._noise("e1b", replicate))

    def e1_ablated(self, ablated_trait: str, replicate: int) -> dict[str, float]:
        i = self.vocab.index(ablated_trait)
        return self._vec(
            self._levels - self.gt.d_true[i, :] + self._noise("e1a", ablated_trait, replicate)
        )

    def e2_pair(self, i: str, j: str, replicate: int) -> float:
        ii, jj = self.vocab.index(i), self.vocab.index(j)
        val = (
            _GENERIC_CONTAMINATION * self.gt.d_generic[ii, jj]
            + (1 - _GENERIC_CONTAMINATION) * abs(self.gt.d_true[ii, jj])
        )
        rng = np.random.default_rng((self._rng_seed, hash(("e2p", i, j, replicate)) % (2**32)))
        return float(np.clip(val + rng.normal(0, self.hist.noise_sd), 0, 100))

    def e2_generic(self, i: str, j: str, replicate: int) -> float:
        ii, jj = self.vocab.index(i), self.vocab.index(j)
        rng = np.random.default_rng((self._rng_seed, hash(("e2g", i, j, replicate)) % (2**32)))
        return float(np.clip(self.gt.d_generic[ii, jj] + rng.normal(0, 3.0), 0, 100))

    def e3_chunk(self, chunk_id: str, replicate: int) -> dict[str, float]:
        t = self.hist.chunk_ids.index(chunk_id)
        base = self._levels + self._z_scale * self.hist.latent_z[t]
        return self._vec(base + self._noise("e3", chunk_id, replicate))

    def evidence_brief(self) -> dict[str, list[dict]]:
        rng = np.random.default_rng((self._rng_seed, 999))
        mult = _VOLUME_NOISE_MULT[self.hist.volume]
        out: dict[str, list[dict]] = {}
        for i, name in enumerate(self.vocab.names):
            salience = abs(self._levels[i] - 50) / 50.0
            n_q = int(round(salience * 6 / mult)) + rng.integers(0, 2)
            direction = "for" if self._levels[i] >= 50 else "against"
            out[name] = [
                {"quote": f"synthetic evidence {q} for {name}", "chunk": q, "direction": direction}
                for q in range(max(0, n_q))
            ]
        return out
