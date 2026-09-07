"""E2 — direct pairwise elicitation. BASELINE / generic-prior estimator.

Asks a pairwise dependency-magnitude question, in persona, for every ordered trait
pair. Also fits ``d_generic`` with no account conditioning at all; the reported
E1/E2 account-specific signal is ``d - d_generic``.

Non-negative weights (0-100 "how much would that change..."). Diagonal 0.
"""

from __future__ import annotations

import numpy as np

from ..network import Network
from ..prompts.base import Prompts
from ..rater import Rater
from ..traits import TraitVocab
from .common import elicit_scalar
from .evidence import EvidenceBrief


def _persona_block(brief: EvidenceBrief | None, vocab: TraitVocab, max_lines: int = 40) -> str:
    if brief is None:
        return "You are a person described only by your own past writing."
    lines = ["Things you have written or done (from your own history):"]
    n = 0
    for name in vocab.names:
        for q in brief.quotes.get(name, []):
            if q.direction == "for":
                lines.append(f'- "{q.text}"')
                n += 1
                break
        if n >= max_lines:
            break
    return "\n".join(lines)


def estimate_e2(
    brief: EvidenceBrief | None,
    vocab: TraitVocab,
    prompts: Prompts,
    rater: Rater,
    *,
    n_replicates: int,
    account_hash: str | None = None,
    config_hash: str | None = None,
) -> Network:
    persona = _persona_block(brief, vocab)
    d = np.zeros((vocab.k, vocab.k))
    sd = np.zeros((vocab.k, vocab.k))
    base = dict(
        account_hash=account_hash,
        prompt_version=prompts.version,
        task="e2_pair",
        config_hash=config_hash,
    )
    for i, ti in enumerate(vocab.names):
        for j, tj in enumerate(vocab.names):
            if i == j:
                continue
            mean, s, _ = elicit_scalar(
                rater,
                prompts.e2_pair(ti, tj, persona),
                {**base, "trait": f"{ti}->{tj}"},
                n_replicates,
                vocab.scale_min,
                vocab.scale_max,
            )
            d[i, j] = 0.0 if np.isnan(mean) else mean
            sd[i, j] = 0.0 if np.isnan(s) else s
    return Network(
        estimator="e2",
        traits=vocab.names,
        d=d,
        account_hash=account_hash,
        config_hash=config_hash,
        prompt_version=prompts.version,
        weight_sd=sd,
        n_replicates=n_replicates,
    )


def estimate_generic(
    vocab: TraitVocab,
    prompts: Prompts,
    rater: Rater,
    *,
    n_replicates: int,
    config_hash: str | None = None,
) -> Network:
    """Study-wide generic-prior network. No account conditioning; cached across
    the whole study (account_hash is None in the cache key).
    """
    d = np.zeros((vocab.k, vocab.k))
    sd = np.zeros((vocab.k, vocab.k))
    base = dict(
        account_hash=None,
        prompt_version=prompts.version,
        task="e2_generic",
        config_hash=config_hash,
    )
    for i, ti in enumerate(vocab.names):
        for j, tj in enumerate(vocab.names):
            if i == j:
                continue
            mean, s, _ = elicit_scalar(
                rater,
                prompts.e2_generic(ti, tj),
                {**base, "trait": f"{ti}->{tj}"},
                n_replicates,
                vocab.scale_min,
                vocab.scale_max,
            )
            d[i, j] = 0.0 if np.isnan(mean) else mean
            sd[i, j] = 0.0 if np.isnan(s) else s
    return Network(
        estimator="e2_generic",
        traits=vocab.names,
        d=d,
        account_hash=None,
        config_hash=config_hash,
        prompt_version=prompts.version,
        weight_sd=sd,
        n_replicates=n_replicates,
    )
