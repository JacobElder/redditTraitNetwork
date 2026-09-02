"""E1 — counterfactual ablation ("Sloman probe"). PRIMARY estimator.

    d[i][j] = rating_j(full_brief) - rating_j(ablate_i_brief)

Ablation edits the account's OWN evidence (remove + reverse the support for
trait i), never an abstract label. Signed weights. Diagonal 0.
"""

from __future__ import annotations

import numpy as np

from ..network import Network
from ..prompts.base import Prompts
from ..rater import Rater
from ..traits import TraitVocab
from .common import elicit_trait_vector
from .evidence import EvidenceBrief


def estimate_e1(
    brief: EvidenceBrief,
    vocab: TraitVocab,
    prompts: Prompts,
    rater: Rater,
    *,
    n_replicates: int,
    ablation_strength: str = "strong",
    config_hash: str | None = None,
) -> Network:
    account = brief.account_hash
    full_text = brief.render(vocab)

    base_parts = dict(
        account_hash=account,
        prompt_version=prompts.version,
        config_hash=config_hash,
    )

    # baseline ratings on the full brief
    full_mean, _, _ = elicit_trait_vector(
        rater,
        prompts.e1_elicit(full_text, vocab),
        vocab,
        {**base_parts, "task": "e1_baseline", "trait": None},
        n_replicates,
    )
    full_vec = np.array([full_mean[n] for n in vocab.names])

    d = np.zeros((vocab.k, vocab.k))
    sd = np.zeros((vocab.k, vocab.k))

    for i, src in enumerate(vocab.names):
        # 1. rewrite the brief with trait `src` ablated
        rewrite_prompt = prompts.e1_ablate(
            full_text, src, vocab.antonym[src], ablation_strength
        )
        rw = rater.complete(
            rewrite_prompt,
            call_parts={
                **base_parts,
                "task": "e1_ablate_rewrite",
                "trait": src,
                "replicate": 0,
            },
            expect_json=False,
        )
        ablated_text = rw.text

        # 2. re-elicit all trait ratings on the ablated brief
        abl_mean, abl_sd, _ = elicit_trait_vector(
            rater,
            prompts.e1_elicit(ablated_text, vocab),
            vocab,
            {**base_parts, "task": "e1_ablate_elicit", "trait": src},
            n_replicates,
        )
        abl_vec = np.array([abl_mean[n] for n in vocab.names])
        d[i, :] = full_vec - abl_vec
        sd[i, :] = np.array([abl_sd[n] for n in vocab.names])

    # a rare elicitation drops a trait key -> NaN cell; treat as no measured
    # ablation effect (0). Count is tracked in meta for the report.
    n_missing = int(np.isnan(d).sum())
    d = np.nan_to_num(d, nan=0.0)
    sd = np.nan_to_num(sd, nan=0.0)
    np.fill_diagonal(d, 0.0)
    return Network(
        estimator="e1",
        traits=vocab.names,
        d=d,
        account_hash=account,
        config_hash=config_hash,
        prompt_version=prompts.version,
        weight_sd=sd,
        n_replicates=n_replicates,
        meta={"ablation_strength": ablation_strength, "n_missing_cells": n_missing},
    )
