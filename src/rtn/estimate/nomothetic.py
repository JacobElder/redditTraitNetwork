"""Pool per-account networks into a nomothetic network `D̄` and per-account
idiographic deviations `Bᵢ` (docs/PLAN.md §2b).

    Dᵢ = D̄ + Bᵢ

`D̄` is the shared "how traits depend on each other" structure; `Bᵢ` is how
account `i` departs from it. Loads the network artifacts written by
`scripts/build_networks.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..network import Network
from ..traits import TraitVocab


def load_account_networks(
    estimator: str, config_hash8: str, prompt_version: str, root: Path | None = None
) -> dict[str, Network]:
    """All per-account artifacts for one estimator/config/prompt combination."""
    root = root or (REPO_ROOT / "data" / "networks")
    name = Network.artifact_name(estimator, config_hash8, prompt_version)
    out: dict[str, Network] = {}
    for path in sorted(root.glob(f"*/{name}")):
        acct = path.parent.name
        if acct.startswith("_"):
            continue
        out[acct] = Network.load(path)
    return out


def stack_long(networks: dict[str, Network]) -> pd.DataFrame:
    """One row per (account, directed trait pair): weight + replicate SD."""
    frames = []
    for acct, net in networks.items():
        df = net.to_long_df()
        df["account_hash"] = acct
        frames.append(df[["account_hash", "source_trait", "dep_trait", "trait_pair",
                          "weight", "weight_sd", "n_replicates"]])
    return pd.concat(frames, ignore_index=True)


def nomothetic_network(
    networks: dict[str, Network], vocab: TraitVocab, estimator: str = "e1_pooled"
) -> tuple[Network, dict[str, np.ndarray]]:
    """`D̄` = mean edge weight across accounts (grand mean of `Dᵢ`); `Bᵢ = Dᵢ − D̄`.

    A flat mean, not a shrinkage estimate — the partial-pooling upgrade
    (statsmodels / pymer crossed RE on replicate-level data) is noted in
    docs/PLAN.md 'Known gaps'. For balanced designs the flat mean and the
    fixed-effect estimate coincide.
    """
    accts = sorted(networks)
    traits = vocab.names
    stack = np.stack([networks[a].d for a in accts])  # (n_acct, k, k)
    d_bar = stack.mean(axis=0)
    np.fill_diagonal(d_bar, 0.0)
    b = {a: (networks[a].d - d_bar) for a in accts}
    dbar_net = Network(
        estimator=estimator, traits=traits, d=d_bar,
        n_replicates=len(accts), meta={"n_accounts": len(accts), "source": estimator},
    )
    return dbar_net, b


def deviation_long(
    b: dict[str, np.ndarray], vocab: TraitVocab, brief_density: dict[str, dict[str, float]] | None = None
) -> pd.DataFrame:
    """Long `Bᵢ` for modelling idiographic deviation on self-description
    covariates. If `brief_density` (account -> {trait: evidence density}) is
    given, attaches per-edge density of the source and dependent trait.
    """
    rows = []
    names = vocab.names
    for acct, mat in b.items():
        for i, si in enumerate(names):
            for j, dj in enumerate(names):
                if i == j:
                    continue
                r = {
                    "account_hash": acct,
                    "source_trait": si,
                    "dep_trait": dj,
                    "trait_pair": f"{si}->{dj}",
                    "deviation": float(mat[i, j]),
                    "abs_deviation": float(abs(mat[i, j])),
                }
                if brief_density and acct in brief_density:
                    dens = brief_density[acct]
                    r["density_source"] = dens.get(si, np.nan)
                    r["density_dep"] = dens.get(dj, np.nan)
                    r["density_min"] = min(dens.get(si, np.nan), dens.get(dj, np.nan))
                rows.append(r)
    return pd.DataFrame(rows)
