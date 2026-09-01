"""``Network``: a trait-dependency matrix plus provenance.

Convention (see docs/PLAN.md): ``d[i, j]`` is the degree to which trait ``j``
(column) depends on trait ``i`` (row). Diagonal is forced to 0. Weights may be
signed (E1) or non-negative (E2/E3 magnitudes) depending on the estimator.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Network:
    estimator: str  # e1 | e2 | e2_generic | e3_undirected | e3_directed | e3_contemp | synthetic
    traits: tuple[str, ...]
    d: np.ndarray  # (k, k) float; d[i, j] = "j depends on i"
    account_hash: str | None = None
    config_hash: str | None = None
    prompt_version: str | None = None
    weight_sd: np.ndarray | None = None  # (k, k) replicate SD, optional
    n_replicates: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        k = len(self.traits)
        self.d = np.asarray(self.d, dtype=float).reshape(k, k)
        np.fill_diagonal(self.d, 0.0)
        if self.weight_sd is not None:
            self.weight_sd = np.asarray(self.weight_sd, dtype=float).reshape(k, k)

    @property
    def k(self) -> int:
        return len(self.traits)

    def to_long_df(self) -> pd.DataFrame:
        rows = []
        for i, src in enumerate(self.traits):
            for j, dep in enumerate(self.traits):
                if i == j:
                    continue
                rows.append(
                    {
                        "estimator": self.estimator,
                        "account_hash": self.account_hash,
                        "source_trait": src,
                        "dep_trait": dep,
                        "trait_pair": f"{src}->{dep}",
                        "weight": float(self.d[i, j]),
                        "weight_sd": (
                            float(self.weight_sd[i, j])
                            if self.weight_sd is not None
                            else np.nan
                        ),
                        "n_replicates": self.n_replicates,
                        "config_hash": self.config_hash,
                        "prompt_version": self.prompt_version,
                    }
                )
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_long_df().to_parquet(path, index=False)
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "estimator": self.estimator,
                    "traits": list(self.traits),
                    "account_hash": self.account_hash,
                    "config_hash": self.config_hash,
                    "prompt_version": self.prompt_version,
                    "n_replicates": self.n_replicates,
                    "meta": self.meta,
                    "created_utc": int(time.time()),
                },
                indent=2,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Network:
        path = Path(path)
        df = pd.read_parquet(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        traits = tuple(meta["traits"])
        idx = {t: n for n, t in enumerate(traits)}
        d = np.zeros((len(traits), len(traits)))
        sd = np.full((len(traits), len(traits)), np.nan)
        for _, r in df.iterrows():
            i, j = idx[r["source_trait"]], idx[r["dep_trait"]]
            d[i, j] = r["weight"]
            sd[i, j] = r["weight_sd"]
        return cls(
            estimator=meta["estimator"],
            traits=traits,
            d=d,
            account_hash=meta.get("account_hash"),
            config_hash=meta.get("config_hash"),
            prompt_version=meta.get("prompt_version"),
            weight_sd=sd,
            n_replicates=meta.get("n_replicates"),
            meta=meta.get("meta", {}),
        )

    @staticmethod
    def artifact_name(estimator: str, config_hash8: str, prompt_version: str) -> str:
        return f"{estimator}__{config_hash8}__{prompt_version}.parquet"
