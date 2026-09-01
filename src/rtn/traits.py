"""Load the fixed trait vocabulary (``traits/traits.yaml``).

The trait *order* returned by ``TraitVocab.names`` is the canonical index order
for every ``d`` matrix and centrality vector in the project. It follows file
order, which is grouped by Big Five domain and pole.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

Valence = Literal["pos", "neg"]


@dataclass(frozen=True)
class BigFive:
    domain: Literal["O", "C", "E", "A", "N"]
    sign: Literal["+", "-"]


@dataclass(frozen=True)
class TraitVocab:
    version: str
    names: tuple[str, ...]
    valence: dict[str, Valence]
    big_five: dict[str, BigFive]
    likableness: dict[str, float]
    likableness_approx: dict[str, bool]
    antonym: dict[str, str]
    scale_min: float
    scale_max: float

    def __post_init__(self) -> None:
        if len(self.names) != len(set(self.names)):
            raise ValueError("duplicate trait names in vocabulary")
        n_pos = sum(v == "pos" for v in self.valence.values())
        n_neg = sum(v == "neg" for v in self.valence.values())
        if n_pos != n_neg:
            raise ValueError(f"vocabulary not valence-balanced: {n_pos} pos / {n_neg} neg")

    @property
    def k(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)


def load_traits(path: str | Path) -> TraitVocab:
    doc = yaml.safe_load(Path(path).read_text())
    scale = doc.get("scale", {"min": 0, "max": 100})
    names: list[str] = []
    valence: dict[str, Valence] = {}
    big_five: dict[str, BigFive] = {}
    likableness: dict[str, float] = {}
    likableness_approx: dict[str, bool] = {}
    antonym: dict[str, str] = {}

    for entry in doc["traits"]:
        name = entry["name"]
        names.append(name)
        valence[name] = entry["valence"]
        bf = entry["big_five"]
        big_five[name] = BigFive(domain=bf["domain"], sign=bf["sign"])
        lk = entry["likableness"]
        likableness[name] = float(lk["value"])
        likableness_approx[name] = bool(lk.get("approx", False))
        antonym[name] = entry["antonym"]

    # antonym pairs must be mutual and present
    for a, b in antonym.items():
        if b not in antonym:
            raise ValueError(f"antonym {b!r} of {a!r} missing from vocabulary")
        if antonym[b] != a:
            raise ValueError(f"antonym pair not mutual: {a!r} <-> {b!r}")

    return TraitVocab(
        version=doc["version"],
        names=tuple(names),
        valence=valence,
        big_five=big_five,
        likableness=likableness,
        likableness_approx=likableness_approx,
        antonym=antonym,
        scale_min=float(scale["min"]),
        scale_max=float(scale["max"]),
    )
