"""Deterministic, offline rater.

Two modes:

* **Oracle mode** (synthetic recovery): constructed with an ``Oracle`` that
  knows the ground-truth dependency matrix and per-chunk latent trait levels.
  The mock answers each task from that structure (plus seeded measurement
  noise), so the pipeline has a signal to recover and tests are exact.
* **Free mode** (no oracle): returns seeded hash-based pseudo-ratings. Enough to
  exercise code paths and keep unit tests fast; carries no real signal.

The ``Oracle`` protocol lives here to keep ``rater`` free of a ``validate``
import. ``validate/synthetic.py`` provides the concrete implementation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from ..traits import TraitVocab
from .base import Rater


class Oracle(Protocol):
    vocab: TraitVocab

    def e1_baseline(self, replicate: int) -> dict[str, float]: ...
    def e1_ablated(self, ablated_trait: str, replicate: int) -> dict[str, float]: ...
    def e2_pair(self, i: str, j: str, replicate: int) -> float: ...
    def e2_generic(self, i: str, j: str, replicate: int) -> float: ...
    def e3_chunk(self, chunk_id: str, replicate: int) -> dict[str, float]: ...
    def evidence_brief(self) -> dict[str, list[dict[str, Any]]]: ...


def _seeded_unit(*parts: Any) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class MockRater(Rater):
    name = "mock"

    def __init__(
        self,
        *,
        vocab: TraitVocab,
        oracle: Oracle | None = None,
        seed: int = 7,
        **kw: Any,
    ):
        super().__init__(**kw)
        self.vocab = vocab
        self.oracle = oracle
        self.seed = seed

    @property
    def model_id(self) -> str:
        return f"mock:{'oracle' if self.oracle else 'free'}:{self.seed}"

    def _generate(
        self, prompt: str, call_parts: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        task = call_parts.get("task")
        trait = call_parts.get("trait")
        rep = int(call_parts.get("replicate") or 0)
        usage = {"mock": True}

        if task == "evidence_brief":
            payload = (
                self.oracle.evidence_brief()
                if self.oracle
                else {t: [] for t in self.vocab.names}
            )
            return json.dumps(payload), usage

        if task == "e1_ablate_rewrite":
            # The ablated brief text is not consumed by the oracle; a marker is
            # enough. Real backends return a genuinely rewritten brief here.
            return f"[ablated brief: evidence for {trait} removed and reversed]", usage

        if task in ("e1_baseline", "e1_ablate_elicit", "e3_chunk"):
            ratings = self._trait_vector(task, trait, rep)
            return json.dumps(ratings), usage

        if task in ("e2_pair", "e2_generic"):
            i, j = str(trait).split("->")
            if self.oracle:
                val = (
                    self.oracle.e2_pair(i, j, rep)
                    if task == "e2_pair"
                    else self.oracle.e2_generic(i, j, rep)
                )
            else:
                val = 100.0 * _seeded_unit(self.seed, task, i, j, rep)
            return json.dumps({"rating": int(round(val))}), usage

        raise ValueError(f"MockRater: unknown task {task!r}")

    def _trait_vector(
        self, task: str, trait: str | None, rep: int
    ) -> dict[str, int]:
        if self.oracle:
            if task == "e1_baseline":
                raw = self.oracle.e1_baseline(rep)
            elif task == "e1_ablate_elicit":
                raw = self.oracle.e1_ablated(str(trait), rep)
            else:  # e3_chunk ; trait carries the chunk_id
                raw = self.oracle.e3_chunk(str(trait), rep)
        else:
            raw = {
                t: 100.0 * _seeded_unit(self.seed, task, trait, t, rep)
                for t in self.vocab.names
            }
        return {t: int(round(max(0.0, min(100.0, v)))) for t, v in raw.items()}
