"""Evidence-brief construction (input to E1 and, optionally, E2).

The brief is the object E1 ablation edits: extracted verbatim quotes from the
account's own chunks, per trait, tagged for/against. Also yields an evidence-
density vector (per-trait share of the account's evidence) used as the
personalised-PageRank teleport vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..prompts.base import Prompts
from ..rater import Rater
from ..traits import TraitVocab


@dataclass
class Quote:
    text: str
    chunk: int
    direction: str  # "for" | "against"


@dataclass
class EvidenceBrief:
    account_hash: str
    quotes: dict[str, list[Quote]]
    density: dict[str, float]  # sums to 1 over traits
    raw: dict[str, Any] = field(default_factory=dict)

    def render(self, vocab: TraitVocab) -> str:
        lines = ["EVIDENCE BRIEF", ""]
        for name in vocab.names:
            qs = self.quotes.get(name, [])
            if not qs:
                lines.append(f"## {name}: (no evidence found)")
                continue
            lines.append(f"## {name}")
            for q in qs:
                lines.append(f'- [{q.direction}] "{q.text}" (chunk {q.chunk})')
            lines.append("")
        return "\n".join(lines)

    def density_vector(self, vocab: TraitVocab) -> np.ndarray:
        return np.array([self.density.get(n, 0.0) for n in vocab.names])


def _parse_quotes(data: dict, vocab: TraitVocab) -> dict[str, list[Quote]]:
    out: dict[str, list[Quote]] = {}
    for name in vocab.names:
        raw_list = data.get(name) or []
        quotes = []
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            txt = str(item.get("quote", "")).strip()
            if not txt:
                continue
            quotes.append(
                Quote(
                    text=txt[:240],
                    chunk=int(item.get("chunk", -1)) if str(item.get("chunk", "")).lstrip("-").isdigit() else -1,
                    direction="against" if item.get("direction") == "against" else "for",
                )
            )
        out[name] = quotes
    return out


def _density(quotes: dict[str, list[Quote]], vocab: TraitVocab) -> dict[str, float]:
    counts = np.array([len(quotes.get(n, [])) for n in vocab.names], dtype=float)
    total = counts.sum()
    if total <= 0:
        return {n: 1.0 / vocab.k for n in vocab.names}
    # Laplace smoothing so no trait gets zero teleport mass
    smoothed = counts + 1.0
    smoothed /= smoothed.sum()
    return {n: float(smoothed[i]) for i, n in enumerate(vocab.names)}


def build_brief(
    account_hash: str,
    chunk_texts: list[str],
    vocab: TraitVocab,
    prompts: Prompts,
    rater: Rater,
    *,
    quotes_per_trait: int = 8,
    strategy: str = "per_chunk",
    max_chunks: int | None = None,
    config_hash: str | None = None,
) -> EvidenceBrief:
    """``per_chunk`` (default): extract trait evidence from each chunk separately,
    then merge and keep the top ``quotes_per_trait`` per trait. Robust to model
    context limits and far more thorough than one giant prompt. ``single``: the
    original one-shot extraction over the whole corpus.

    ``max_chunks`` (per_chunk only): use an evenly-spaced sample of that many
    chunks for the brief, to cap model calls. E3 still uses the full chunk set.
    """
    base = {
        "account_hash": account_hash,
        "prompt_version": prompts.version,
        "config_hash": config_hash,
    }
    if strategy != "single" and max_chunks and len(chunk_texts) > max_chunks:
        import numpy as np

        idx = np.linspace(0, len(chunk_texts) - 1, max_chunks, dtype=int)
        chunk_texts = [chunk_texts[i] for i in sorted(set(idx.tolist()))]
    if strategy == "single":
        prompt = prompts.evidence_brief_rendered(chunk_texts, vocab, quotes_per_trait)
        resp = rater.complete(
            prompt,
            call_parts={**base, "task": "evidence_brief", "trait": None, "replicate": 0},
            expect_json=True,
        )
        data = resp.data if isinstance(resp.data, dict) else {}
        quotes = _parse_quotes(data, vocab)
    else:
        merged: dict[str, list[Quote]] = {n: [] for n in vocab.names}
        raw: dict[str, Any] = {}
        for ci, text in enumerate(chunk_texts):
            resp = rater.complete(
                prompts.evidence_brief_rendered([text], vocab, quotes_per_trait),
                call_parts={**base, "task": "evidence_chunk", "trait": f"c{ci:05d}", "replicate": 0},
                expect_json=True,
            )
            d = resp.data if isinstance(resp.data, dict) else {}
            for name, qs in _parse_quotes(d, vocab).items():
                for q in qs:
                    merged[name].append(Quote(text=q.text, chunk=ci, direction=q.direction))
            raw[f"c{ci}"] = d
        quotes = {
            n: _dedupe_rank(merged[n], quotes_per_trait) for n in vocab.names
        }
        data = raw

    return EvidenceBrief(
        account_hash=account_hash,
        quotes=quotes,
        density=_density(quotes, vocab),
        raw=data,
    )


def _dedupe_rank(quotes: list[Quote], keep: int) -> list[Quote]:
    seen: set[str] = set()
    out: list[Quote] = []
    # prefer longer (more informative) quotes, spread across chunks
    for q in sorted(quotes, key=lambda x: -len(x.text)):
        norm = " ".join(q.text.lower().split())[:120]
        if norm in seen:
            continue
        seen.add(norm)
        out.append(q)
        if len(out) >= keep * 2:
            break
    out.sort(key=lambda x: x.chunk)
    return out[: keep] if keep else out
