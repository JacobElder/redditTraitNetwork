"""Shared helpers for the estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..rater import Rater
from ..traits import TraitVocab


def _coerce_rating(v: Any, lo: float, hi: float) -> float:
    if v is None:
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return float(min(hi, max(lo, f)))


def elicit_trait_vector(
    rater: Rater,
    prompt: str,
    vocab: TraitVocab,
    call_parts_base: dict[str, Any],
    n_replicates: int,
) -> tuple[dict[str, float], dict[str, float], int]:
    """Run ``prompt`` ``n_replicates`` times, parse a {trait: rating} object each
    time, return (mean, sd, n) over replicates. Missing / unparseable cells are
    NaN and ignored in the mean.
    """
    mat = np.full((n_replicates, vocab.k), np.nan)
    for r in range(n_replicates):
        parts = {**call_parts_base, "replicate": r}
        resp = rater.complete(prompt, call_parts=parts, expect_json=True)
        data = resp.data if isinstance(resp.data, dict) else {}
        for idx, name in enumerate(vocab.names):
            mat[r, idx] = _coerce_rating(
                data.get(name), vocab.scale_min, vocab.scale_max
            )
    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    return (
        {n: float(mean[i]) for i, n in enumerate(vocab.names)},
        {n: float(sd[i]) for i, n in enumerate(vocab.names)},
        n_replicates,
    )


def elicit_scalar(
    rater: Rater,
    prompt: str,
    call_parts_base: dict[str, Any],
    n_replicates: int,
    lo: float,
    hi: float,
) -> tuple[float, float, int]:
    """Run ``prompt`` ``n_replicates`` times, parse ``{"rating": x}`` each time."""
    vals = []
    for r in range(n_replicates):
        parts = {**call_parts_base, "replicate": r}
        resp = rater.complete(prompt, call_parts=parts, expect_json=True)
        data = resp.data if isinstance(resp.data, dict) else {}
        v = _coerce_rating(data.get("rating"), lo, hi)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)
