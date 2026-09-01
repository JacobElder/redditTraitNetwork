"""Milestone 1.2 — nomothetic / idiographic variance partition.

Decomposes the variance in dependency-edge weights `d[i][j]` across accounts and
directed trait pairs. Answers: **how much of the structure is shared (nomothetic)
vs. person-specific (idiographic)?**

Method of moments on the per-cell means and replicate SDs already stored in the
network artifacts (one network per account, R replicates per cell):

    weight[pair, account, rep] = grand
                               + a[account]           ~ N(0, σ²_account)
                               + p[pair]              ~ N(0, σ²_pair)      <- nomothetic
                               + ap[account, pair]    ~ N(0, σ²_account:pair)  <- idiographic pattern
                               + ε[pair,account,rep]  ~ N(0, σ²_rep)

* σ²_rep       — mean of the cell replicate variances (weight_sd²).
* σ²_pair      — the nomothetic structure: pairs differ in typical dependency.
* σ²_account   — additive person effect (rates everything higher/lower).
* σ²_account:pair — the interesting one: person has a *distinctive pattern* of
  which dependencies they weight. This is the idiographic signal.

Two-way ANOVA-without-replication EMS on the table of cell means, with σ²_rep
subtracted from the interaction term.

    ICC_idiographic = (σ²_account + σ²_account:pair)
                      / (σ²_account + σ²_pair + σ²_account:pair)

Bootstrap CI by resampling accounts. Not a pass/fail gate — the number selects
the Milestone 2 analysis branch (docs/PLAN.md §7.2, §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class VarianceComponents:
    sigma2_rep: float
    sigma2_account: float
    sigma2_pair: float
    sigma2_account_pair: float
    icc_idiographic: float
    icc_account_only: float
    n_accounts: int
    n_pairs: int
    replicates: int
    ci: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "sigma2_rep": self.sigma2_rep,
            "sigma2_account": self.sigma2_account,
            "sigma2_pair_nomothetic": self.sigma2_pair,
            "sigma2_account_pair_idiographic": self.sigma2_account_pair,
            "icc_idiographic": self.icc_idiographic,
            "icc_account_only": self.icc_account_only,
            "n_accounts": self.n_accounts,
            "n_pairs": self.n_pairs,
            "replicates": self.replicates,
            **{f"ci_{k}": v for k, v in self.ci.items()},
        }


def _pivot(long_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list, list]:
    accts = sorted(long_df["account_hash"].unique())
    pairs = sorted(long_df["trait_pair"].unique())
    ai = {a: i for i, a in enumerate(accts)}
    pj = {p: j for j, p in enumerate(pairs)}
    m = np.full((len(accts), len(pairs)), np.nan)
    v = np.full((len(accts), len(pairs)), np.nan)
    for row in long_df.itertuples(index=False):
        m[ai[row.account_hash], pj[row.trait_pair]] = row.weight
        v[ai[row.account_hash], pj[row.trait_pair]] = (
            row.weight_sd**2 if np.isfinite(row.weight_sd) else np.nan
        )
    return m, v, accts, pairs


def _decompose(m: np.ndarray, cell_var: np.ndarray, replicates: int) -> dict:
    """Two-way ANOVA without replication on the R×C table of cell means `m`."""
    n_a, n_p = m.shape
    ok = np.isfinite(m)
    grand = np.nanmean(m)
    row_mean = np.nanmean(m, axis=1, keepdims=True)
    col_mean = np.nanmean(m, axis=0, keepdims=True)

    ss_a = np.nansum((row_mean - grand) ** 2 * ok.sum(axis=1, keepdims=True))
    ss_p = np.nansum((col_mean - grand) ** 2 * ok.sum(axis=0, keepdims=True))
    resid = m - row_mean - col_mean + grand
    ss_e = np.nansum(resid[ok] ** 2)

    df_a, df_p = n_a - 1, n_p - 1
    df_e = max(df_a * df_p, 1)
    ms_a, ms_p, ms_e = ss_a / max(df_a, 1), ss_p / max(df_p, 1), ss_e / df_e

    sigma2_rep = float(np.nanmean(cell_var)) if np.isfinite(cell_var).any() else 0.0
    # interaction MS still carries mean-level replicate noise σ²_rep/R
    sigma2_ap = max(ms_e - sigma2_rep / max(replicates, 1), 0.0)
    sigma2_a = max((ms_a - ms_e) / n_p, 0.0)
    sigma2_p = max((ms_p - ms_e) / n_a, 0.0)

    denom_idio = sigma2_a + sigma2_p + sigma2_ap
    icc_idio = (sigma2_a + sigma2_ap) / denom_idio if denom_idio > 0 else np.nan
    icc_acct = sigma2_a / denom_idio if denom_idio > 0 else np.nan
    return dict(
        sigma2_rep=sigma2_rep, sigma2_account=sigma2_a, sigma2_pair=sigma2_p,
        sigma2_account_pair=sigma2_ap, icc_idiographic=icc_idio, icc_account_only=icc_acct,
    )


def fit_variance_partition(
    long_df: pd.DataFrame, replicates: int = 3, n_boot: int = 500, seed: int = 0
) -> VarianceComponents:
    m, cell_var, accts, pairs = _pivot(long_df)
    base = _decompose(m, cell_var, replicates)

    rng = np.random.default_rng(seed)
    boot = {k: [] for k in ("icc_idiographic", "icc_account_only", "sigma2_pair", "sigma2_account_pair")}
    for _ in range(n_boot):
        idx = rng.integers(0, m.shape[0], m.shape[0])
        d = _decompose(m[idx], cell_var[idx], replicates)
        for k, lst in boot.items():
            lst.append(d[k])
    ci = {
        k: (round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4))
        for k, v in boot.items()
    }

    return VarianceComponents(
        sigma2_rep=round(base["sigma2_rep"], 4),
        sigma2_account=round(base["sigma2_account"], 4),
        sigma2_pair=round(base["sigma2_pair"], 4),
        sigma2_account_pair=round(base["sigma2_account_pair"], 4),
        icc_idiographic=round(base["icc_idiographic"], 4),
        icc_account_only=round(base["icc_account_only"], 4),
        n_accounts=len(accts),
        n_pairs=len(pairs),
        replicates=replicates,
        ci=ci,
    )
