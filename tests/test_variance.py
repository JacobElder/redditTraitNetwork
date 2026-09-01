"""Variance partition recovers known nomothetic / idiographic components."""

import numpy as np
import pandas as pd

from rtn.validate.variance import fit_variance_partition


def _synthetic_long(n_acct, n_pairs, sd_pair, sd_acct, sd_pattern, sd_rep, seed=0):
    """Build a per-account long table with known variance components."""
    rng = np.random.default_rng(seed)
    pair_effect = rng.normal(0, sd_pair, n_pairs)
    acct_effect = rng.normal(0, sd_acct, n_acct)
    rows = []
    for a in range(n_acct):
        for p in range(n_pairs):
            pattern = rng.normal(0, sd_pattern)
            cell_mean = 10 + pair_effect[p] + acct_effect[a] + pattern
            rows.append(
                {
                    "account_hash": f"a{a:02d}",
                    "trait_pair": f"p{p:03d}",
                    "weight": cell_mean + rng.normal(0, sd_rep / np.sqrt(3)),
                    "weight_sd": sd_rep,
                    "n_replicates": 3,
                }
            )
    return pd.DataFrame(rows)


def test_recovers_mostly_nomothetic():
    df = _synthetic_long(30, 120, sd_pair=5.0, sd_acct=0.5, sd_pattern=0.5, sd_rep=2.0)
    vc = fit_variance_partition(df, replicates=3, n_boot=100)
    # pair variance dominates -> low idiographic ICC
    assert vc.sigma2_pair > vc.sigma2_account_pair
    assert vc.icc_idiographic < 0.35


def test_recovers_mostly_idiographic_pattern():
    df = _synthetic_long(30, 120, sd_pair=1.0, sd_acct=0.5, sd_pattern=4.0, sd_rep=1.5)
    vc = fit_variance_partition(df, replicates=3, n_boot=100)
    assert vc.sigma2_account_pair > vc.sigma2_pair
    assert vc.icc_idiographic > 0.6
    lo, hi = vc.ci["icc_idiographic"]
    assert 0.0 <= lo <= hi <= 1.0 + 1e-9
    assert hi - lo < 0.4  # a usefully tight interval with 30 accounts


def test_replicate_noise_excluded_from_icc_denominator():
    df = _synthetic_long(25, 100, sd_pair=3.0, sd_acct=0.5, sd_pattern=1.0, sd_rep=8.0)
    vc = fit_variance_partition(df, replicates=3, n_boot=50)
    # big elicitation noise should not inflate the idiographic share
    assert vc.sigma2_rep > vc.sigma2_pair
    assert 0.0 <= vc.icc_idiographic <= 1.0
