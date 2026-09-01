"""End-to-end: the pipeline recovers known structure from oracle-backed
synthetic data. Small grid, mock rater only."""

import numpy as np
from scipy.stats import pearsonr

from rtn.config import load_config
from rtn.estimate import build_brief, estimate_e1
from rtn.prompts import get_prompts
from rtn.rater import build_rater
from rtn.traits import load_traits
from rtn.validate.synthetic import (
    SyntheticOracle,
    make_generic_prior,
    make_ground_truth,
    simulate_history,
)


def _offdiag(m):
    return m[~np.eye(m.shape[0], dtype=bool)]


def test_e1_recovers_ground_truth_on_thick_low_noise():
    cfg = load_config("config/default.yaml", "config/synthetic.yaml")
    v = load_traits(cfg.trait_vocab_path)
    p = get_prompts("p1")
    dg = make_generic_prior(v, 1)
    gt = make_ground_truth(v, dg, seed=42)
    hist = simulate_history(gt, "thick", 0.0, seed=7, account_hash="acct")
    rater = build_rater(cfg, vocab=v, oracle=SyntheticOracle(hist, seed=7))

    brief = build_brief("acct", hist.chunk_texts, v, p, rater, config_hash=cfg.hash8)
    e1 = estimate_e1(brief, v, p, rater, n_replicates=5, config_hash=cfg.hash8)

    r, _ = pearsonr(_offdiag(e1.d), _offdiag(gt.d_true))
    assert r > 0.8  # low-noise, thick evidence -> strong recovery


def test_run_recovery_smoke(tmp_path):
    from rtn.validate import run_recovery, summarize

    ov = tmp_path / "tiny.yaml"
    ov.write_text(
        "model:\n  backend: mock\ncache:\n  enabled: false\n"
        "estimate:\n  replicates: 2\n"
        "validate:\n  synthetic:\n    n_ground_truth_dags: 2\n"
        "    evidence_volume: [thin, thick]\n    noise_levels: [0.0]\n"
        "    replicates_per_cell: 1\n"
    )
    cfg = load_config("config/default.yaml", str(ov))
    df = run_recovery(cfg)
    assert len(df) == 2 * 2 * 1 * 1
    s = summarize(df)
    # thick evidence should recover E1 edges better than thin
    thick = s[s.volume == "thick"]["edge_r_e1"].mean()
    thin = s[s.volume == "thin"]["edge_r_e1"].mean()
    assert thick >= thin
