# redditTraitNetwork

Estimating **idiographic trait-dependency networks** from public Reddit
histories, and computing conceptual centrality over them.

Nodes are personality traits (fixed 40-trait vocabulary). Edge `d[i][j]` is the
degree to which trait *j* depends on trait *i* — "if this person were no longer
*i*, how much would that change how *j* they are?" (Sloman, Love & Ahn, 1998).
This is a computational reimagining of the pairwise-dependency-rating paradigm in
Elder, Cheung, Davis & Hughes (*JPSP*, 2023): there is no participant to ask, so
the dependency structure is estimated from text.

**This is a measurement-validation project first.** See
[`docs/PLAN.md`](docs/PLAN.md) for the full spec and
[`docs/STATUS.md`](docs/STATUS.md) for current progress. Agent guidance is in
[`CLAUDE.md`](CLAUDE.md).

## Quick start

```bash
pip install -e ".[dev]"
pytest -q

# Milestone 1.1 — synthetic recovery (offline, mock rater)
python -m scripts.run_synthetic_recovery \
    --config config/default.yaml --override config/synthetic.yaml
# -> reports/milestone1.md  (section "1.1 Synthetic recovery")
```

## Three dependency estimators

| | what it does | role |
|---|---|---|
| **E1** ablation | remove & reverse a trait's evidence in the account's own brief, re-rate all traits; `d = full − ablated` | primary |
| **E2** pairwise | ask the original dependency item in persona for every trait pair; also fit a no-account `d_generic` | baseline / generic prior |
| **E3** covariation | rate each history chunk on all traits, estimate structure (EBICglasso / graphical VAR) from the chunk × trait matrix | independent convergent evidence |

## Status

Milestone 1.1 pipeline is built and runs end-to-end on synthetic data. Reddit
ingest (Arctic Shift / PRAW) is stubbed pending Milestone 1.2. Nothing in
Milestone 2 is built — it is gated on `ICC_account ≥ .10` from Milestone 1.2.
