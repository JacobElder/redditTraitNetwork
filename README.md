# redditTraitNetwork

Estimating a **trait-dependency network** from public Reddit histories and
computing conceptual centrality over it.

Nodes are personality traits (fixed 40-trait IPIP vocabulary). An edge `i → j`
means **trait *j* depends on trait *i*** in the sense of Sloman, Love & Ahn
(1998) — a feature is central to a concept to the extent that other features
depend on it.

The reference paradigm is Elder, Cheung, Davis & Hughes (*JPSP*, 2023): a
**consensus** directed dependency network built from separate participants
free-nominating *"which traits does [TARGET] depend upon?"* (edge kept if ≥ 25 %
endorse) — a **nomothetic** network. Individual differences enter later, when
people rate themselves on the traits.

Here there is no participant to nominate dependencies, so the structure is
**estimated from each account's text** (see `docs/METHOD.md`), then combined
across people into a shared network **`D̄`** — the analog of the consensus
matrix. The idiographic layer, per the framework, is **node *weighting*** (how
much each trait matters to a given person), not per-person edge structures.

**This is a measurement-validation project first.**

- **[`docs/METHOD.md`](docs/METHOD.md)** — how the pipeline turns Reddit text
  into a network, and what every symbol/number in the report means. **Start here.**
- [`docs/PLAN.md`](docs/PLAN.md) — full spec, build order, the Milestone 1 checks.
- [`docs/STATUS.md`](docs/STATUS.md) — current progress and results.
- [`CLAUDE.md`](CLAUDE.md) — agent guidance.

## Status (2026-09-07)

| | |
|---|---|
| Pipeline | ingest → chunk → evidence brief → E1/E2/E3 → `D̄` / `Bᵢ` / node weighting → centrality → variance partition — **all built**, tested |
| Milestone 1.1 (synthetic recovery) | passes: E1 edge recovery *r* ≈ 0.72–0.86, E1↔E3 directed convergence ρ ≈ 0.75 (thick evidence) |
| Milestone 1.2 (real accounts) | **11 networks built** (free-tier model), §1.2 run |
| — shared network `D̄` | **validated** — E1 (ablation) and E3 (covariation) converge, *r* ≈ 0.66 and rising with sample size |
| — per-account edge deviations `Bᵢ` | not corroborated by E3 (expected — the framework doesn't predict them) |
| — per-account node weighting | not yet corroborated (weak signal at n = 11 / free model / homogeneous sample) |
| Milestone 1.3–1.5, Milestone 2 | not started |

The idiographic layer is not measurable yet with the current sample/model.
`ICC_idiographic` is reported with a noise bracket, not used as a pass/fail gate.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q

# Milestone 1.1 — synthetic recovery (offline, mock rater, no API key)
python -m scripts.run_synthetic_recovery \
    --config config/default.yaml --override config/synthetic.yaml

# Real accounts (needs RTN_HASH_SALT + a model key — see docs/STATUS.md)
RTN_HASH_SALT=... python -m scripts.ingest_accounts \
    --config config/default.yaml --override config/milestone1_2.yaml \
    --from-subreddit changemyview,AskHistorians,personalfinance --since 2023-01-01 --sample 40
python -m scripts.build_networks --config config/default.yaml \
    --override config/milestone1_2.yaml --override config/milestone1_2_free.yaml
python -m scripts.milestone1_2 --config config/default.yaml   # -> reports/milestone1.md §1.2
```

## The three estimators

| | what it does | role |
|---|---|---|
| **E1** counterfactual ablation | remove & reverse a trait's evidence in the account's own brief, re-rate all traits; `d[i][j] = rating_j(full) − rating_j(ablate_i)` | primary |
| **E2** pairwise elicitation | ask a pairwise "how much would *j* change" magnitude question in persona; also fit a no-persona `d_generic` (folk theory) | baseline / prior (off for the free-tier run) |
| **E3** behavioural covariation | rate each history chunk on all traits; estimate structure (Ledoit-Wolf partial correlations, graphical VAR) from the chunk × trait matrix — the model never states a dependency | independent convergent evidence |

`D̄` = mean of the per-account E1 matrices. Centrality: **outdegree** (Elder et
al.'s headline), SLA iteration (Sloman/Love/Ahn), PageRank + personalised
PageRank (teleport = per-account node weights), strength, betweenness.
