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

## Worked example

One (illustrative, not real) account. Its history is cut into ~45 time-window
chunks; here is one:

> *(r/changemyview)* I came in sure I was right, but two replies actually moved
> me — I'll give people that.
> *(r/hiking)* Happy to lend an axe or a bag to anyone doing the traverse, just DM.
> *(r/personalfinance)* You're overthinking it. Automate it and stop looking.

**1. Evidence brief.** A model pulls verbatim quotes bearing on each of the 40
traits (the `evidence_brief` prompt in `prompts/p1.py`), most ending up empty:

```json
{
  "openminded": [{"quote": "two replies actually moved me", "direction": "for"}],
  "generous":   [{"quote": "Happy to lend an axe or a bag ... just DM", "direction": "for"}],
  "warm":       [{"quote": "I'll give people that", "direction": "for"}],
  "disciplined":[{"quote": "Automate it and stop looking", "direction": "for"}]
}
```

**2. E1 — ablate one trait.** An assessor reads the whole brief and rates all 40
traits (`e1_elicit`): say `warm 70, kind 64, forgiving 58, considerate 55`. Then
the brief is rewritten with *warm*'s evidence removed **and reversed, in the same
voice** (`e1_ablate`) — not "imagine a colder person":

> *(r/changemyview)* Two replies "moved me"? People rarely argue in good faith —
> not worth engaging.
> *(r/hiking)* Not lending gear. People don't bring it back.

Re-rate the ablated brief: `warm 18, kind 47, forgiving 46, considerate 41`.

```
d[warm → kind]        = 64 − 47 = +17
d[warm → forgiving]   = 58 − 46 = +12
d[warm → considerate] = 55 − 41 = +14
```

Repeat for all 40 traits (× 5 replicates) → this account's 40×40 matrix `Dᵢ`.

**3. E2 — pairwise, in persona** (`e2_pair`; off for the free-tier run):

> "You are the person described below. If you were no longer **forgiving**, how
> much would that change how **resentful** you are? 0 = not at all, 100 =
> completely." → `52`

`e2_generic` asks the same with no persona ("If a person were no longer
forgiving…") — the folk-theory baseline `d_generic` that E1's headline signal is
measured against.

**4. E3 — covariation.** The model only rates each chunk on all 40 traits
(`e3_chunk`), never stating a dependency:

```
chunk 07:  openminded 68  generous 60  warm 62  disciplined 38  ...
chunk 22:  openminded 41  generous 33  warm 44  disciplined 66  ...
```

Ledoit-Wolf partial correlations across the 45 chunk rows → an undirected
network; lag-1 graphical VAR → a directed one. If either agrees with `Dᵢ` above a
permutation baseline, the structure is not just an E1 rewriting artefact.

**5. Combine.** `D̄` = mean of every account's `Dᵢ`. Centrality of a trait = how
much the rest of the network depends on it. On the current 11-account `D̄` the
highest-centrality nodes are *forgiving, passive, openminded, kind* (SLA; see
`reports/milestone1.md` §1.2).
