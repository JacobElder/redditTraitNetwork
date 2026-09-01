# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read this first

`docs/PLAN.md` is the authoritative, self-contained spec for the whole project —
architecture, module contracts, data schemas, build order, and the Milestone 1
gate. If anything here and there disagree, `docs/PLAN.md` wins. `docs/STATUS.md`
tracks what is actually built vs. stubbed. Start every session by reading both.

## What this project is

A pipeline that estimates a **weighted directed trait-dependency network** from a
Reddit account's public history: nodes are traits from a fixed 40-trait
vocabulary, edge `d[i][j]` is the degree to which trait *j* depends on trait *i*
(Sloman, Love & Ahn, 1998). It then computes conceptual centrality over that
network. It is a computational reimagining of the pairwise-dependency-rating
paradigm in Elder, Cheung, Davis & Hughes (*JPSP* 2023), where there is no
participant to ask, so the dependency structure must be estimated from text.

**This is a measurement-validation project first.** The failure mode that
Milestone 1 exists to detect: the LLM imposes a generic folk theory of how
traits relate, producing roughly the same network for every account. If that is
what is happening the pipeline is an expensive constant.

## Hard rules (do not violate without the user saying so explicitly)

- **Build Milestone 1 in order.** Show the user `reports/milestone1.md` before
  writing any `analysis/` (Milestone 2) code.
- **Nomothetic AND idiographic (see `docs/PLAN.md` §2b).** The trait-dependency
  structure is partly shared (nomothetic `D̄`) and partly person-specific
  (idiographic `Bᵢ`, where `Dᵢ = D̄ + Bᵢ`). Both are estimated and reported.
  "Every account gets ~the same network" is **not automatically a failure** — it
  may be the correct nomothetic answer. Milestone 1.2 *estimates* the
  idiographic share of variance (`ICC_account`); it is not a pass/fail gate.
  Only a pathological result (`D̄` ≈ generic prior **and** `ICC_account` ≈ 0)
  means stop and redesign.
- **Do not tune any threshold to pass.** Report the variance partition plainly.
- **Statistical-inference framing, not predictive framing.** Effect estimates
  with uncertainty. No "accuracy" scores.
- **Ethics enforced in code, not prose:**
  - Usernames are hashed at ingest (`ingest/hashing.py`). Analysis code must
    never receive a raw username. The raw→hash map lives in one gitignored file
    under `secrets/`.
  - The subreddit exclusion filter (`config/excluded_subreddits.yaml`) is
    applied at ingest. Ingest hard-fails if ≥ `max_excluded_fraction` of an
    account's items are in excluded subreddits.
  - Aggregate reporting only. No per-account profile with an identifier in any
    output, figure, or report.
- **No notebooks in the repo.** Scripts under `scripts/`, output under `reports/`.
- **Everything is config-driven and cached.** Every network artifact records the
  config hash and prompt version. Model calls are cached on
  `(account_hash, prompt_version, task, trait, replicate, config_hash)` — reruns
  cost nothing.

## Repo layout

```
config/        default.yaml, excluded_subreddits.yaml
traits/        traits.yaml (fixed 40-trait vocab) + provenance README
src/rtn/
  config.py        load + hash resolved config
  traits.py        load trait vocab
  prompts/         versioned prompt templates (p1, ...)
  rater/           Rater interface, anthropic + mock backends, sqlite cache
  ingest/          hashing, exclusion filter, chunking, arctic_shift + praw clients
  estimate/        e1_ablation, e2_pairwise, e3_covariation, evidence brief
  centrality.py    SLA iteration, PageRank + personalized, strength, betweenness
  validate/        synthetic recovery, variance decomposition, reliability, nulls
  analysis/        h1..h3  (MILESTONE 2 — do not build until gate passes)
scripts/       thin CLI entrypoints
reports/       milestone1.md and generated figures (figures/ gitignored)
tests/
```

## Commands

```bash
pip install -e ".[dev]"                         # install
pytest                                          # all tests
pytest tests/test_centrality.py -k pagerank     # single test
ruff check src tests                            # lint
ruff format src tests                           # format

# Milestone 1.1 — synthetic recovery (no Reddit access, mock rater by default)
python -m scripts.run_synthetic_recovery --config config/default.yaml \
    --override config/synthetic.yaml

# Ingest real accounts (Arctic Shift HTTP API, no credentials). RTN_HASH_SALT required.
RTN_HASH_SALT=... python -m scripts.ingest_accounts --users alice,bob --config config/default.yaml
RTN_HASH_SALT=... python -m scripts.ingest_accounts --from-subreddit AskHistorians --since 2023-01-01 --sample 60

# Build one network from a cached account corpus
python -m scripts.build_network --account <account_hash> --config config/default.yaml
```

Set `ANTHROPIC_API_KEY` for the `anthropic` backend. Use `model.backend: mock`
(in an override file) for tests and dry runs — it needs no key and is
deterministic.

## Key design facts that are easy to get wrong

- **`d[i][j]` = "j depends on i"**, i.e. row = source/independent trait,
  column = dependent trait. Centrality of a node is driven by its **row**
  (out-) influence. Keep this convention everywhere; `centrality.py` assumes it.
- **E1 ablation operates on the account's own quoted evidence**, never on an
  abstract trait label. "Remove the passages showing this person is competitive
  and replace them with evidence they are not" — not "imagine a version of this
  person who isn't competitive."
- **E2 also fits `d_generic`** (trait pairs only, no account conditioning). The
  reported E1 signal is `d_E1 − d_generic`; how much survives that subtraction
  is a headline result.
- **E3 estimates covariation, not conceptual dependency.** The LLM is only a
  chunk rater in E3; it never states a dependency. E1↔E3 convergence above a
  permutation baseline is the strongest evidence the pipeline measures something
  real. Say "convergence is evidence; identity is not claimed" in the writeup.
- **SLA centrality** iterates on dependency **magnitudes** `|d|` with a lazy
  `(1−δ)c` damping term (δ=0.85). It still converges to the dominant eigenvector
  of `|d|`, so SLA and eigenvector centrality should rank-correlate ~1.0; if not,
  there is a bug. The damping only matters for near-acyclic `d` (a strict DAG is
  nilpotent and would otherwise collapse SLA to a constant).

## Data access (Milestone 1.2 onward)

- **Arctic Shift** (Pushshift successor): monthly Parquet dumps on Hugging Face,
  queried with DuckDB without full download. Primary route for building the
  sample frame under the inclusion criteria.
- **PRAW** (official Reddit API): freshness / gap-filling only. ~1000-item
  listing cap, no date-range search.
- PullPush: unreliable secondary mirror.
- Cache everything to Parquet under `data/` (gitignored). Never re-fetch during
  analysis. Deterministic account IDs.
- Reddit's terms restrict automated and commercial access. Non-commercial
  research only; keep it documented.
