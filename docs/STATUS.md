# STATUS.md — what is actually built

Update this at the end of every working session. `docs/PLAN.md` §10 is the
checklist this mirrors.

## Session log

### 2026-08-31 / 09-01 — scaffold + Milestone 1.1 pipeline (first pass)
Built the entire estimator + centrality + synthetic-validation pipeline against
a deterministic mock/oracle rater. `pytest -q` → 16 passed. `ruff check` clean.
`python -m scripts.run_synthetic_recovery --config config/default.yaml
--override config/synthetic.yaml` runs end-to-end (~2–3 min, 90 cells) and
writes `reports/milestone1.md` §1.1 + `reports/figures/`.

Files created:
`config/{default,synthetic,excluded_subreddits}.yaml`,
`traits/{traits.yaml,README.md}`, `CLAUDE.md`, `docs/{PLAN,STATUS}.md`,
`src/rtn/{__init__,config,traits,network,centrality}.py`,
`src/rtn/prompts/{__init__,base,p1}.py`,
`src/rtn/rater/{__init__,base,cache,mock_backend,anthropic_backend}.py`,
`src/rtn/ingest/{__init__,hashing,exclusion,chunking,clients}.py`,
`src/rtn/estimate/{__init__,common,evidence,e1_ablation,e2_pairwise,e3_covariation}.py`,
`src/rtn/validate/{__init__,synthetic,recovery,report}.py`,
`scripts/{run_synthetic_recovery,build_network}.py`,
`tests/test_{config_traits,centrality,ingest,rater_cache,hashing,synthetic_pipeline}.py`.

Decisions locked:
- LLM backend: Anthropic API (`claude-sonnet-5`), config-driven; deterministic
  `MockRater` + `SyntheticOracle` for tests and Milestone 1.1.
- Trait vocab: curated valence-balanced 40-trait list (IPIP + Anderson 1968).
  OSF list NOT retrieved. Only `honest`/`dishonest` likableness exact.
- Ground truth is a **near-DAG** (≈12% back-edges), not a strict DAG — a strict
  DAG is nilpotent and makes SLA/eigenvector centrality degenerate.
- SLA centrality operates on `|d|` with a lazy `(1−δ)c` damping term (δ=0.85);
  converges to the dominant eigenvector of `|d|`, robust on near-acyclic `d`.
- `ingest/clients.py` (Arctic Shift, PRAW) intentionally stubbed until 1.2.

## Component status

| Component | State |
|---|---|
| scaffold / config / traits / `network.py` | **done** + tests |
| `prompts/p1.py` | **done** (paraphrases p1a/b/c for 1.4 — TODO) |
| `rater/` base+cache+mock | **done** + tests |
| `rater/anthropic_backend.py` | **written, never run against the real API** |
| `ingest/` hashing+exclusion+chunking | **done** + tests |
| `ingest/clients.py` (arctic_shift, praw) | **stub** (`NotImplementedError`) — 1.2 |
| `centrality.py` | **done** (SLA≈eigenvector asserted) |
| `estimate/` evidence,e1,e2,e3 | **done** + tests; **E3 undirected weak — see gaps** |
| `validate/synthetic,recovery,report` | **done** |
| `scripts/run_synthetic_recovery.py` | **done** |
| `scripts/build_network.py` | **skeleton** (needs cached corpus from 1.2 ingest) |
| `reports/milestone1.md` §1.1 | **generated, first pass** — analyst TODOs unfilled |
| Milestone 1.2–1.5 | **not started** (blocked on real ingest) |
| Milestone 2 (`analysis/`) | **not started** (blocked on ICC gate) |

## First-pass synthetic recovery numbers (mock rater, 90 cells)

| | thin | medium | thick |
|---|---|---|---|
| E1 edge recovery `r` (noise 0) | ~0.49 | ~0.72 | ~0.85 |
| E1 centrality rank ρ — out_strength | ~0.58 | ~0.87 | ~0.95 |
| E1 centrality rank ρ — sla/eigenvector | ~0.52 | ~0.77 | ~0.86 |
| E1 − generic edge recovery | ~0.35 | ~0.44 | ~0.47 |
| E2 edge recovery (vs \|true\|) | ~0.15 | ~0.20 | ~0.22 |
| E3 directed edge recovery | n/a¹ | ~0.52 | ~0.81 |
| E3 undirected edge recovery | ~0 | ~0 | ~0 |
| E1↔E3 convergence — SLA centrality ρ | n/a¹ | ~0.40 | ~0.75 |
| E1↔E3 convergence — edge r | n/a¹ | ~0.35 | ~0.70 |

¹ thin has < `min_chunks` (20) chunks so E3 directed is not estimated.

Reads: E1 recovers edge weights and (out-strength / eigenvector) centrality well
at medium+ evidence, degrading with noise and thin corpora, as intended. E2 is
dominated by the folk prior by construction. E1−generic keeps ~60% of E1's
recovery. **E1↔E3 directed converge strongly in simulation** (ρ ≈ 0.75 at thick)
— the two estimators without a shared failure mode agree, which is what the
design wants. E3 *undirected* still recovers ~0 (see gap 1).

## Known gaps / debt

1. **`estimate/e3_covariation.ebicglasso` over-sparsifies at p=40** — EBIC picks a
   near-empty graph; the "densest fit" fallback is a band-aid. E3 undirected
   recovery is ~0 as a result. Real fix: proper EBICglasso port or
   `GraphicalLassoCV`. This is a genuine methods gap, not only a synthetic one.
2. ~~E1↔E3 centrality convergence negative in synthetic~~ — FIXED. Was a harness
   artefact: `_var_transition` row-normalised the VAR matrix, flattening the
   out-strength signal. Now scales by spectral radius; `e1_e3_sla_rho` ≈ +0.75
   at thick evidence.
3. **E2 = 1560 ordered pairs × replicates per account** — the cost driver for real
   runs. Decide: subsample pairs, or drop E2 to undirected-only, before 1.2.
4. **`traits.yaml` likableness** values are placeholders except honest/dishonest.
5. `anthropic_backend` unexercised — first real call may surface schema/format
   issues in the prompts (JSON adherence, brief-rewrite length).

## Immediate next steps (for whoever resumes — Gemini or otherwise)

1. Read `reports/milestone1.md` §1.1. Fill the "Read / decisions" TODOs.
2. Scale the grid: raise `validate.synthetic.*` in `config/default.yaml`
   (12 DAGs, 3 noise levels, 10 reps) and re-run for final 1.1 numbers.
3. Fix gap (1): swap `ebicglasso` for a real implementation; re-run; confirm E3
   undirected + E1↔E3 convergence recover in synthetic.
4. Tiny real-API sanity run: `model.backend: anthropic`, 2 DAGs, thick only,
   compare recovery curve shape to the mock. Needs `ANTHROPIC_API_KEY`.
5. Milestone 1.2: implement `ingest/clients.py` (Arctic Shift DuckDB sample
   frame + fetch, PRAW gap-fill), wire `scripts/build_network.py` to write
   `data/networks/`, pull ~50 accounts, write `validate/variance.py`
   (crossed random effects → `ICC_account`) → **run the ICC gate**.

## Open questions for the user

- Confirm `claude-sonnet-5` is the intended model id (`config/default.yaml`).
- Username-hash salt: expected in env `RTN_HASH_SALT`. OK, or prefer a file?
- Arctic Shift access route: HF `hf://datasets/...` via DuckDB httpfs, or local
  monthly dumps?
- E2 pair-count: full 1560, subsample, or drop to undirected-only?
