# STATUS.md — what is actually built

Update this at the end of every working session. `docs/PLAN.md` §10 is the
checklist this mirrors.

## Session log

### 2026-09-01 (cont. 3) — first real LLM run (free tier, small)
- Gemini key in `secrets/gemini_key` (file fallback), verified working.
- **Free-tier reality**: `gemini-2.5-flash` free = ~5 rpm + ~250/day → weeks for
  a full run. `gemini-3.5-flash-lite` free = ~100 rpm in bursts → usable. User
  chose the small-run-then-scale path.
- `config/milestone1_2_free.yaml`: gemini-3.5-flash-lite, replicates 1, generic
  prior deferred. Running `build_networks --limit 8` now.
- Fixes from the first (aborted) run against real Gemini:
  - **per-chunk evidence brief** (`build_brief strategy='per_chunk'`): one
    110k-token prompt gave flash-lite ~16 quotes total; per-chunk gives ~250+.
  - **per-call JSON mode**: the E1 ablation *rewrite* (free text) was being
    forced into `{"brief": ...}` JSON. Fixed in gemini + openai_compatible.
  - rating prompt lists all 40 keys + requires each (was getting 14-key partials).
  - max_tokens 2048 → 3072.
- Scale-up: raise `--limit`, then flip `e2.fit_generic_prior` on, re-run
  `milestone1_2`. Consider enabling billing later for the full 48 + confirmatory.



### 2026-09-01 (cont. 2) — Milestone 1.2 wiring + free-model backends + real ingest running
- **Salt generated** → `secrets/salt` (gitignored). Real study salt, keep it safe.
- **Ingest running** (`scripts/ingest_accounts`, background): 60 candidates seeded
  from r/changemyview, AmItheAsshole, CasualConversation, AskReddit, self
  (commenters 2023-06 … 2025-06), full histories via Arctic Shift, capped at
  6000 items newest-first. ~65% pass the ≥300c / ≥2y / ≥5-sub filter → expect
  ~35-40 eligible in `data/accounts/index.parquet`.
- **Free/local rater backends added** so the estimation step needn't cost money:
  `rater/gemini_backend.py` (AI Studio free tier), `rater/openai_compatible.py`
  (Ollama local / Groq / OpenRouter free / vLLM). Wired in `build_rater`
  (`backend: gemini | openai_compatible | ollama | openai`). `config/rater_free.yaml`
  is a template. Cache key now includes the model id.
- **`scripts/build_networks.py`** — batch E1/E2/E3 over the eligible index,
  resumable, builds `d_generic` once.
- **Milestone 1.2 analysis built**: `estimate/nomothetic.py` (D̄ + Bᵢ),
  `validate/variance.py` (MoM variance partition → σ²_pair / σ²_account /
  σ²_account:pair / σ²_rep, ICC_idiographic + bootstrap CI),
  `validate/report.write_variance_partition`, `scripts/milestone1_2.py`.
  `tests/test_variance.py` recovers known components.
- **Chunking**: `sampled_window` mode — exactly `target_chunks` period-snapshot
  chunks for prolific accounts (was splitting into 200+); ~5× fewer E3 calls.
  NOTE: the running ingest used the pre-fix chunker; re-run `ingest_accounts`
  (no `--overwrite`) after it finishes to re-chunk from cached items.
- 21 tests pass, ruff clean. Branch `feat/milestone1-pipeline` pushed (7 commits).

**Ingest complete: 48/60 eligible** in `data/accounts/index.parquet`. Median
5083 comments, 2237-day (6yr) span, 180 subreddits, 45 chunks/account (re-chunked
with the sampled-window chunker). Full estimation chain verified end-to-end on
these corpora with the mock backend.

**BLOCKED ON: a model key for the estimation step.** No API keys / Ollama in
this environment. Add a key, then two commands:
```
export GEMINI_API_KEY=...        # free: aistudio.google.com/apikey
                                 # (or ANTHROPIC_API_KEY for a faster paid run)

python -m scripts.build_networks --config config/default.yaml \
    --override config/milestone1_2.yaml --override config/rater_free.yaml

python -m scripts.milestone1_2 --config config/default.yaml   # auto-detects the build
```
`milestone1_2` auto-detects the config hash from the artifacts on disk, so it
does NOT need the same `--override` chain. `build_networks` is resumable (skips
built accounts; the rater cache covers partial runs) — safe to Ctrl-C and rerun.

Rough size of the run: ~19k model calls (E1 ~8k, E3 ~6.5k, generic prior 4.7k
once). Gemini free tier: ~a day (rate limits). Paid Gemini Flash: ~$10-20, ~2h.
Sonnet: ~$80-120.



### 2026-09-01 (cont.) — real-data ingest + nomothetic framing
- **Nomothetic path added to the plan (docs/PLAN.md §2b).** Per the user: the
  Sloman/Love/Ahn and Elder conceptualization is substantially *nomothetic* — a
  shared trait-dependency map that individuals re-weight by how they
  self-describe. `Dᵢ = D̄ + Bᵢ`. We now estimate and report BOTH: `D̄` (pooled
  fixed effect / E2 generic / pooled E3) and `Bᵢ` (per-account deviation,
  modelled on self-description covariates). Milestone 1.2's `ICC_account` is
  reframed from a pass/fail gate to *the estimate of the idiographic variance
  share*, which selects which branch (nomothetic vs idiographic) the Milestone 2
  hypotheses run on. Only a pathological result (`D̄` ≈ generic prior AND
  `ICC_account` ≈ 0) still means stop.
- **`ingest/clients.py` is real now.** `ArcticShiftClient` uses the Arctic Shift
  HTTP API (`arctic-shift.photon-reddit.com/api`, no key), time-paginating
  `/comments/search` + `/posts/search` per author. Handles 429 backoff and
  skips 4xx "dense window" pages forward by a day. `ArcticShiftDumpClient`
  (DuckDB over `hf://datasets/open-index/arctic`) for sample-frame scans.
  `PrawClient` for `--gap-fill`.
- `ingest/pipeline.py` + `scripts/ingest_accounts.py`: username → hash (UserMap)
  → fetch → exclusion filter → inclusion check → chunk → `data/accounts/{hash}/`
  + `data/accounts/index.parquet` (hashes only). `--from-subreddit` seeds a
  candidate pool from a sub's commenters.
- `chunking.py`: added `target_chunks` (default 60) — prolific accounts produced
  ~2900 thread-chunks, unusable for per-chunk LLM rating / graphical VAR; now
  coarsens to ~60 time-ordered windows.
- **Tested on real Reddit accounts** (`spez`, `Poem_for_your_sprog`,
  `shitty_watercolour`): fetch + normalize + hash + exclusion + chunk all work.
  e.g. Poem_for_your_sprog → 4228c/10p, 25 subs, 5159d span, 0% excluded,
  338 chunks, eligible. Test artifacts (`data/`, `secrets/`, `cache/`) deleted.
- Tests: +`test_ingest_pipeline.py` (fake client). 18 pass, ruff clean.
- Shared a visualization of the 1.1 synthetic-recovery results as an Artifact.



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
| `ingest/clients.py` (ArcticShift API + dump, PRAW) | **done**, tested on real accounts |
| `ingest/pipeline.py` + `scripts/ingest_accounts.py` | **done** + tests |
| `estimate/nomothetic.py` (D̄ pooled fixed effect, Bᵢ) | **not started** — next |
| `validate/variance.py` (ICC partition, D̄ centrality) | **not started** — next |
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

## Immediate next steps (for whoever resumes)

1. **Decide the real-account sample.** Give the resumer a username list, or a
   `--from-subreddit` seed + criteria. Set a real `RTN_HASH_SALT` (not the
   `dev-test-salt` used in throwaway testing).
2. `python -m scripts.ingest_accounts --users-file accounts.txt` for ~50–80
   candidates → keep the `eligible` ones from `data/accounts/index.parquet`.
3. Build `estimate/nomothetic.py`: N2 pooled crossed random-effects fit over all
   accounts' E1 long tables → `D̄` (fixed effects) + `Bᵢ` (per-account); N3
   pooled E3. Then `validate/variance.py`: `ICC_account` + bootstrap CI,
   `D̄` centrality, `corr(D̄, D_generic)`, N1/N2/N3 convergence. Run 1.2.
4. `reports/milestone1.md` §1.2 — report the partition; note which branch
   (nomothetic / idiographic) the data selects.
5. In parallel (cheap, offline): fix the `ebicglasso` gap; scale the synthetic
   grid (`config/default.yaml`) for final 1.1 numbers; tiny real-API run to
   check the mock's realism (needs `ANTHROPIC_API_KEY`).
6. Decide the E2 pair-count question before the first real network build
   (§ Known gaps 3) — it sets the API bill.

## Open questions for the user

- Confirm `claude-sonnet-5` is the intended model id (`config/default.yaml`).
- Username-hash salt: expected in env `RTN_HASH_SALT`. OK, or prefer a file?
- Arctic Shift access route: HF `hf://datasets/...` via DuckDB httpfs, or local
  monthly dumps?
- E2 pair-count: full 1560, subsample, or drop to undirected-only?
