# PLAN.md — Idiographic trait dependency networks from Reddit histories

Authoritative, self-contained implementation spec. Written so a fresh session
(any model) can resume without the original handoff. If this disagrees with
`CLAUDE.md`, this wins. Progress is tracked separately in `docs/STATUS.md`.

---

## 0. One-paragraph statement of the problem

Given a Reddit account's public post/comment history, estimate a weighted
**directed** dependency network over a fixed vocabulary of ~40 personality
traits. Node = trait. Edge `d[i][j]` = the degree to which trait *j* depends on
trait *i*, in the sense of Sloman, Love & Ahn (1998): a feature is central to a
concept to the extent that other features depend on it. Then compute conceptual
centrality over that network and study its behavioural correlates in the Reddit
history. This reimagines the lab paradigm of Elder, Cheung, Davis & Hughes
(*JPSP* 2023), in which participants rated pairwise trait dependencies directly;
here there is no participant, so the dependency structure is estimated from text.
**That estimation problem is the project. Everything else is plumbing.**

### Edge orientation convention (used everywhere)

`d[i][j]`: **row `i` = the trait depended upon (source / independent);
column `j` = the dependent trait.** "If this person were no longer `i`, how much
would that change how `j` they are?" A trait with large outgoing row weights is
depended upon by many others → high centrality. `centrality.py`, the SLA
iteration, and PageRank all assume this. Diagonal is 0.

---

## 1. Non-negotiable framing

1. **Measurement validation first, substance second.** The failure mode to rule
   out: the pipeline is an expensive constant — it returns a number with no
   information beyond a fixed prior. Milestone 1 tests this *before* any
   substantive analysis exists.
2. **Nomothetic and idiographic are both real targets — see §2b.** Sloman, Love
   & Ahn (1998) and much of the Elder social-neuro work treat feature/trait
   centrality as a largely **shared** conceptual structure: people carry roughly
   the same map of how traits depend on each other, and differ in *which parts
   they weight* given how they self-describe. The Elder, Cheung, Davis & Hughes
   (*JPSP* 2023) paradigm then measures each participant's own weighted version.
   So "every account gets ~the same network" is **not automatically a failure** —
   it may be the correct nomothetic answer, and the question becomes how much
   idiographic deviation rides on top of it. The old hard gate
   (`ICC_account ≥ .10` or stop) is reframed accordingly in §7.2.
3. **Do not build Milestone 2 until Milestone 1 has run and been reviewed.**
4. **Do not tune any threshold to pass.** Report the variance partition plainly.
5. **Inference, not prediction.** Effect estimates with uncertainty, not
   accuracy/F1.
6. **Ethics is enforced in code** (hashing, exclusion filter, aggregate-only
   output), not asserted in a README.

---

## 2. The three dependency estimators

All three are implemented. They are the design, not alternatives to choose
between. Each produces a `d` matrix on the same trait vocabulary for a given
account, plus metadata (config hash, prompt version, per-cell replicate spread).

### E1 — Counterfactual ablation ("Sloman probe"). PRIMARY.

Pipeline:

1. **Evidence brief.** From the account's chunked corpus, extract short quoted
   passages that bear on each trait, with chunk IDs. Target
   `estimate.e1.evidence_quotes_per_trait` (default 8) per trait, fewer if the
   corpus does not support it. This brief is the object ablation operates on.
2. **Baseline elicitation.** With the full brief in the prompt, in persona,
   elicit a 0–100 rating for every trait `j`. Repeat `estimate.replicates`
   times. `rating_j(full)` = mean over replicates.
3. **Ablation passes.** For each trait `i`: build an **ablated brief** =
   full brief with trait `i`'s supporting quotes removed **and** replaced with
   constructed passages contradicting trait `i` (strength controlled by
   `estimate.e1.ablation_strength`). Re-elicit ratings for all `j`.
   `rating_j(ablate_i)` = mean over replicates.
4. `d[i][j] = rating_j(full) − rating_j(ablate_i)`, **signed**. `d[i][i] := 0`.

Cost: `O(k)` ablation passes × `k` traits per pass × `replicates` calls, plus
the baseline. `k = 40` → ~`41 × replicates` elicitation calls per account
(each call returns all 40 ratings) + brief construction.

**Critical:** ablation edits *the account's own evidence*, never an abstract
label. The ablated passages must read as things this specific person wrote/did
that are inconsistent with trait `i`. If you find yourself writing "imagine a
person who…", stop — that is a request for the model's generic prior.

### E2 — Direct pairwise elicitation. BASELINE / PRIOR.

- Ask the original dependency item **verbatim**, in persona, for ordered trait
  pairs `(i, j)`: "If you were no longer `i`, how much would that change how `j`
  you are? (0–100)". `replicates` calls per pair; `d_E2[i][j]` = mean.
- **Also fit `d_generic`**: identical elicitation with **no account
  conditioning at all** — trait pair only, no persona, no brief. This is the
  pure generic-prior network. One `d_generic` for the whole study (not
  per-account), `replicates × n_pairs` calls, heavily cached.
- E2 is the estimator most exposed to generic-prior contamination, which is why
  it is here. The account-specific signal is `d_E1 − d_generic` (and
  `d_E2 − d_generic`); **how much survives that subtraction is a headline
  result either way.**

### E3 — Behavioural covariation. INDEPENDENT CONVERGENT EVIDENCE.

- Chunk the history (thread- or window-level; time-ordered).
- **LLM is only a rater here** — score each chunk on all 40 traits (0–100, plus
  "not enough evidence" → NaN). `replicates` calls per chunk; cell = mean.
  Result: a `n_chunks × 40` matrix `X`.
- **Undirected structure:** EBICglasso (graphical lasso with EBIC model
  selection, `estimate.e3.ebic_gamma`) on `X` → partial-correlation network.
- **Directed structure:** graphical VAR (lag-1) over the time-ordered chunk
  matrix → temporal (directed) and contemporaneous networks. Requires
  `estimate.e3.min_chunks` (default 20) chunks.
- E3 does not share E1's failure mode (the model never states a dependency), so
  **E1↔E3 convergence above a permutation baseline is the strongest single
  piece of validation evidence.**
- Writeup must state: E3 estimates covariation, not conceptual dependency in
  Sloman's sense. Convergence is evidence; identity is not claimed.

---

## 2b. Nomothetic and idiographic networks — estimate both

The dependency structure has two levels, and the project reports both. They come
from the *same* per-account estimator outputs (E1/E2/E3), combined two ways.

### The nomothetic network `D̄`

The shared "how traits depend on each other, in general" map. Three converging
estimates, in increasing order of grounding:

- **N1 = E2 generic prior** (`estimate_generic`, already built). Direct pairwise
  elicitation, no persona, no corpus. Pure folk theory. One matrix for the study.
- **N2 = pooled fixed effect.** Stack every account's E1 (and separately E3)
  long-format edge table and fit, per directed pair `p`,
  `weight_p,account ~ N(μ_p, τ_p²)` with partial pooling across accounts
  (a crossed model: `weight ~ 0 + pair + (0 + pair | account)`; the fixed-effect
  vector is `D̄`, the per-account random deviations are the idiographic part).
  `μ_p` shrinks thin/noisy accounts toward the grand structure.
- **N3 = pooled E3.** Concatenate all accounts' chunk×trait matrices (account as
  a grouping factor / with account-mean-centering) and estimate one graphical
  VAR / EBICglasso — covariation structure pooled over people.

`D̄` has its own centrality vector; this is directly comparable to the shared
structure in Sloman/Love/Ahn and Elder et al. and is a result in its own right.

### The idiographic deviation `Bᵢ`

For account `i`: `Dᵢ = D̄ + Bᵢ`. `Bᵢ` is the account-level random-effect matrix
from N2 (or `Dᵢ_E1 − D̄`). The substantive claim the user is after: **`Bᵢ` is
structured by how the person self-describes** — an edge `i→j` deviates from the
nomothetic weight more when the account has more self-relevant evidence bearing
on `i` and `j` (evidence density from the brief), or more extreme standing on
them. Model `|Bᵢ,p|` (or `Bᵢ,p` signed) on per-account, per-trait self-description
covariates. Personalised PageRank (teleport = evidence density) is the
centrality-level version of the same idea.

### What the variance partition means

The crossed random-effects fit in §7.2 gives
`ICC_account = σ²_account / (σ²_account + σ²_pair + σ²_resid)` =
**the share of dependency-edge variance that is idiographic** rather than shared.

- Low `ICC_account` → the network is mostly nomothetic. The headline analyses run
  on `D̄`'s centrality, with person-level self-description weighting as the
  source of individual differences. Still a paper; still matches the theory.
- High `ICC_account` → idiographic centrality (`Dᵢ`) is a meaningful per-person
  DV, and H1–H3 (§8) run per account as originally framed.

Either way the number is *estimated and reported*, not used as a pass/fail gate.

---

## 3. Trait vocabulary

- `traits/traits.yaml`: 40 traits, 20 antonym pairs, valence-balanced (20 pos /
  20 neg), Big Five coverage on both poles of all five domains.
- **Fixed, not discovered** in v1 — shared nodes are required for cross-account
  centrality comparison; discovery confounds "trait absent" with "trait not
  mentioned".
- Provenance and the outstanding task (transcribe real Anderson 1968 likableness
  values; only `honest`/`dishonest` are currently exact) are in
  `traits/README.md`.
- Editing the vocab bumps `version:` and invalidates all caches (it is in the
  config hash). Do not edit mid-study.

---

## 4. Centrality (`src/rtn/centrality.py`)

All measures operate on the same `d` matrix (negative weights handled per
measure — document the choice in each docstring).

| Measure | Definition / notes |
|---|---|
| **SLA iteration** | `c <- normalize( δ·(\|d\|@c) + (1−δ)·c )`, L2 each step, δ=0.85. Operates on **magnitudes** `\|d\|` (central = others depend strongly on it, sign-agnostic). The lazy `(1−δ)c` term shifts eigenvalues without changing eigenvectors, so it still converges to the dominant eigenvector of `\|d\|` but does not collapse on near-acyclic `d`. Params `centrality.sla_max_iter`, `sla_tol`. |
| **Eigenvector** | dominant eigenvector of `\|d\|` directly (`scipy.linalg.eig`). **Sanity check: rank-correlate with SLA; ~1.0** (asserted in `tests/test_centrality.py`). |
| **PageRank** | on `\|d\|`, column-stochastic, damping `centrality.pagerank_alpha` (0.85). An edge i→j (j depends on i) is a vote from j to i. |
| **Personalised PageRank** | same, teleport vector = per-trait **evidence density** (fraction of the account's corpus, by chars or chunks, that bears on that trait). Distinguishes "structurally central" from "structurally central *and* constantly returned to". This is the interesting variant. |
| **Out-strength / In-strength** | row sum / column sum of `d` (signed and absolute variants). |
| **Betweenness** | on a distance graph derived from `d` (e.g. `1/|d|` for present edges). Comparison measure only. |

`centrality.py` returns a tidy DataFrame: one row per (trait, measure, value),
plus a helper that reports the full rank-correlation matrix among measures.

---

## 5. Data schemas (Parquet, under `data/`, all gitignored)

### `data/accounts/{account_hash}/items.parquet`
One row per post or comment.

| col | type | notes |
|---|---|---|
| `item_id` | str | Reddit fullname (`t1_...`/`t3_...`) |
| `account_hash` | str | sha256(username + salt)[:16]; **never the raw name** |
| `kind` | str | `comment` \| `post` |
| `subreddit` | str | lowercased |
| `created_utc` | int64 | epoch seconds |
| `body` | str | markdown as retrieved |
| `score` | int32 | net votes at fetch time |
| `parent_id` | str | for threads |
| `link_id` | str | submission fullname |
| `permalink` | str | for the feedback-signal join in H2 |
| `fetched_utc` | int64 | provenance |
| `source` | str | `arctic_shift` \| `praw` |

### `data/accounts/{account_hash}/chunks.parquet`
| col | type | notes |
|---|---|---|
| `chunk_id` | str | `{account_hash}:{k:05d}` |
| `account_hash` | str | |
| `strategy` | str | `thread` \| `window` |
| `item_ids` | list[str] | items composing the chunk |
| `t_start`, `t_end` | int64 | epoch bounds |
| `n_chars` | int32 | |
| `text` | str | assembled, ≤ `ingest.chunk.max_chars` |
| `epoch` | int8 | time-bin index for temporal split-half |

### Network artifact — `data/networks/{account_hash}/{estimator}__{config_hash8}__{prompt_version}.parquet`
Long format: `estimator, account_hash, source_trait, dep_trait, weight, weight_sd, n_replicates, config_hash, prompt_version, created_utc`.
Sidecar `.json` with the full resolved config and evidence-density vector.

### Ratings cache — `cache/model_calls.sqlite`
Table `calls(key TEXT PRIMARY KEY, account_hash, prompt_version, task, trait,
replicate, config_hash, request_json, response_json, model, created_utc)`.
`key = sha256` of the canonicalised `(account_hash, prompt_version, task, trait,
replicate, config_hash)` tuple. `task ∈ {e1_baseline, e1_ablate, e2_pair,
e2_generic, e3_chunk, evidence_brief}`.

---

## 6. Module contracts

Signatures are the intended API; adjust for ergonomics but keep the data
contracts (schemas above) stable.

### `src/rtn/config.py`
- `load_config(path, *overrides) -> Config` — deep-merges YAML overrides onto
  the base, resolves relative paths, reads and inlines `traits.yaml`.
- `Config.hash8 -> str` — first 8 hex of SHA-256 of the canonical JSON of the
  fully-resolved config (including the trait list and prompt version).
- `Config` is a frozen dataclass / pydantic model; attribute access, not dict.

### `src/rtn/traits.py`
- `load_traits(path) -> TraitVocab` with `.names -> list[str]` (stable order),
  `.valence[name]`, `.big_five[name]`, `.likableness[name]`, `.antonym[name]`,
  `.version`.

### `src/rtn/prompts/` (package, one module per version)
- `get_prompts(version) -> Prompts` with methods returning fully-rendered
  strings: `evidence_brief(chunks, traits)`, `e1_elicit(brief, traits)`,
  `e1_ablated_brief(brief, trait, strength)`, `e2_pair(i, j, persona)`,
  `e2_generic(i, j)`, `e3_chunk(chunk_text, traits)`.
- Each `Prompts` object exposes `.version` and a frozen dict of raw templates so
  paraphrase variants (Milestone 1.4) are just sibling modules `p1a`, `p1b`.

### `src/rtn/rater/`
- `base.Rater` — abstract. `rate(task, payload, *, account_hash, trait,
  replicate, config) -> RaterResult`. Handles cache lookup/store, ret/backoff,
  JSON-schema validation of the response, and replicate bookkeeping.
- `RaterResult`: `{ratings: dict[str, float | None], raw: str, cached: bool,
  usage: dict}`.
- `anthropic_backend.AnthropicRater(model, max_tokens, temperature, seed)` —
  uses the Messages API, `temperature=0`, forces JSON output, one retry on
  malformed JSON with a "return only JSON" nudge.
- `mock_backend.MockRater(seed)` — **deterministic**, no network. Given a
  ground-truth `d` and trait-level vector injected via the payload (synthetic
  runs) it returns internally-consistent ratings; given real text it returns a
  seeded hash-based pseudo-rating. Used for all tests and synthetic dry runs.
- `cache.CallCache(path)` — the sqlite wrapper. `get(key)`, `put(key, record)`.

### `src/rtn/ingest/`
- `hashing.hash_username(name, salt) -> str`; `UserMap(path)` load/append,
  refuses to run if `secrets/` is not gitignored.
- `exclusion.ExclusionFilter(path)` — `.is_excluded(subreddit) -> bool`,
  `.excluded_fraction(items) -> float`, `.check_or_raise(items, max_fraction)`.
- `chunking.chunk_items(items, cfg) -> list[Chunk]` — thread or window strategy,
  char caps, epoch assignment, drops sub-`min_chars` chunks.
- `arctic_shift.ArcticShiftClient` — DuckDB-over-HF-Parquet. `sample_frame(
  criteria) -> list[account_hash]`, `fetch_history(username) -> DataFrame`.
  **Stub acceptable in Milestone 1.1**; real impl needed for 1.2.
- `praw_client.PrawClient` — `fetch_recent(username) -> DataFrame` for
  gap-filling. Stub acceptable until 1.2.

### `src/rtn/estimate/`
- `evidence.build_brief(chunks, traits, rater, cfg) -> EvidenceBrief` —
  `{quotes: dict[trait, list[Quote]], density: dict[trait, float]}` where
  `Quote = {text, chunk_id, item_id}`.
- `e1_ablation.estimate_e1(brief, traits, rater, cfg) -> Network`
- `e2_pairwise.estimate_e2(brief_or_none, traits, rater, cfg) -> Network` and
  `estimate_generic(traits, rater, cfg) -> Network`
- `e3_covariation.estimate_e3(chunks, traits, rater, cfg) ->
  {undirected: Network, directed: Network, contemporaneous: Network, X: DataFrame}`
- `Network` = thin wrapper around a `40×40` numpy array + trait order + metadata,
  with `.to_long_df()`, `.save(path)`, `.load(path)`.

### `src/rtn/centrality.py`
- `all_centralities(net, density=None, cfg=...) -> DataFrame`
- `sla_centrality(d, ...)`, `eigenvector_centrality(d)`, `pagerank(d, alpha)`,
  `personalized_pagerank(d, alpha, teleport)`, `strength(d)`, `betweenness(d)`
- `rank_correlation_matrix(cent_df, method="spearman") -> DataFrame`

### `src/rtn/validate/`
- `synthetic.py`:
  - `make_generic_prior(vocab, seed)` → one folk-theory matrix shared across all
    synthetic accounts (antonyms + same-domain traits depend strongly).
  - `make_ground_truth(vocab, d_generic, seed) -> GroundTruth` where
    `GroundTruth = {d_true: 40×40 signed weights, trait_levels: 40-vec, ...}`.
    **Near-DAG, not strict DAG**: mostly forward edges in a random topological
    order + ~12% weaker back-edges (`feedback_fraction`). A strict DAG is
    nilpotent → SLA / eigenvector centrality degenerate; real trait networks
    (Elder et al.) have feedback loops, so the near-DAG is the faithful target.
  - `simulate_history(gt, volume, noise, seed, account_hash) -> SyntheticHistory`
    — chunk stubs + a per-chunk latent trait matrix `z` from a lag-1 VAR whose
    transition matrix is derived from `d_true` (so across-chunk covariation
    carries the structure for E3). `volume ∈ {thin,medium,thick}` sets `n_chunks`;
    noise inflates measurement + innovation SD.
  - `SyntheticOracle(hist, seed)` — answers every rater task (`e1_baseline`,
    `e1_ablated`, `e2_pair`, `e2_generic`, `e3_chunk`, `evidence_brief`) from the
    ground truth + seeded noise. Measurement model documented in the module
    docstring. Injected into `MockRater` via `build_rater(cfg, vocab, oracle=...)`.
  - `run_recovery(cfg) -> DataFrame` — full grid
    (`n_ground_truth_dags × volume × noise × replicates_per_cell`), runs the
    whole pipeline with the configured rater, returns per-cell:
    edge-weight recovery `r` (E1, E2, E3 vs. true `d`), centrality rank-recovery
    `ρ` (per measure), plus E1↔E3 convergence.
- `variance.py`:
  - `fit_icc(networks_long_df) -> ICCResult` — crossed random effects on
    `weight ~ 1 + (1|account_hash) + (1|trait_pair)`; report
    `ICC_account = σ²_account / (σ²_account + σ²_pair + σ²_resid)` with a
    bootstrap CI. **Gate: `ICC_account ≥ 0.10`.** Use `statsmodels` MixedLM with
    variance components, or a method-of-moments crossed decomposition — document
    which and its assumptions.
- `reliability.py`:
  - `split_half(account, method) -> {r_edges, rho_centrality}` for
    `method ∈ {random, temporal}`. Random = random chunk assignment; temporal =
    first-half vs second-half by `epoch`. **The temporal split is the real
    test.** Report both.
- `nulls.py`:
  - `permutation_networks(accounts, n_perm, cfg) -> DataFrame` — build corpora
    from randomly mixed accounts, run the pipeline, get a null distribution of
    between-account distinctiveness. Real corpora must exceed this null.
- `report.py` — assembles `reports/milestone1.md` + figures. Blunt about
  failures. No account identifiers.

### `src/rtn/analysis/` — MILESTONE 2, do not build yet
`h1_coherence.py`, `h2_feedback_asymmetry.py`, `h3_propagation.py`,
`pandora_convergence.py`. See §9.

---

## 7. Milestone 1 — validation. Build in this order.

Write everything to `reports/milestone1.md` with figures. Be blunt about
failures. Nothing in Milestone 2 starts until the user has seen this report.

### 1.1 Synthetic recovery
Ground-truth DAGs → synthetic histories crossing evidence volume
(thin/medium/thick) × noise → full pipeline → report edge-weight recovery `r`
and centrality rank recovery `ρ`. **Purpose: set `k` (E1 ablation count), chunk
count, and replicate count before spending anything on real accounts.** Runs on
the `mock` rater end-to-end for free; also run a small grid on the real
`anthropic` rater to check the mock's realism.
Output section: "Synthetic recovery".

### 1.2 Nomothetic / idiographic variance partition
~50 real accounts (needs ingest — now built, see `scripts/ingest_accounts.py`).
Fit crossed random effects on `d[i][j]` with `account_hash` and `trait_pair` as
grouping factors (§2b, N2). Report:
- the **nomothetic network `D̄`** (fixed-effect edge vector) and its centrality,
  next to N1 (E2 generic) and N3 (pooled E3) for convergence;
- `ICC_account` with a bootstrap CI = the idiographic share of edge variance;
- `corr(D̄, D_generic)` — how far the grounded pooled network sits from pure
  folk theory.

**Not a pass/fail gate.** `ICC_account` near 0 routes the substantive work to
the nomothetic branch (H1–H3 on `D̄` centrality + person-level self-description
weighting of `Bᵢ`); a large `ICC_account` routes it to the idiographic branch
(H1–H3 per account). Write which branch the data selects and why. Only a
*pathological* result — `D̄` itself uninformative (≈ `D_generic` **and** no
structure beyond antonym pairs) **and** `ICC_account` ≈ 0 — means stop and
redesign the estimator.

### 1.3 Split-half reliability
Per corpus: random chunk split and temporal split. Correlate centrality vectors
across halves. Report both; the temporal split is the real test.

### 1.4 Prompt invariance
Three paraphrases of the ablation and rating prompts (`prompts/p1a.py`,
`p1b.py`, `p1c.py`) + trait-order permutation. Report centrality rank stability.

### 1.5 Permutation null
Networks from corpora assembled from randomly mixed accounts. Between-account
distinctiveness on real corpora must exceed this null.

---

## 8. Milestone 2 — substantive analyses (only after the gate)

Reddit is the right corpus because it is a **naturally occurring social-feedback
environment** — the thing the lab paradigm studied under controlled conditions.
Votes and replies are feedback; subsequent posting is updating.

**Branch selected by 1.2.** If the variance partition is mostly nomothetic,
`centrality` in H1–H3 below is centrality on `D̄` (same value for everyone) and
the individual-difference term is the person's self-description weighting of `Bᵢ`
(§2b). If it is meaningfully idiographic, `centrality` is per-account centrality
on `Dᵢ`. The hypotheses are stated the same way either way.

- **H1 — coherence / resistance to change.** Trait expression drifts less across
  time epochs for high-centrality traits. Model: `centrality → temporal
  stability of chunk-level trait scores`, controlling for evidence density.
- **H2 — feedback asymmetry.** After identifiable interpersonal **negative**
  feedback (heavily downvoted comments; direct criticism in replies), subsequent
  trait expression shifts **more for peripheral than central** traits, and
  **more for positive than negative** feedback. Naturalistic analog of the
  original asymmetry result.
- **H3 — propagation.** Post-feedback change in trait `j` decays with network
  distance from the fed-back trait `i`.
- **PANDORA convergent validity (node levels only).** PANDORA pairs Reddit users
  with self-reported Big Five. Map the trait vocab onto Big Five facets (the
  `big_five` field) and check node-level convergence. **Validates the nodes, not
  the edges — state this explicitly.**

`preregistration.md` for H1–H3 is written and committed **before** running them
on the full sample. Hold out a confirmatory subsample.

---

## 9. Cross-cutting engineering rules

- **Config hash on every artifact.** A network file name embeds
  `{config_hash8}__{prompt_version}`; the sidecar JSON has the full config.
- **Cache is authoritative.** A rerun with an unchanged config + prompt version
  must issue zero model calls. Cache key includes `config_hash` so a config
  change correctly misses.
- **`temperature = 0`, fixed `seed`.** Replicates still vary (API
  non-determinism); that variance is the per-cell `weight_sd` and is reported.
- **No raw usernames past `ingest/`.** `hash_username` is called exactly once per
  account, at fetch time. `secrets/usermap.json` is the only raw store and is
  gitignored; `UserMap` refuses to run otherwise.
- **Aggregate-only outputs.** `report.py` has no code path that writes an
  account hash next to a trait profile.
- **Tests use `MockRater`.** No test hits the network or needs `ANTHROPIC_API_KEY`.
- Python + SQL (DuckDB) where SQL is cleaner. No notebooks. Scripts in
  `scripts/`, output in `reports/`.

---

## 10. Build order checklist (keep `docs/STATUS.md` in sync)

1. [x] Repo scaffold, `pyproject.toml`, `.gitignore`, `config/*.yaml`, `traits/*`
2. [x] `config.py`, `traits.py`, `network.py` + tests
3. [x] `prompts/p1.py` (templates) — paraphrases p1a/p1b/p1c still TODO (1.4)
4. [x] `rater/` — `base`, `cache`, `mock_backend`, `anthropic_backend` + tests
   (anthropic backend written, not yet run against the real API)
5. [x] `ingest/` — `hashing`, `exclusion`, `chunking` + tests; `clients.py` stubbed
6. [x] `centrality.py` + tests (SLA≈eigenvector assertion)
7. [x] `estimate/` — `common`, `evidence`, `e1_ablation`, `e2_pairwise`,
   `e3_covariation` + tests
8. [x] `validate/synthetic.py` — near-DAG ground truth + oracle + history sim
9. [x] `validate/recovery.py`, `report.py` + `scripts/run_synthetic_recovery.py`
10. [~] **Run 1.1 (first pass done, `reports/milestone1.md` §1.1 written).**
    Remaining: analyst fills the "Read / decisions" TODOs, scale grid up
    (`config/default.yaml` values), run a tiny real-API grid to check mock realism.
11. [x] `ingest/clients.py` real impl — `ArcticShiftClient` (HTTP API, time-
    paginated per-author fetch), `ArcticShiftDumpClient` (DuckDB/HF sample
    frame), `PrawClient` (gap-fill). `ingest/pipeline.py` +
    `scripts/ingest_accounts.py` (hash → fetch → exclusion → inclusion → chunk →
    `data/accounts/index.parquet`). Tested on real accounts.
12. [ ] `estimate/nomothetic.py` — N2 pooled fixed effect (`D̄`) + per-account
    deviations `Bᵢ`; N3 pooled E3. `validate/variance.py` — crossed random
    effects → `ICC_account`, `D̄` centrality, N1/N2/N3 convergence. Run 1.2 on
    ~50 accounts. **Reports the nomothetic/idiographic partition; not a gate.**
13. [ ] `validate/reliability.py`, `nulls.py`, prompt paraphrases → 1.3–1.5
14. [ ] `scripts/build_network.py` batch-run over the account index (real rater)
15. [ ] Full `reports/milestone1.md`, user review → choose nomothetic vs
    idiographic branch
16. [ ] `preregistration.md`, then `analysis/`

### Known gaps / debt (carry forward)
- `estimate/e3_covariation.ebicglasso` over-sparsifies at p=40 (EBIC picks a
  near-empty graph; a "densest fit" fallback masks it). E3 *undirected* recovery
  in synthetic is ~0 as a result. Fix: proper EBICglasso port or
  `GraphicalLassoCV`. Real methods gap, not just a synthetic artefact.
  (E3 *directed* / graphical VAR is fine — recovers r≈0.8 at thick evidence and
  converges with E1 at ρ≈0.75.)
- E2 runs all 1560 ordered pairs × replicates — the cost driver for real runs.
  Decide whether to subsample pairs or drop E2 to undirected-only before 1.2.
- `traits.yaml` likableness values are placeholders except honest/dishonest.

---

## 11. Key references

- Sloman, S. A., Love, B. C., & Ahn, W. (1998). Feature centrality and
  conceptual coherence. *Cognitive Science, 22*(2), 189–228.
- Elder, J., Cheung, B., Davis, T., & Hughes, B. (2023). Mapping the self:
  Idiographic trait networks and belief updating. *JPSP*. (and Elder, Davis &
  Hughes.)
- Anderson, N. H. (1968). Likableness ratings of 555 personality-trait words.
  *JPSP, 9*(3), 272–279.
- Epskamp et al. — `qgraph` / `bootnet` / `graphicalVAR` (methods reference for
  EBICglasso and graphical VAR; we reimplement in Python).
- PANDORA dataset (Gjurković et al., 2021) — Reddit users × Big Five.
- Arctic Shift — https://github.com/ArthurHeitmann/arctic_shift ; HF datasets.
