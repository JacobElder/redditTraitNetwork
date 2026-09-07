# METHOD.md — how the pipeline turns Reddit text into a trait network

Written for someone who knows the original paradigm (Sloman, Love & Ahn, 1998;
Elder, Cheung, Davis & Hughes, *JPSP* 2023) but not this implementation. Every
symbol and every number in `reports/milestone1.md` is defined here and tied to
the code that produces it. If this disagrees with the code, the code wins — tell
me and I'll fix the doc.

---

## 0. The translation problem

**Original paradigm** (Elder, Cheung, Davis & Hughes, *JPSP* 2023 — network
construction). A large trait list is reduced by normative rating consistency to
~296 words. Separate participants then, for each **target trait**, free-nominate
*"which traits does [TARGET] depend upon?"* from the remaining list. The
**directed adjacency matrix** is built by **consensus threshold**: `A[i][j] = 1`
iff ≥ 25 % of participants nominated *j* as dependent on *i*. This network is
**nomothetic by construction** — the paper is explicit that it "requires a
certain degree of consensus [and] does not necessarily reflect people's
individual beliefs about dependencies." Outdegree centrality (how many traits
depend on a given trait) is the headline measure. Individual differences enter
only *later*, when people rate *themselves* on the traits and centrality-weighted
self-structure predicts belief updating and vmPFC response.

**Here.** There is no participant to nominate dependencies. We have only the
person's public Reddit writing. So the dependency structure has to be
**estimated from the text**, per person, and then combined across people into a
nomothetic network — the analog of the consensus matrix.

The per-person estimate is a **naturalistic adaptation**: instead of asking
"which traits does X depend on," we perturb the *textual evidence* for X and
measure how the other trait judgments move (E1, §4). Two other estimators (E2,
E3) are designed to fail differently, so agreement between them is evidence the
estimate is real rather than an artifact of one method. The per-person matrices
`Dᵢ` are averaged into `D̄` (§7), which plays the role of the original's
consensus network; centrality math on `D̄` is the same as the original's.

---

## 1. From a Reddit account to a corpus

`scripts/ingest_accounts.py` → `src/rtn/ingest/`

1. **Fetch.** `ArcticShiftClient` pulls the account's full public comment + post
   history from the Arctic Shift API (the Pushshift successor), newest-first,
   capped at 6,000 items. No credentials, no Reddit API.
2. **Hash.** The username is run through `sha256(salt + username)[:16]` exactly
   once, at fetch time (`ingest/hashing.py`). Nothing downstream ever sees the
   raw name; the raw→hash map lives only in one gitignored file.
3. **Exclusion filter.** Items from support communities (mental health,
   addiction/recovery, abuse survivor, eating disorder, grief — full list in
   `config/excluded_subreddits.yaml`) are dropped. If ≥ 50 % of an account's
   history is in those subs, the account is rejected entirely.
4. **Inclusion criteria.** Keep the account only if ≥ 300 comments, ≥ 2-year
   span, ≥ 5 subreddits (`config/default.yaml → ingest`).
5. **Chunk.** The history is cut into **45 contiguous time windows**; from each
   window an evenly-spaced ≤ 10 000-character *sample* of the items is taken
   (`ingest/chunking.py`, `_sampled_windows`). Each chunk is a snapshot of one
   period of the person's activity — the unit E3 needs for a time series.

Output: `data/accounts/{hash}/{items,chunks}.parquet` and one line in
`data/accounts/index.parquet` (hashes + counts only, no identifiers).

---

## 2. The trait vocabulary

`traits/traits.yaml` — **40 traits, 20 antonym pairs, valence-balanced**
(20 positive, 20 negative), from **IPIP Big Five adjective markers**
(Goldberg, 1992) with Anderson (1968) desirability values, chosen for Big Five
coverage on both poles of every domain. This is the intended vocabulary, not a
placeholder.

Deliberately **smaller than the original's ~296** and **fixed** (not discovered
per person):

- **Cost.** E1 (§4) is `O(k²)` model calls per account. 40 traits ≈ 1,700
  calls/account; 296 would be ≈ 55× that — infeasible at the current
  throughput. 40 is the Milestone-1 scaffold; the confirmatory study can widen
  it on a paid model. Widening also needs the E1 ablation "reverse toward the
  opposite" step re-thought, since it currently uses the antonym pairing.
- **Comparability.** Centrality vectors only compare across accounts if the
  nodes are the same.
- **Discovery would confound** "trait absent from the person" with "trait never
  mentioned in the corpus."

The original's node set is also fixed and nomothetic; the idiographic part there
and here is **node weighting** (§7), not node selection.

---

## 3. The evidence brief — what the corpus says about each trait

`src/rtn/estimate/evidence.py`, `prompts/p1.py → evidence_brief`

For each account we build a **brief**: a structured dossier of quoted textual
evidence, per trait.

- **Per-chunk extraction** (`strategy="per_chunk"`, the default): the LLM reads
  *one chunk at a time* and pulls every short verbatim quote (≤ 240 chars) in
  that chunk that bears on any of the 40 traits, tagged `for` / `against`. Most
  traits get nothing from a given chunk.
- **Merge**: quotes from all chunks are pooled, de-duplicated, and the top
  `quotes_per_trait` (8) kept per trait, spread across chunks.

Why per-chunk rather than one giant prompt: a whole 6,000-comment history is
~110k tokens; a single prompt made the model lazy (≈ 16 quotes for an entire
person). Per-chunk gets ~250–300 and works with any model's context window.

The brief is the object E1 edits. It also yields the **evidence-density** vector
used for node weighting (§7): the share of the account's quotes that are about
each trait.

---

## 4. E1 — counterfactual ablation (the primary estimator)

`src/rtn/estimate/e1_ablation.py`, `prompts/p1.py → e1_elicit`, `e1_ablate`

A **naturalistic adaptation** of the dependency question. The original asks a
participant "which traits does X depend upon?" We can't — there's no
participant, only text. So we operationalise "j depends on i" as: **if the
evidence that this person is *i* were removed, would an assessor still judge them
*j*?** If removing *i*'s evidence pulls *j*'s rating down, *j* depends on *i*.
The signed magnitude is the edge weight (a continuous relaxation of the
original's binary, consensus-thresholded edge).

1. **Baseline.** The LLM reads the full brief *as an assessor* and rates the
   person on all 40 traits, 0–100. Call this `r_j(full)`.
2. **Ablate trait i.** The LLM rewrites the brief so that every quote supporting
   trait *i* is removed and replaced with plausible passages **in this person's
   own voice and context** showing the opposite (`antonym(i)`). Everything about
   the other 39 traits is left untouched. (It is explicitly told *not* to write
   "imagine a person who…" — that would be asking for the model's generic prior
   instead of an edit grounded in this person's evidence.)
3. **Re-rate.** The LLM rates all 40 traits from the *ablated* brief:
   `r_j(ablate_i)`.
4. **Edge weight.**

   ```
   d[i][j] = r_j(full) − r_j(ablate_i)        (signed; d[i][i] := 0)
   ```

   i.e. *how much does trait j's rating move when trait i's evidence is pulled?*
   Positive → removing *i* lowers *j* (they support each other). Negative →
   removing *i* raises *j* (antagonistic; this is what antonym pairs do).

Doing this for all 40 traits gives a **40×40 signed directed matrix `Dᵢ`** for
account *i* — the "if you were no longer X, how much would Y change" table, one
row per *X*. Convention: **row = the trait removed, column = the trait that
moves.** Centrality keys off the rows (a trait many others depend on is
central).

Cost: 1 baseline + 40 (rewrite + re-rate) = ~81 model calls per account, plus
the brief. This is the expensive estimator.

**Its failure mode:** the LLM could impose a generic folk theory of how traits
relate, producing ~the same matrix for everyone. That is exactly what E3 is for.

---

## 5. E2 — direct pairwise elicitation (baseline / generic prior)

`src/rtn/estimate/e2_pairwise.py` — **currently OFF for the free-tier run.**

Asks a pairwise dependency-magnitude question, in persona, for every ordered
pair. Also fits `d_generic`: the same elicitation with **no persona and no
corpus** — the model's pure folk theory of trait dependence, one matrix for the
whole study. The point of E2 is to measure how much of E1's signal survives
subtracting `d_generic`. It's off now only because it's ~1,560 calls/account and
not needed for the current questions; it flips back on for the confirmatory
sample.

---

## 6. E3 — behavioural covariation (the independent check)

`src/rtn/estimate/e3_covariation.py`

Here the LLM is **only a rater** — it never reasons about dependencies.

1. **Rate every chunk.** For each of the 45 chunks, the LLM scores all 40 traits
   0–100 ("how strongly does *this excerpt* express each trait about its
   author"). Result: a **45 × 40 matrix `X`** (period × trait).
2. **Undirected network.** Ledoit-Wolf shrinkage covariance of `X`, inverted to
   a **partial-correlation matrix**. (Earlier we used graphical-lasso + EBIC;
   with ~45 observations and 40 variables it collapsed to an empty graph.
   Shrinkage is stable in that `p ≈ n` regime.) Edge = how two traits co-vary
   across the person's life *after* controlling for the other 38.
3. **Directed network.** Lag-1 graphical VAR (per-target LassoCV) over the
   time-ordered chunks. Currently too noisy at 45 time-points × 1 replicate to
   be useful; the undirected version is what we rely on.

**Why E3 is the check.** E1's risk is "the model imposes a folk theory." E3
can't do that — it never sees pairs, never states a dependency, just reads one
excerpt and rates traits. If E1 and E3 agree on the structure, the structure
isn't an E1 artifact. E3 estimates *covariation*, not conceptual dependency in
Sloman's sense — convergence is evidence, identity is not claimed.

---

## 7. Combining accounts: D̄, Bᵢ, and node weighting

`src/rtn/estimate/nomothetic.py`, `node_weights.py`

### D̄ ("D-bar") — the nomothetic network

The **mean of all accounts' E1 matrices**, `D̄ = mean_i Dᵢ`. This is "the shared
structure of how traits depend on each other" — the thing the framework says is
largely universal. It has its own centrality vector.

(A partial-pooling / crossed random-effects estimate is the planned upgrade; for
a balanced design the flat mean and the fixed-effect estimate coincide, so the
mean is fine for now.)

### Bᵢ — per-account edge deviation

`Bᵢ = Dᵢ − D̄`. Account *i*'s departure from the shared structure, edge by edge.
The framework does **not** predict that people have idiosyncratic edge
structures, so `Bᵢ` is treated as a **diagnostic**, not a substantive object.
Test: does `Bᵢ` (from E1) correlate with `E3ᵢ − Ē3` (the same account's
deviation in the independent estimator)? If `Bᵢ` were real structure, yes. First
results: ≈ 0 — so most of `Bᵢ` is elicitation noise.

### Node weighting — the idiographic layer

Per the framework, individuals differ in **how much each trait matters to their
self-concept**, not in the dependency structure. We estimate that weighting
per account, per trait, three ways (all reconstructed from the rater cache, no
new model calls):

| source | definition |
|---|---|
| **evidence density** | share of the account's extracted quotes that are about the trait (`evidence.py`) |
| **self-relevance** | `\|self-rating − 50\| / 50` from the E1 baseline — how strongly, either way, the trait defines them |
| **E3 salience** | mean over chunks of `\|chunk-score − 50\|` — how strongly and consistently the writing signals the trait |

**Idiographic centrality** = `personalised_pagerank(D̄, teleport = node_weightᵢ)`
(`centrality.py`) — the shared network, walked with the person's own weighting.
This is the intended DV for the Milestone-2 hypotheses.

The check: do the weight sources agree with each other within an account
(especially brief-density vs. E3-salience, which are independent)? First
results: ρ ≈ 0.04–0.08 — **they don't agree yet**, so we can't say the weighting
is a reliably measured signal at this sample size / model / replicate count.

---

## 8. Centrality

`src/rtn/centrality.py` — all measures operate on the same `d` matrix.

| measure | what it is |
|---|---|
| **out-strength** | row sum of `\|d\|` — the continuous analog of **Elder et al.'s outdegree centrality** (their headline measure: how many traits depend on this one). `out_strength_signed` keeps the sign. |
| **in-strength** | column sum — analog of their indegree. |
| **SLA iteration** | the Sloman–Love–Ahn (1998) conceptual-centrality operationalisation (Elder's theoretical basis): `c ← normalize(δ·(\|d\|·c) + (1−δ)·c)`, δ = 0.85; converges to the dominant eigenvector of `\|d\|`. Recursive: a trait is central if the traits that depend on it are themselves central. |
| **eigenvector** | dominant eigenvector of `\|d\|` directly. Sanity check — should rank-correlate ≈ 1 with SLA. |
| **PageRank** | on `\|d\|`, column-stochastic, damping 0.85. |
| **personalised PageRank** | same, teleport vector = the account's node weights (§7). This is the idiographic-centrality DV. |
| **betweenness** | comparison measure. |

Elder et al. also report a **pairwise-similarity** measure (shared neighbours,
degree-weighted); not yet implemented here.

---

## 9. The variance partition (where `ICC_idiographic` comes from)

`src/rtn/validate/variance.py`

Question: **of the total variation in dependency-edge weights, how much is
shared across people vs. person-specific?**

1. **Stack.** Every account's E1 matrix → one long table: one row per
   `(account, directed trait-pair)`, value = the weight. With 9 accounts and
   `40×39 = 1560` pairs that's a `9 × 1560` table `m`.
2. **Two-way ANOVA without replication** on `m` (`_decompose`): decompose the
   total variance into

   | component | meaning | code |
   |---|---|---|
   | **σ²_pair** | trait pairs differ, *the same way for everyone* — the nomothetic structure | `(MS_pair − MS_resid) / n_accounts` |
   | **σ²_account** | a person rates *every* edge higher/lower — an additive personal offset | `(MS_account − MS_resid) / n_pairs` |
   | **σ²_account:pair** | a person weights *particular* pairs unusually — idiographic *pattern* | `MS_resid − σ²_rep/R` |
   | **σ²_replicate** | elicitation noise | mean of the stored per-cell replicate variances |

3. **The noise problem.** We run 1 replicate, and the model is near-deterministic
   at temperature 0, so **σ²_replicate is measured as 0**. That means elicitation
   noise cannot be separated from `σ²_account:pair` — the idiographic term is
   inflated. As a stopgap we compute an **off-target noise proxy**: the median
   squared weight over the (mostly-null) trait pairs, since most pairs have no
   real dependency and their weight is just noise. Currently ≈ 25 (noise SD ≈ 5).

4. **The number.**

   ```
   ICC_idiographic       = (σ²_account + σ²_account:pair) / (σ²_account + σ²_pair + σ²_account:pair)
   ICC_idiographic (adj) = same, but with the off-target proxy subtracted from σ²_account:pair
   ```

   The raw value is an **upper bound** (noise is in it); the adjusted value is a
   **lower bound**. Bootstrap resampling *over accounts* gives the CI.

   At 9 accounts: raw ≈ 0.77, adjusted ≈ 0.72, bootstrap CI ≈ [0.54, 0.76].

5. **What it does *not* tell you.** A high `ICC_idiographic` here means "E1
   produces different matrices for different people." It does **not** by itself
   mean those differences are real conceptual structure — that requires the
   independent corroboration (`Bᵢ` vs E3), which is currently ≈ 0. So we are
   **not** leaning on `ICC_idiographic` as a headline; it is reported with both
   bounds and the corroboration caveat.

---

## 10. "E1↔E3 convergence" — what the phrase means

Several related correlations, all computed in `scripts/milestone1_2.py`:

| reported as | computation | at n = 9 |
|---|---|---|
| **D̄ vs pooled E3 (undirected)** | Pearson *r* between the off-diagonal of symmetrised `D̄` and the off-diagonal of the mean E3-undirected network | **+0.68** |
| E1↔E3, mean per-account | same correlation computed per account, then averaged | +0.33 |
| **E1-residual ↔ E3-residual** | per account, correlate `(Dᵢ − D̄)` with `(E3ᵢ − Ē3)` — isolates whether the *idiographic* part agrees | **≈ 0** |
| D̄ vs generic prior (E2) | how far D̄ sits from pure folk theory (only when E2 is run) | — |

The first row is the validation that matters: two methods that fail differently
agree on the shared structure, and the agreement **rises** as accounts are added
(0.56 at n = 5 → 0.64 at n = 8 → 0.68 at n = 9).

---

## 11. Current status (2026-09-07, 9 accounts, free-tier model)

| claim | evidence | verdict |
|---|---|---|
| A shared trait-dependency network can be estimated from Reddit text | E1↔E3 on D̄, *r* = 0.68 and rising with n | **supported** |
| People have idiosyncratic *edge* structures | `Bᵢ` vs E3-residual ≈ 0 | **no** (and not predicted by the framework) |
| Per-account node *weighting* is a reliably measured signal | weight sources agree at ρ ≈ 0.04–0.08 | **not yet** — needs replicates / p1↔p1b, more varied accounts |
| `ICC_idiographic` | raw 0.77 / adj 0.72, CI [0.54, 0.76] | **reported, not relied on** — inflated by unmeasured noise |

---

## 12. Known limitations / TODO

- **1 replicate.** σ²_replicate is unmeasured. Fix: run each account under
  `prompts/p1` *and* `prompts/p1b` (a semantic paraphrase, already written) and
  treat the two as replicates — the disagreement is the real elicitation noise.
- **Free model.** `gemini-3.5-flash-lite` does the ablation rewrites. Quality
  unknown; a stronger model (or Anthropic/OpenAI) is a config switch.
- **Homogeneous sample.** First 9 accounts are all from
  changemyview / AITA / self / CasualConversation / AskReddit. A varied second
  batch (AskHistorians, personalfinance, Fitness, travel, Cooking, books, DIY,
  dataisbeautiful) is ingesting now to test whether the weak node-weighting
  effect is real-but-suppressed.
- **D̄ is a flat mean**, not a shrinkage / partial-pooling estimate.
- **E3 directed (VAR)** is too noisy to use at 45 time-points × 1 replicate.
- **E2 is off** — no `d_generic` comparison in the current numbers.
- **n = 9.** The bootstrap CI is wide; scaling to ~30–48 is the main lever.
