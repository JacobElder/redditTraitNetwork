# Milestone 1 — measurement validation

<!-- SECTION:synthetic_recovery:start -->
## 1.1 Synthetic recovery

_Generated 2026-09-01 04:11 UTC. 90 pipeline runs (5 ground-truth DAGs × 3 volumes × 2 noise levels × 3 replicates)._

### Edge-weight recovery (mean correlation with true `d`)

| volume   |   noise |   edge_r_e1 |   edge_r_e1_minus_generic |   edge_r_e2 |   edge_r_e3_undirected |   edge_r_e3_directed |
|:---------|--------:|------------:|--------------------------:|------------:|-----------------------:|---------------------:|
| medium   |    0    |       0.723 |                     0.441 |       0.202 |                 -0.009 |                0.52  |
| medium   |    0.25 |       0.648 |                     0.423 |       0.2   |                 -0.008 |                0.513 |
| thick    |    0    |       0.862 |                     0.469 |       0.221 |                 -0.009 |                0.806 |
| thick    |    0.25 |       0.807 |                     0.461 |       0.214 |                 -0.019 |                0.803 |
| thin     |    0    |       0.47  |                     0.355 |       0.154 |                 -0.002 |              nan     |
| thin     |    0.25 |       0.405 |                     0.323 |       0.137 |                  0     |              nan     |

### Centrality rank recovery (mean Spearman ρ, E1 network vs true)

| volume   |   noise |   cent_rho_e1_betweenness |   cent_rho_e1_eigenvector |   cent_rho_e1_in_strength |   cent_rho_e1_out_strength |   cent_rho_e1_out_strength_signed |   cent_rho_e1_pagerank |   cent_rho_e1_sla |
|:---------|--------:|--------------------------:|--------------------------:|--------------------------:|---------------------------:|----------------------------------:|-----------------------:|------------------:|
| medium   |    0    |                     0.347 |                     0.745 |                     0.606 |                      0.857 |                             0.858 |                  0.665 |             0.745 |
| medium   |    0.25 |                     0.273 |                     0.738 |                     0.471 |                      0.793 |                             0.788 |                  0.685 |             0.738 |
| thick    |    0    |                     0.636 |                     0.882 |                     0.794 |                      0.949 |                             0.925 |                  0.797 |             0.882 |
| thick    |    0.25 |                     0.492 |                     0.837 |                     0.705 |                      0.915 |                             0.891 |                  0.763 |             0.837 |
| thin     |    0    |                     0.013 |                     0.476 |                     0.336 |                      0.554 |                             0.651 |                  0.435 |             0.477 |
| thin     |    0.25 |                     0.096 |                     0.299 |                     0.216 |                      0.377 |                             0.593 |                  0.28  |             0.298 |

### E1 ↔ E3 convergence (estimators without a shared failure mode)

| volume   |   noise |   e1_e3_sla_rho |   e1_e3_edge_r |
|:---------|--------:|----------------:|---------------:|
| medium   |    0    |           0.418 |          0.373 |
| medium   |    0.25 |           0.387 |          0.335 |
| thick    |    0    |           0.752 |          0.696 |
| thick    |    0.25 |           0.725 |          0.65  |
| thin     |    0    |         nan     |        nan     |
| thin     |    0.25 |         nan     |        nan     |

![E1 edge recovery](figures/m1_1_recovery_e1.png)

### Read / decisions

- TODO(analyst): state the minimum evidence volume at which E1 edge recovery and centrality rank recovery clear your bar, and set `ingest.min_comments` / chunk targets accordingly.
- TODO(analyst): pick `estimate.replicates` from where the recovery curves flatten.
- TODO(analyst): if E1−generic recovery collapses relative to raw E1, note that the account-specific signal is weak even in simulation.
- KNOWN GAP: E3 undirected recovery is ~0 and E1↔E3 convergence is weak in synthetic — `ebicglasso` over-sparsifies at p=40 (see docs/PLAN.md 'Known gaps'). Fix the estimator before reading E3 numbers as evidence.
<!-- SECTION:synthetic_recovery:end -->

<!-- SECTION:variance_partition:start -->
## 1.2 Nomothetic / idiographic variance partition

_Generated 2026-09-07 17:38 UTC. 9 accounts × 1560 directed trait pairs × 1 replicates. Method-of-moments decomposition._

### Where the edge-weight variance lives

| component | variance | share | 95% CI |
|---|--:|--:|---|
| **σ²_pair — nomothetic** (shared structure) | 28.607 | 23% | (29.3604, 56.9542) |
| σ²_account (additive person shift) | 1.765 | 1% | — |
| **σ²_account:pair — idiographic pattern** | 91.832 | 75% | (55.5984, 104.359) |
| σ²_replicate (measured, from stored replicate SDs) | 0.000 | — | — |
| σ²_replicate — off-target proxy (median null-cell d²) | 19.753 | — | — |

> ⚠️ σ²_replicate is 0 as measured (1 replicate / deterministic model). The **off-target proxy** — the median squared weight over the mostly-null trait pairs — is a data-driven noise floor. `ICC_idiographic` (proxy removed) below is a **lower bound**; the raw `ICC_idiographic` is an **upper bound**. Milestone 1.4 (prompt paraphrases) gives the real number.

**ICC_idiographic = 0.766** (upper bound; CI (0.5575, 0.7541))  &nbsp;·&nbsp;  **noise-adjusted = 0.721** (lower bound) — share of edge variance that is person-specific.
ICC_account-only = 0.014 (CI (0.0034, 0.0205)).

### D̄ — the nomothetic network

Top traits by SLA centrality on the pooled fixed-effect network:

| trait       | measure   |    value |
|:------------|:----------|---------:|
| forgiving   | sla       | 0.281604 |
| openminded  | sla       | 0.256538 |
| passive     | sla       | 0.231583 |
| resentful   | sla       | 0.228006 |
| kind        | sla       | 0.226746 |
| hardworking | sla       | 0.199807 |
| disciplined | sla       | 0.194028 |
| considerate | sla       | 0.189605 |

Convergence checks:

- D̄ vs pooled E3 (undirected): r = +0.675
- E1↔E3 (undirected), mean per-account: r = +0.340
- D̄ vs pooled E3 (directed VAR): r = +0.108
- E1↔E3 (directed VAR), mean per-account: r = +0.013
- E1 residual ↔ E3 residual (idiographic corroboration): r = +0.012
-   — range across accounts: (-0.047, 0.058)
- [node weighting] weight consistency: density ↔ self-relevance, mean ρ: r = +0.079
- [node weighting] weight CORROBORATION: brief-density ↔ E3-salience, mean ρ: r = +0.044
- [node weighting]   — E3-salience range across accounts: (-0.12, 0.4)
- [node weighting] between-account personalised-centrality, mean ρ: r = +0.873
- [node weighting] personalised vs unweighted D̄ centrality, mean ρ: r = +0.849

### Branch selection (docs/PLAN.md §7.2, §8)

**Only the nomothetic layer is validated so far.**

- **Shared network `D̄`: supported** — E1 and E3 converge on it (and the convergence rises as accounts are added). The framework predicts the dependency *structure* is largely universal; that holds.
- **Per-account *edge* deviations (`Bᵢ`): not corroborated** — E1-residual ↔ E3-residual r = +0.012. Expected — the framework doesn't predict idiosyncratic edges — so the raw `ICC_idiographic` [0.72, 0.77] is largely elicitation noise.
- **Per-account *node weighting*: NOT yet corroborated** — the weight sources disagree (brief-density ↔ E3-salience ρ = +0.04; density ↔ self-relevance also low). Either they capture genuinely different facets, or the per-trait weight estimates are too noisy at 1 replicate / this model / this sample. Cannot yet say the idiographic *weighting* is a real, measurable signal.

**Next:** more (varied) accounts; replicate or p1↔p1b the weight-source elicitations; re-check. Until then only `D̄` and its (unweighted) centrality are on solid ground.

<!-- SECTION:variance_partition:end -->
