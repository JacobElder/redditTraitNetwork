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

_Generated 2026-09-02 12:33 UTC. 5 accounts × 1560 directed trait pairs × 1 replicates. Method-of-moments decomposition._

### Where the edge-weight variance lives

| component | variance | share | 95% CI |
|---|--:|--:|---|
| **σ²_pair — nomothetic** (shared structure) | 22.696 | 17% | (22.6958, 84.3817) |
| σ²_account (additive person shift) | 1.113 | 1% | — |
| **σ²_account:pair — idiographic pattern** | 106.757 | 82% | (42.1006, 115.6461) |
| σ²_replicate (elicitation noise, excluded above) | 0.000 | — | — |

> ⚠️ σ²_replicate ≈ 0 (run used 1 replicate, or a deterministic model). Elicitation noise is **not** separated from the idiographic-pattern term, so σ²_account:pair and ICC_idiographic below are **upper bounds**. Milestone 1.4 (prompt paraphrases) gives the real noise estimate; re-run 1.2 after.

**ICC_idiographic = 0.826**  (CI (0.3782, 0.8262)) — share of non-noise edge variance that is person-specific (additive + pattern).
ICC_account-only = 0.009 (CI (0.0009, 0.011)).

### D̄ — the nomothetic network

Top traits by SLA centrality on the pooled fixed-effect network:

| trait         | measure   |    value |
|:--------------|:----------|---------:|
| forgiving     | sla       | 0.249197 |
| incurious     | sla       | 0.246422 |
| secure        | sla       | 0.236306 |
| disciplined   | sla       | 0.233229 |
| passive       | sla       | 0.207785 |
| resentful     | sla       | 0.20626  |
| outgoing      | sla       | 0.202421 |
| unintelligent | sla       | 0.19113  |

Convergence of the three nomothetic estimates:

- D̄ vs pooled E3 (undirected): r = +0.556
- E1↔E3 (undirected), mean per-account: r = +0.293
- D̄ vs pooled E3 (directed VAR): r = +0.094
- E1↔E3 (directed VAR), mean per-account: r = +0.009

### Branch selection (docs/PLAN.md §7.2, §8)

`ICC_idiographic = 0.826`, and the idiographic variance is mostly *pattern* (σ²_account:pair > σ²_account) rather than an additive shift — people have **distinctive dependency structures**. Milestone 2 runs H1–H3 on per-account `Dᵢ` centrality.

<!-- SECTION:variance_partition:end -->
