# Trait vocabulary — provenance

## What this is

`traits.yaml` is the fixed node set for v1: **40 traits, 20 antonym pairs,
valence-balanced** (20 positive-valence, 20 negative-valence). The set is fixed,
not discovered per account, for two reasons:

1. Centrality vectors are only comparable across accounts if they are defined
   over the same nodes.
2. Idiographic node discovery confounds "this trait is absent from the person"
   with "this trait was never mentioned in the corpus". A fixed vocabulary lets
   a trait score low without disappearing from the graph.

Valence balance is load-bearing: the original paradigm (Elder, Cheung, Davis &
Hughes, *JPSP* 2023) is partly about positivity motives, and Milestone 2's
feedback-asymmetry hypotheses (H2) require negative-trait coverage.

## Source

The *Mapping the Self* OSF materials were **not** retrieved for this build (per
the session's working decision). The list is instead a curated, valence-balanced
draw constructed to satisfy:

- **20 antonym pairs**, one positive and one negative pole each.
- **Big Five coverage on both poles of every domain** (O, C, E, A, N) — 4 pairs
  for A, 4 for C, 4 for O, 3 for E, 3 for N. The `big_five` field records the
  primary IPIP domain and loading sign for each trait, for the PANDORA
  node-level convergent-validity check in Milestone 2.
- **Single-word, non-clinical adjectives** in common use, so that Reddit text
  can plausibly bear on them and so no trait points at a protected disclosure
  category.

Lexical anchors are the **IPIP** Big Five adjective markers
(Goldberg, 1992; https://ipip.ori.org/) and the **Anderson (1968)** trait-word
norms:

> Anderson, N. H. (1968). Likableness ratings of 555 personality-trait words.
> *Journal of Personality and Social Psychology, 9*(3), 272–279.

### `likableness` values

The `likableness` field is the Anderson (1968) mean likableness rating on the
0–6 scale. **Only `honest` (5.55) and `dishonest` (0.41) are transcribed
directly from the published table** (`approx: false`). Every other value is
`approx: true` — a reconstruction placed to preserve the rank ordering and
valence split, not a citable figure.

**Before any analysis that uses the exact likableness number** (e.g. as a
covariate in the variance decomposition, or in the H2 positive-vs-negative
contrast), replace the `approx: true` values with the real ratings from
Anderson (1968) Table 1, or drop traits not in that table and re-balance.

## Changing the vocabulary

Any edit to `traits.yaml` bumps `version:` and invalidates every cached model
call and network artifact (the config hash covers the trait list). Do not edit
it mid-study. A v2 with idiographic node discovery is out of scope here and is
noted in the top-level plan as a separate design.
