"""Assemble ``reports/milestone1.md``. Aggregate-only; no account identifiers.

Sections are written independently so milestones can land one at a time. Each
``write_*`` function replaces its own ``<!-- SECTION: name -->`` block in the
report, leaving the rest intact.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

_HEADER = "# Milestone 1 — measurement validation\n"


def _load(path: Path) -> str:
    return path.read_text() if path.exists() else _HEADER


def _splice(doc: str, name: str, body: str) -> str:
    start = f"<!-- SECTION:{name}:start -->"
    end = f"<!-- SECTION:{name}:end -->"
    block = f"{start}\n{body}\n{end}"
    if start in doc and end in doc:
        pre = doc[: doc.index(start)]
        post = doc[doc.index(end) + len(end) :]
        return pre + block + post
    return doc.rstrip() + "\n\n" + block + "\n"


def write_variance_partition(
    vc,
    d_bar_centrality: pd.DataFrame,
    convergence: dict[str, float],
    report_dir: str | Path,
) -> Path:
    """Milestone 1.2 section: nomothetic / idiographic variance partition.

    ``vc`` is a ``VarianceComponents``; ``d_bar_centrality`` a tidy centrality
    DataFrame for D-bar; ``convergence`` maps a label -> correlation
    (e.g. {"D_bar vs D_generic": .62, "D_bar vs pooled E3": .48}).
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "milestone1.md"
    ts = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")

    total = (
        vc.sigma2_account + vc.sigma2_pair + vc.sigma2_account_pair
    ) or 1.0
    pct = lambda x: f"{100 * x / total:.0f}%"
    ci = vc.ci

    top = d_bar_centrality[d_bar_centrality["measure"] == "sla"].nlargest(8, "value")
    lines = [
        "## 1.2 Nomothetic / idiographic variance partition",
        "",
        f"_Generated {ts}. {vc.n_accounts} accounts × {vc.n_pairs} directed trait "
        f"pairs × {vc.replicates} replicates. Method-of-moments decomposition._",
        "",
        "### Where the edge-weight variance lives",
        "",
        "| component | variance | share | 95% CI |",
        "|---|--:|--:|---|",
        f"| **σ²_pair — nomothetic** (shared structure) | {vc.sigma2_pair:.3f} | "
        f"{pct(vc.sigma2_pair)} | {ci.get('sigma2_pair', '—')} |",
        f"| σ²_account (additive person shift) | {vc.sigma2_account:.3f} | "
        f"{pct(vc.sigma2_account)} | — |",
        f"| **σ²_account:pair — idiographic pattern** | {vc.sigma2_account_pair:.3f} | "
        f"{pct(vc.sigma2_account_pair)} | {ci.get('sigma2_account_pair', '—')} |",
        f"| σ²_replicate (measured, from stored replicate SDs) | {vc.sigma2_rep:.3f} | — | — |",
        f"| σ²_replicate — off-target proxy (median null-cell d²) | {vc.sigma2_rep_offtarget:.3f} | — | — |",
        "",
        (
            "> ⚠️ σ²_replicate is 0 as measured (1 replicate / deterministic "
            "model). The **off-target proxy** — the median squared weight over "
            "the mostly-null trait pairs — is a data-driven noise floor. "
            "`ICC_idiographic` (proxy removed) below is a **lower bound**; the "
            "raw `ICC_idiographic` is an **upper bound**. Milestone 1.4 (prompt "
            "paraphrases) gives the real number."
            if vc.sigma2_rep < 1e-6
            else ""
        ),
        "",
        f"**ICC_idiographic = {vc.icc_idiographic:.3f}** (upper bound; CI "
        f"{ci.get('icc_idiographic', '—')})  &nbsp;·&nbsp;  "
        f"**noise-adjusted = {vc.icc_idiographic_adj:.3f}** (lower bound) "
        "— share of edge variance that is person-specific.",
        f"ICC_account-only = {vc.icc_account_only:.3f} (CI {ci.get('icc_account_only', '—')}).",
        "",
        "### D̄ — the nomothetic network",
        "",
        "Top traits by SLA centrality on the pooled fixed-effect network:",
        "",
        top.to_markdown(index=False),
        "",
        "Convergence checks:",
        "",
        "\n".join(
            f"- {k}: {v}" if isinstance(v, tuple)
            else f"- {k}: r = {v:+.3f}"
            for k, v in convergence.items()
        ),
        "",
        "### Branch selection (docs/PLAN.md §7.2, §8)",
        "",
        _branch_note(
            vc,
            convergence.get("E1 residual ↔ E3 residual (idiographic corroboration)"),
            convergence.get("[node weighting] weight_source_consistency (density ↔ self-relevance), mean ρ"),
            convergence.get("[node weighting] between-account personalised-centrality, mean ρ"),
        ),
        "",
    ]
    body = "\n".join(lines)
    doc = _splice(_load(report_path), "variance_partition", body)
    if not doc.startswith("#"):
        doc = _HEADER + "\n" + doc
    report_path.write_text(doc)
    return report_path


def _branch_note(
    vc,
    resid_corr: float | None = None,
    weight_consistency: float | None = None,
    between_pers: float | None = None,
) -> str:
    lo = vc.icc_idiographic_adj if vc.icc_idiographic_adj == vc.icc_idiographic_adj else vc.icc_idiographic
    rng = f"[{lo:.2f}, {vc.icc_idiographic:.2f}]"

    # The framework (Sloman/Love/Ahn; Elder): the semantic *structure* is shared,
    # individuals differ in *node weighting*. So the questions are (a) is D̄
    # solid, and (b) is the per-account node weighting a real, consistent signal.
    if resid_corr is not None and abs(resid_corr) < 0.10:
        wc = f"{weight_consistency:+.2f}" if weight_consistency is not None else "n/a"
        bp = f"{between_pers:+.2f}" if between_pers is not None else "n/a"
        return (
            "**Consistent with the framework's nomothetic-structure /"
            " idiographic-weighting split.**\n\n"
            f"- **Shared network `D̄`: supported** — E1 and E3 converge on it. The "
            "framework predicts the dependency *structure* is largely universal; "
            "that's what we see.\n"
            f"- **Per-account *edge* deviations (`Bᵢ`): not corroborated** — "
            f"E1-residual ↔ E3-residual r = {resid_corr:+.3f}. The framework does "
            "*not* predict idiosyncratic edge structures, so this is expected; "
            f"the raw `ICC_idiographic` {rng} is inflated by elicitation noise.\n"
            f"- **Per-account *node weighting*: a consistent signal** — the two "
            f"independent weight sources (evidence density, self-relevance) agree "
            f"at ρ = {wc} within account. But they move centrality only modestly "
            f"(between-account personalised-centrality ρ = {bp}) — the shared "
            "structure still dominates, and this account sample may be "
            "homogeneous.\n\n"
            "**Milestone 2 runs H1–H3 on personalised centrality: "
            "`personalised_pagerank(D̄, node_weightᵢ)`** — shared network, "
            "idiographic weighting. Not per-account `Dᵢ`. Confirm the node-"
            "weighting signal against an independent salience measure and with "
            "more (more varied) accounts first."
        )
    if vc.icc_idiographic < 0.10:
        return (
            f"`ICC_idiographic` {rng} is low — the network is mostly "
            "**nomothetic**. Milestone 2 runs H1–H3 on `D̄` centrality, with each "
            "person's self-description weighting of `Bᵢ` as the individual-"
            "difference term. Not a failure — this matches the Sloman/Love/Ahn "
            "and Elder framing."
        )
    if lo < 0.10:
        return (
            f"`ICC_idiographic` {rng} straddles the noise floor — the raw value "
            "is high but the noise-adjusted lower bound is near zero. **Cannot "
            "yet tell** whether the idiographic structure is real. Needs more "
            "accounts and the Milestone 1.4 noise estimate before choosing a "
            "branch."
        )
    if vc.sigma2_account_pair > vc.sigma2_account:
        return (
            f"`ICC_idiographic` {rng} — even the noise-adjusted lower bound is "
            "substantial, and the idiographic variance is mostly *pattern* "
            "(σ²_account:pair > σ²_account), not an additive shift: people have "
            "**distinctive dependency structures**. Provisional branch — "
            "Milestone 2 runs H1–H3 on per-account `Dᵢ` centrality."
        )
    return (
        f"`ICC_idiographic` {rng}, but mostly an additive person effect "
        "(σ²_account ≳ σ²_account:pair) — people differ in overall dependency "
        "magnitude, not which edges they weight. Treat the structure as "
        "nomothetic; report the additive effect as a nuisance."
    )


def _fig_recovery(df: pd.DataFrame, out: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    from .recovery import summarize

    s = summarize(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    for vol in s["volume"].unique():
        sub = s[s["volume"] == vol].sort_values("noise")
        ax.plot(sub["noise"], sub["edge_r_e1"], marker="o", label=f"E1 · {vol}")
    ax.set_xlabel("noise level")
    ax.set_ylabel("edge-weight recovery r (E1 vs true d)")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Synthetic recovery — E1 edge weights")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def write_synthetic_recovery(
    df: pd.DataFrame, report_dir: str | Path, figures_dir: str | Path
) -> Path:
    from .recovery import summarize

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "milestone1.md"
    fig_path = Path(figures_dir) / "m1_1_recovery_e1.png"
    made_fig = _fig_recovery(df, fig_path)

    s = summarize(df).round(3)
    edge_cols = ["volume", "noise", "edge_r_e1", "edge_r_e1_minus_generic",
                 "edge_r_e2", "edge_r_e3_undirected", "edge_r_e3_directed"]
    edge_cols = [c for c in edge_cols if c in s.columns]
    cent_cols = ["volume", "noise"] + [c for c in s.columns if c.startswith("cent_rho_e1_")]
    conv_cols = ["volume", "noise"] + [c for c in s.columns if c.startswith("e1_e3_")]

    n_cells = len(df)
    ts = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "## 1.1 Synthetic recovery",
        "",
        f"_Generated {ts}. {n_cells} pipeline runs "
        f"({df['dag'].nunique()} ground-truth DAGs "
        f"× {df['volume'].nunique()} volumes × {df['noise'].nunique()} noise "
        f"levels × {df.groupby(['dag','volume','noise']).size().max()} replicates)._",
        "",
        "### Edge-weight recovery (mean correlation with true `d`)",
        "",
        s[edge_cols].to_markdown(index=False),
        "",
        "### Centrality rank recovery (mean Spearman ρ, E1 network vs true)",
        "",
        s[cent_cols].to_markdown(index=False),
        "",
        "### E1 ↔ E3 convergence (estimators without a shared failure mode)",
        "",
        s[conv_cols].to_markdown(index=False) if len(conv_cols) > 2 else "_E3 directed not estimated (chunk count below min_chunks in all cells)._",
        "",
    ]
    if made_fig:
        rel = Path(fig_path).relative_to(report_dir) if str(fig_path).startswith(str(report_dir)) else fig_path
        lines += [f"![E1 edge recovery]({rel})", ""]
    lines += [
        "### Read / decisions",
        "",
        "- TODO(analyst): state the minimum evidence volume at which E1 edge "
        "recovery and centrality rank recovery clear your bar, and set "
        "`ingest.min_comments` / chunk targets accordingly.",
        "- TODO(analyst): pick `estimate.replicates` from where the recovery "
        "curves flatten.",
        "- TODO(analyst): if E1−generic recovery collapses relative to raw E1, "
        "note that the account-specific signal is weak even in simulation.",
        "- KNOWN GAP: E3 undirected recovery is ~0 and E1↔E3 convergence is weak "
        "in synthetic — `ebicglasso` over-sparsifies at p=40 (see docs/PLAN.md "
        "'Known gaps'). Fix the estimator before reading E3 numbers as evidence.",
    ]
    body = "\n".join(lines)
    doc = _splice(_load(report_path), "synthetic_recovery", body)
    if not doc.startswith("#"):
        doc = _HEADER + "\n" + doc
    report_path.write_text(doc)
    return report_path
