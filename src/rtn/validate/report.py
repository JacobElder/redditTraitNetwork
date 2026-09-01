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
