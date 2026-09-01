"""Milestone 1.1 entrypoint.

    python -m scripts.run_synthetic_recovery \
        --config config/default.yaml --override config/synthetic.yaml

Writes the per-cell table to ``reports/figures/m1_1_recovery.parquet`` and the
"1.1 Synthetic recovery" section into ``reports/milestone1.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rtn.config import load_config
from rtn.validate import run_recovery
from rtn.validate.report import write_synthetic_recovery


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--out", default=None, help="parquet path for the per-cell table")
    args = ap.parse_args()

    cfg = load_config(args.config, *args.override)
    print(f"config hash {cfg.hash8} · backend {cfg.model['backend']}")

    df = run_recovery(cfg)

    report_dir = Path(cfg.get("report.dir", "reports"))
    figures_dir = Path(cfg.get("report.figures_dir", "reports/figures"))
    out = Path(args.out) if args.out else figures_dir / "m1_1_recovery.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    report_path = write_synthetic_recovery(df, report_dir, figures_dir)
    print(f"wrote {out} ({len(df)} cells)")
    print(f"wrote {report_path}")
    from rtn.validate.recovery import summarize

    print(summarize(df).to_string(index=False))


if __name__ == "__main__":
    main()
