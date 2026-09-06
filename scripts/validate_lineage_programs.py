#!/usr/bin/env python3
"""Validate the frozen CD4 lineage scorer on an external reference NPZ.

The NPZ must contain ``expression`` (cells×genes), ``gene_names``, ``labels``
and ``donors``.  D2 data are intentionally not accepted by this command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cd4perturb.state_d2 import load_lineage_programs, score_programs, validate_program_scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="external reference NPZ")
    parser.add_argument("--programs", default="config/th_lineage_programs.v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = np.load(args.input, allow_pickle=False)
    required = {"expression", "gene_names", "labels", "donors"}
    if not required.issubset(payload.files):
        raise ValueError(f"external scorer input missing: {sorted(required.difference(payload.files))}")
    programs = load_lineage_programs(args.programs)
    scores = score_programs(payload["expression"], payload["gene_names"].astype(str), programs)
    identity = {name: scores[name] for name in programs["identity_programs"]}
    report = validate_program_scores(identity, payload["labels"].astype(str), payload["donors"].astype(str))
    report.update({"version": "th_lineage_program_validation.v1", "source_npz": str(args.input),
                   "external_only": True, "macro_auc_minimum": programs["validation"]["macro_auc_minimum"],
                   "pass": bool(report.get("macro_auc") is not None and
                                  report["macro_auc"] >= programs["validation"]["macro_auc_minimum"])})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target), "macro_auc": report["macro_auc"], "pass": report["pass"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

