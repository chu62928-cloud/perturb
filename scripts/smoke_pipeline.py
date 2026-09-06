#!/usr/bin/env python
"""Fresh-process smoke for the non-deep pipeline interfaces."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from cd4perturb.baselines import fit_baseline_suite
from cd4perturb.metrics import bootstrap_rmse_improvement, summarize
from cd4perturb.planner import beam_search
from cd4perturb.support import pass_support, support_score


def main() -> int:
    rng = np.random.default_rng(7)
    control = rng.normal(size=(64, 8))
    truth = control + rng.normal(scale=.1, size=(64, 8))
    suite = fit_baseline_suite(control, truth)
    pred = suite["linear_residual_ridge"].prediction
    metrics = summarize(truth, pred)
    boot = bootstrap_rmse_improvement(truth, pred, suite["ntc_no_change"].prediction)
    support = support_score(pred[:4, :3], control[:, :3], k=20)
    support_ok, support_reasons = pass_support(support, {"nearest_distance_p95": 10, "density_p05": .01, "min_effective_cells": 20})
    candidates = beam_search(["A", "B", "C"], lambda a, b: {"target_improvement": 1, "source_decay": .1,
        "uncertainty": .1, "model_disagreement": .1, "ood_distance": .1, "intermediate_support": .9,
        "stress": .1, "apoptosis": .1, "general_activation": .1, "donor_stability": .8})
    payload = {"status": "PASS", "baseline": metrics.__dict__, "bootstrap": boot,
               "support": {**support, "pass": support_ok, "reasons": support_reasons},
               "planner_candidates": len(candidates)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
