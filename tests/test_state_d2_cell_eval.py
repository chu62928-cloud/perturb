import numpy as np
import pytest

from cd4perturb.state_d2_cell_eval import (
    macro_mean,
    paper_pearson_delta,
    paper_pds_l1,
    summarize_context_metrics,
)


def _fixture():
    genes = ["A", "B", "C"]
    perts = ["A", "B"]
    control = np.zeros((2, 3), dtype=float)
    real = np.asarray([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    return real, real.copy(), control, control.copy(), genes, perts


def test_perfect_prediction_has_perfect_pds_and_delta_pearson():
    real, pred, real_ctrl, pred_ctrl, genes, perts = _fixture()
    pds = paper_pds_l1(real, pred, real_ctrl, pred_ctrl, genes, perts)
    pearson = paper_pearson_delta(real, pred, real_ctrl, pred_ctrl, genes, perts)
    assert pds == {"A": pytest.approx(1.0), "B": pytest.approx(1.0)}
    assert pearson == {"A": pytest.approx(1.0), "B": pytest.approx(1.0)}


def test_pds_excludes_target_gene_and_label_permutation_is_not_perfect():
    real, _pred, real_ctrl, _pred_ctrl, genes, perts = _fixture()
    # Swapping the two predicted effects is not rescued by the target-gene
    # exclusion because the remaining response fingerprints differ.
    pred = real[[1, 0]]
    pred_ctrl = real_ctrl.copy()
    pds = paper_pds_l1(real, pred, real_ctrl, pred_ctrl, genes, perts)
    assert pds["A"] < 1.0
    assert pds["B"] < 1.0


def test_constant_delta_is_nan_not_a_fake_perfect_score():
    genes = ["A", "B", "C"]
    perts = ["A"]
    real = np.asarray([[1.0, 1.0, 1.0]])
    pred = np.asarray([[2.0, 2.0, 2.0]])
    control = np.zeros((1, 3))
    result = paper_pearson_delta(real, pred, control, control, genes, perts)
    assert np.isnan(result["A"])


def test_macro_mean_does_not_turn_nan_into_zero():
    assert macro_mean({"a": 1.0, "b": float("nan")}) == pytest.approx(1.0)
    assert np.isnan(macro_mean({"a": float("nan")}))


def test_context_summary_reports_finite_counts():
    summary = summarize_context_metrics(
        "Rest",
        [{"perturbation": "A", "pearson_delta": 1.0},
         {"perturbation": "B", "pearson_delta": None}],
        ["pearson_delta"],
    )
    assert summary.condition == "Rest"
    assert summary.n_perturbations == 2
    assert summary.finite_counts == {"pearson_delta": 1}
    assert summary.macro_metrics["pearson_delta"] == pytest.approx(1.0)


def test_absolute_pds_and_signed_delta_separate_opposite_direction():
    genes = ["A", "B", "C"]
    perts = ["A", "B"]
    control = np.zeros((2, 3), dtype=float)
    real = np.asarray([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    opposite = -real
    pds = paper_pds_l1(real, opposite, control, control, genes, perts)
    signed = paper_pearson_delta(real, opposite, control, control, genes, perts)
    # Paper-era PDS is deliberately absolute; the operational Pearson Delta
    # remains signed and therefore catches the direction reversal.
    assert pds["A"] == pytest.approx(1.0)
    assert pds["B"] == pytest.approx(1.0)
    assert signed["A"] == pytest.approx(-1.0)
    assert signed["B"] == pytest.approx(-1.0)


def test_condition_mean_like_zero_effect_does_not_get_fake_perfect_pds():
    genes = ["A", "B", "C", "D"]
    perts = ["A", "B", "C"]
    control = np.zeros((3, 4), dtype=float)
    real = np.asarray([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 4.0]])
    predicted = np.zeros_like(real)
    pds = paper_pds_l1(real, predicted, control, control, genes, perts)
    assert any(value < 1.0 for value in pds.values())


def test_target_exclusion_only_applies_to_a_measured_target_gene():
    genes = ["A", "B", "C"]
    perts = ["NOT_IN_PANEL"]
    real = np.asarray([[0.0, 2.0, 0.0]])
    pred = real.copy()
    control = np.zeros_like(real)
    result = paper_pds_l1(real, pred, control, control, genes, perts)
    assert result["NOT_IN_PANEL"] == pytest.approx(1.0)
