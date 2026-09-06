from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from cd4perturb.config import load_config
from cd4perturb.contracts import TargetContract
from cd4perturb.metrics import (biological_gate, bootstrap_rmse_improvement, distribution_metrics,
                                summarize, hierarchical_bootstrap, model_global_gate, model_state_gate)
from cd4perturb.planner import beam_search
from cd4perturb.registry import build_registry
from cd4perturb.selection import stratified_pilot, make_pilot_selection
from cd4perturb.support import pass_support, support_score
from cd4perturb.composition import composition_gate, match_intermediate_states, seal_match, score_composition
from cd4perturb.ensemble import model_disagreement, robust_rank
from cd4perturb.holdouts import (gene_holdout, interaction_holdout, state_holdout,
                                 continuous_state_regions, validate_no_region_overlap)
from cd4perturb.release import create_locked_release
from cd4perturb.data_contract import validate_pilot_contract
from cd4perturb.guide_correction import correct_guides
from cd4perturb.data import build_d1_effect_matrix


def test_config_and_contract(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"runtime_root": str(tmp_path / "runtime")}), encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.pilot.n_genes == 96
    assert len(cfg.hash()) == 64
    frozen = TargetContract().freeze("v0.1")
    assert frozen.frozen and frozen.version == "v0.1"


def test_pilot_is_deterministic_and_labelled():
    rows = [{"gene": f"G{i:03d}", "expression_bin": str(i % 4), "guide_coverage": 10 - i % 5,
             "effect_bin": ["strong", "mid", "weak"][i % 3],
             "guide_concordance": "discordant" if i % 7 == 0 else "concordant"} for i in range(140)]
    selected = stratified_pilot(rows, n_genes=12, core_genes=8, challenge_genes=4)
    assert len(selected) == 12
    assert len({x.gene for x in selected}) == 12
    assert {x.tier for x in selected} == {"core", "challenge"}


def test_metrics_gate_and_distribution():
    y = np.array([[1., 0.], [2., 1.], [3., 1.]])
    pred = y * 0.98
    base = y * 0.7
    summary = summarize(y, pred)
    assert summary.n == 6
    boot = bootstrap_rmse_improvement(y, pred, base, n_boot=50, seed=1)
    assert boot["median"] > 0
    dist = distribution_metrics(y, pred)
    assert dist["sliced_wasserstein"] >= 0
    ok, reasons = biological_gate({"relative_rmse_improvement": .08, "bootstrap_ci_lower": .01,
                                   "program_direction_drop": 0, "critical_program_error_ratio": 1,
                                   "max_condition_rmse_ratio": 1, "ntc_zero_effect_error": 0,
                                   "ntc_null_p95": .1}, {"rmse_improvement": .05, "bootstrap_ci_lower": 0,
                                   "program_direction_margin": .02, "critical_program_tolerance": .05,
                                   "condition_rmse_tolerance": .1})
    assert ok and not reasons


def test_support_and_planner():
    train = np.eye(30, 3)
    score = support_score(np.array([[0., 0., 0.]]), train, k=20)
    assert score["effective_cells"] == 20
    assert pass_support({"nearest_distance": 0.5, "local_density": 1, "effective_cells": 20},
                        {"nearest_distance_p95": 1, "density_p05": .1, "min_effective_cells": 20})[0]
    out = beam_search(["A", "B", "C"], lambda a, b: {"target_improvement": 1.0 if a == "A" else .5,
        "source_decay": .1, "uncertainty": .1, "model_disagreement": .1, "ood_distance": .1,
        "intermediate_support": .9, "stress": .1, "apoptosis": .1, "general_activation": .1,
        "donor_stability": .8})
    assert len(out) == 6 and out[0].first != out[0].second


def test_holdouts_composition_and_ensemble():
    assert len(gene_holdout(["A", "B", "C", "D"]).test) == 1
    assert state_holdout(["s1", "s2"]).test
    assert interaction_holdout([("A", "s1"), ("B", "s2"), ("C", "s1")]).test
    ok, reasons = composition_gate({"second_step_rmse_ratio": .9, "error_amplification": .05,
        "program_direction_drop": .01, "support_pass_rate": .9})
    assert ok and not reasons
    d = model_disagreement({"STATE": np.array([1., -1.]), "STACK": np.array([.8, .5])})
    assert d["direction_conflict_fraction"] == .5
    assert robust_rank({"A": np.array([.9, .1]), "B": np.array([.8, .2])}).tolist() == [0., 1.]


def test_locked_release_is_immutable(tmp_path: Path):
    registry = tmp_path / "registry.json"
    rows = [{"donor_id": "D2", "status": "complete", "dataset_id": x,
             "fingerprint": x, "role": "provisional_external"} for x in ["D2_Rest", "D2_Stim8hr", "D2_Stim48hr"]]
    registry.write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "release.json"
    manifest = create_locked_release(registry, "cfg", {}, out)
    assert manifest["release_type"] == "D2_locked_eval"
    try:
        create_locked_release(registry, "cfg", {}, out)
    except FileExistsError:
        pass
    else:
        raise AssertionError("locked release must not be overwritten")


def test_primary_challenge_are_disjoint_and_primary_ignores_response():
    rows = [{"gene": f"G{i:03d}", "expression_bin": str(i % 4), "guide_coverage": 100 - i,
             "effect_bin": "strong", "guide_concordance": "discordant",
             "stimulus_dependent": True} for i in range(120)]
    selection = make_pilot_selection(rows, 64, 32)
    assert len(selection.primary) == 64 and len(selection.challenge) == 32
    assert not ({x.gene for x in selection.primary} & {x.gene for x in selection.challenge})
    assert all(x.effect_bin == "unknown" and not x.stimulus_dependent for x in selection.primary)


def test_continuous_state_regions_have_buffered_folds():
    rng = np.random.default_rng(3)
    latent = np.vstack([rng.normal(0, 1, (120, 3)), rng.normal(4, 1, (120, 3))])
    split = continuous_state_regions(latent, ["Rest"] * 120 + ["Stim"] * 120,
                                     n_anchors=10, k_neighbors=20, n_folds=5, seed=3)
    assert len(split.regions) == 20
    assert split.state_unit == "continuous_knn_ball"
    assert validate_no_region_overlap(split)
    assert all(np.all(m["test"] <= m["buffer"]) for m in split.fold_masks)


def test_hierarchical_bootstrap_and_split_gates():
    values = np.arange(24, dtype=float)
    gene = np.repeat(["A", "B", "C"], 8)
    guide = np.tile(np.repeat(["g1", "g2"], 4), 3)
    cell = np.arange(24)
    boot = hierarchical_bootstrap(values, [gene, guide, cell], n_boot=30, seed=4)
    assert boot["scope"] == "D1_technical_state_uncertainty"
    ok, _ = model_global_gate({"evaluation_scope": "primary_64", "relative_standardized_rmse_improvement": .08,
        "bootstrap_ci_lower": .01, "program_direction_drop": 0, "critical_program_error_ratio": 1,
        "max_condition_rmse_ratio": 1, "ntc_zero_effect_error": 0, "ntc_null_p95": .1}, {})
    assert ok
    ok, reasons = model_state_gate({"evaluation_scope": "primary_64", "relative_standardized_rmse_improvement": .08,
        "bootstrap_ci_lower": .01, "local_direction_drop": 0, "distribution_metrics_improved": 2,
        "state_proportion_error_ratio": 1, "max_region_rmse_ratio": 1}, {})
    assert ok and not reasons


def test_composition_match_must_be_sealed_before_scoring():
    pred = np.array([[0., 0.], [1., 1.]])
    real = np.array([[0., 0.], [1., 1.], [5., 5.]])
    manifest = match_intermediate_states(pred, real, ["Rest", "Rest", "Rest"],
                                         {"max_distance": .2, "min_density": 0, "min_effective_cells": 1})
    with pytest.raises(RuntimeError):
        score_composition(manifest, {0: 1., 1: 2.})
    result = score_composition(seal_match(manifest), {0: 1., 1: 2.})
    assert result["n_valid"] == 2


def test_target_contract_is_multiobjective_and_safe():
    contract = TargetContract()
    score = contract.score_multiobjective({"Th17": 2, "Th2": -1, "stress": .1,
                                           "apoptosis": .1, "generic_activation": .1})
    assert score > 0
    assert contract.meets_constraints({"Th17": .2, "Th2": -.1, "stress": .01,
                                      "apoptosis": .01, "generic_activation": .01},
                                     {"stress": .1, "apoptosis": .1, "generic_activation": .1})


def test_data_contract_rejects_primary_response_leak(tmp_path: Path):
    primary = tmp_path / "primary.json"
    challenge = tmp_path / "challenge.json"
    registry = tmp_path / "registry.json"
    primary.write_text(json.dumps([{"gene": f"G{i}", "tier": "primary_64",
                                    "effect_bin": "unknown", "guide_concordance": "unknown",
                                    "stimulus_dependent": False} for i in range(64)]))
    challenge.write_text(json.dumps([{"gene": f"C{i}", "tier": "challenge_32"} for i in range(32)]))
    registry.write_text(json.dumps([{"donor_id": "D1", "status": "complete"}] * 3))
    assert validate_pilot_contract(primary, challenge, registry)["status"] == "DATA_CONTRACT_READY"
    leaked = json.loads(primary.read_text()); leaked[0]["effect_bin"] = "strong"
    primary.write_text(json.dumps(leaked))
    assert not validate_pilot_contract(primary, challenge, registry)["valid"]


def test_data_contract_evidence_directory_cannot_be_missing(tmp_path: Path):
    primary = tmp_path / "primary.json"
    challenge = tmp_path / "challenge.json"
    registry = tmp_path / "registry.json"
    primary.write_text(json.dumps([{"gene": f"G{i}", "tier": "primary_64",
                                    "effect_bin": "unknown", "guide_concordance": "unknown",
                                    "stimulus_dependent": False, "n_guides": 2,
                                    "guide_coverage": 200} for i in range(64)]))
    challenge.write_text(json.dumps([{"gene": f"C{i}", "tier": "challenge_32"} for i in range(32)]))
    registry.write_text(json.dumps([{"dataset_id": x, "donor_id": "D1", "status": "complete",
                                     "expected_size": 1, "observed_size": 1, "fingerprint": "x",
                                     "shape": [1, 1]} for x in ["D1_Rest", "D1_Stim8hr", "D1_Stim48hr"]]))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    result = validate_pilot_contract(primary, challenge, registry, evidence)
    assert result["status"] == "DATA_CONTRACT_READY"
    assert not result["valid"]
    assert any("guide QC" in reason for reason in result["reasons"])


def test_guide_correction_preserves_labels_and_rejects_conflicts():
    library = [{"sgRNA": "g1", "target_gene_id": "ENSG1", "target_gene_name": "A", "flag": False},
               {"sgRNA": "g2", "target_gene_id": "ENSG2", "target_gene_name": "B", "flag": False},
               {"sgRNA": "ntc1", "target_gene_id": "", "target_gene_name": "", "flag": False}]
    rows = [{"guide_id": "g1", "guide_type": "target", "low_quality": False},
            {"guide_id": "ntc1", "guide_type": "NTC", "low_quality": False},
            {"guide_id": "g2", "guide_type": "NTC", "low_quality": False}]
    out = correct_guides(rows, library)
    assert out[0]["original_guide_id"] == "g1" and out[0]["corrected_target_gene_id"] == "ENSG1"
    assert out[1]["is_ntc"] and out[1]["eligible"]
    assert "ntc_type_has_target_mapping" in out[2]["conflict_reason"]


def test_effect_matrix_rejects_d2_before_opening():
    with pytest.raises(ValueError, match="D1 files only"):
        build_d1_effect_matrix(["/tmp/D2_Rest.assigned_guide.h5ad"], ["A"])


def test_effect_matrix_fixture_uses_ids_ntc_and_bounded_blocks(tmp_path: Path):
    ad = pytest.importorskip("anndata")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    pandas = pytest.importorskip("pandas")

    def make(path: Path, target_scale: float):
        # Gene names intentionally differ from fixed IDs.  A correct reader
        # must align by var.gene_ids and must not use raw gene names.
        counts = np.array([
            [10, 10], [10, 10], [10, 10], [10, 10],  # condition NTC
            [20 * target_scale, 10], [20 * target_scale, 10],
            [20 * target_scale, 10], [20 * target_scale, 10],
            [10, 20], [10, 20], [10, 20], [10, 20],
        ], dtype=np.float32)
        obs = pandas.DataFrame({
            "guide_id": pandas.Categorical(["ntc"] * 4 + ["g1"] * 4 + ["g2"] * 4),
            "guide_group": pandas.Categorical(["targeting single sgRNA"] * 12),
            "guide_type": pandas.Categorical(["non-targeting"] * 4 + ["targeting"] * 8),
            "low_quality": np.zeros(12, dtype=bool),
            "perturbed_gene_id": pandas.Categorical([""] * 4 + ["ENSG_A"] * 4 + ["ENSG_B"] * 4),
            "perturbed_gene_name": pandas.Categorical([""] * 4 + ["WrongName"] * 4 + ["B"] * 4),
            "total_counts": counts.sum(axis=1),
        })
        obj = ad.AnnData(scipy_sparse.csr_matrix(counts), obs=obs)
        obj.var["gene_ids"] = ["ENSG_A", "ENSG_B"]
        obj.var["gene_name"] = ["A", "B"]
        obj.write_h5ad(path)

    first = tmp_path / "D1_Rest.assigned_guide.h5ad"
    second = tmp_path / "D1_Stim8hr.assigned_guide.h5ad"
    make(first, 2.0)
    make(second, 3.0)
    qc = {"d2_responses_used": False, "guide_rows": [
        {"original_guide_id": "ntc", "corrected_target_gene_id": "",
         "corrected_target_gene_name": "", "guide_group": "targeting single sgRNA",
         "guide_type": "non-targeting", "low_quality": False, "is_ntc": True,
         "is_targeting": False, "eligible": True},
        {"original_guide_id": "g1", "corrected_target_gene_id": "ENSG_A",
         "corrected_target_gene_name": "A", "guide_group": "targeting single sgRNA",
         "guide_type": "targeting", "low_quality": False, "is_ntc": False,
         "is_targeting": True, "eligible": True},
        {"original_guide_id": "g2", "corrected_target_gene_id": "ENSG_B",
         "corrected_target_gene_name": "B", "guide_group": "targeting single sgRNA",
         "guide_type": "targeting", "low_quality": False, "is_ntc": False,
         "is_targeting": True, "eligible": True},
    ]}
    result = build_d1_effect_matrix([first, second], ["ENSG_A", "ENSG_B"],
                                    min_cells_per_guide=2, guide_qc=qc, block_rows=3)
    assert result["normalization"] == "log1p(CP10K)"
    assert result["condition_metadata"]["D1_Rest"]["library_size_field"] == "total_counts"
    assert {row["guide_id"] for row in result["guide_effects"]} == {"g1", "g2"}
    assert all(len(row["effect"]) == 2 for row in result["guide_effects"])
    assert result["gene_order"] == ["ENSG_A", "ENSG_B"]
    assert result["d2_responses_used"] is False


def test_effect_matrix_requires_observed_total_counts(tmp_path: Path):
    ad = pytest.importorskip("anndata")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    pandas = pytest.importorskip("pandas")
    obs = pandas.DataFrame({
        "guide_id": pandas.Categorical(["ntc"]),
        "guide_group": pandas.Categorical(["targeting single sgRNA"]),
        "guide_type": pandas.Categorical(["non-targeting"]),
        "low_quality": [False], "perturbed_gene_id": pandas.Categorical([""]),
        "perturbed_gene_name": pandas.Categorical([""]),
    })
    obj = ad.AnnData(scipy_sparse.csr_matrix([[1.0]]), obs=obs)
    obj.var["gene_ids"] = ["ENSG_A"]
    path = tmp_path / "D1_Rest.assigned_guide.h5ad"
    obj.write_h5ad(path)
    with pytest.raises(ValueError, match="total_counts"):
        build_d1_effect_matrix([path], ["ENSG_A"], min_cells_per_guide=1)


def test_effect_matrix_rejects_corrupt_csr_before_scipy(tmp_path: Path):
    ad = pytest.importorskip("anndata")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    pandas = pytest.importorskip("pandas")
    h5py = pytest.importorskip("h5py")
    obs = pandas.DataFrame({
        "guide_id": pandas.Categorical(["ntc"]),
        "guide_group": pandas.Categorical(["targeting single sgRNA"]),
        "guide_type": pandas.Categorical(["non-targeting"]),
        "low_quality": [False], "perturbed_gene_id": pandas.Categorical([""]),
        "perturbed_gene_name": pandas.Categorical([""]), "total_counts": [1.0],
    })
    obj = ad.AnnData(scipy_sparse.csr_matrix([[1.0]]), obs=obs)
    obj.var["gene_ids"] = ["ENSG_A"]
    path = tmp_path / "D1_Rest.assigned_guide.h5ad"
    obj.write_h5ad(path)
    with h5py.File(path, "r+") as handle:
        # An interior decrease reproduces the real D1 Rest corruption, while
        # keeping the fixture tiny and deterministic.
        handle["X/indptr"][1] = 0
    with pytest.raises(ValueError, match="H5AD is corrupt"):
        build_d1_effect_matrix([path], ["ENSG_A"], min_cells_per_guide=1)


def test_continuous_state_regions_scale_without_pairwise_matrix():
    rng = np.random.default_rng(11)
    latent = rng.normal(size=(1500, 8))
    split = continuous_state_regions(latent, ["Rest"] * len(latent),
                                     n_anchors=10, k_neighbors=25, n_folds=5, seed=11)
    assert validate_no_region_overlap(split)
    test_cells = [int(x) for fold in split.fold_masks for x in np.flatnonzero(fold["test"])]
    assert len(test_cells) == len(set(test_cells))
