from __future__ import annotations

import pytest
import numpy as np

from cd4perturb.state_d2_data import _normalize_panel_counts, select_pilot_perturbations
from cd4perturb.state_d2_model import D2StateConfig, train_d2_pilot
from cd4perturb.state_d2_training import (
    PrefetchBatchIterator,
    restore_state_output,
    freeze_training_contract,
    train_two_phase,
    validate_frozen_training_contract,
)


def test_prefetch_iterator_preserves_order_and_propagates_errors():
    produced = []

    def source():
        for value in range(6):
            produced.append(value)
            yield value

    iterator = PrefetchBatchIterator(source(), max_prefetch=2)
    try:
        assert [next(iterator) for _ in range(6)] == list(range(6))
        with pytest.raises(StopIteration):
            next(iterator)
    finally:
        iterator.close()
    assert produced == list(range(6))

    def failing_source():
        yield "first"
        raise RuntimeError("prefetch failure")

    iterator = PrefetchBatchIterator(failing_source(), max_prefetch=2)
    try:
        assert next(iterator) == "first"
        with pytest.raises(RuntimeError, match="prefetch failure"):
            next(iterator)
    finally:
        iterator.close()


def _artifacts():
    panel = {"gene_order": [f"G{i}" for i in range(2000)], "gene_order_hash": "g" * 64}
    vocab = {"perturbation_names": ["NTC", "G1"], "vocab_hash": "v" * 64}
    splits = {"split_hash": "s" * 64}
    model = {
        "architecture": "official_state_adapter",
        "checkpoint_hparams_hash": "c" * 64,
        "n_genes": 2000,
        "n_perturbations": 2,
        "cell_set_len": 32,
        "hidden_dim": 328,
        "transformer_layers": 8,
        "attention_heads": 12,
        "batch_size": 64,
        "head_learning_rate": 1e-3,
        "backbone_learning_rate": 2e-4,
        "phase1_steps": 1000,
    }
    return panel, vocab, splits, model


def test_frozen_contract_rejects_runtime_model_drift():
    panel, vocab, splits, model = _artifacts()
    contract = freeze_training_contract(panel, vocab, splits, model, "Transfer", 20260901)

    assert contract.hidden_dim == 328
    assert contract.transformer_layers == 8
    assert contract.checkpoint_hparams_hash == "c" * 64

    changed = dict(model, hidden_dim=256)
    with pytest.raises(ValueError, match="frozen training contract mismatch"):
        validate_frozen_training_contract(contract.as_dict(), panel, vocab, splits, changed,
                                          "Transfer", 20260901)


def test_pilot_selection_uses_only_supported_training_backgrounds():
    counts = {
        "Rest": {"P1": 40, "P2": 50, "A": 100, "B": 80, "C": 500},
        "Stim8hr": {"P1": 42, "P2": 51, "A": 90, "B": 79, "C": 500},
        "Stim48hr": {"P1": 44, "P2": 52, "A": 95, "B": 78, "C": 500},
    }
    records = []
    for gene in ("P1", "P2", "A", "B", "C"):
        for condition in counts:
            split = "test" if gene == "C" and condition != "Rest" else "train"
            records.append({"perturbation_name": gene, "condition": condition, "split": split})
    manifest = select_pilot_perturbations(
        counts, {"records": records, "split_hash": "s" * 64},
        ["NTC", "P1", "P2", "A", "B", "C"], ["P1", "P2"], n_targets=4,
    )
    assert manifest["selected_perturbations"] == ["P1", "P2", "A", "B"]
    assert manifest["selection"]["A"]["minimum_training_cell_count"] == 90
    assert "C" not in manifest["selected_perturbations"]


def test_panel_normalization_uses_all_gene_library_size():
    panel = np.asarray([[10.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    normalized = _normalize_panel_counts(panel, np.asarray([100.0, 100.0], dtype=np.float32))
    expected = np.log1p(panel / 100.0 * 10000.0)
    np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)
    panel_only_denominator = np.log1p(panel / np.maximum(panel.sum(axis=1), 1.0)[:, None] * 10000.0)
    assert not np.allclose(normalized, panel_only_denominator)


def test_pilot_training_reports_loss_descent_and_reload(tmp_path):
    torch = pytest.importorskip("torch")

    class PilotStream:
        def __iter__(self):
            while True:
                yield {
                    "expression": torch.ones(2, 2, 2),
                    "perturbation": torch.tensor([[[0., 1., 0.]] * 2,
                                                   [[0., 0., 1.]] * 2]),
                    "target": torch.zeros(2, 2, 2),
                    "perturbation_names": ["G1", "G2"],
                }

    config = D2StateConfig(n_genes=2, n_perturbations=3, cell_set_len=2,
                           hidden_dim=8, n_encoder_layers=1, n_decoder_layers=1,
                           n_attention_heads=2, dropout=0.0)
    result = train_d2_pilot(config, PilotStream(), ["G1", "G2"], tmp_path,
                            steps=20, device="cpu", comparison_window=5)
    assert result["loss_descent"] is True
    assert result["covered_all_perturbations"] is True
    assert result["checkpoint_reload"] is True


def test_official_output_restore_requires_exact_set_elements():
    torch = pytest.importorskip("torch")
    flattened = torch.zeros(1, 8, 2)
    restored = restore_state_output(flattened, batch_size=2, set_len=4, n_genes=2)
    assert tuple(restored.shape) == (2, 4, 2)
    with pytest.raises(ValueError, match="official STATE output shape"):
        restore_state_output(torch.zeros(1, 7, 2), batch_size=2, set_len=4, n_genes=2)


def test_scratch_updates_backbone_while_transfer_phase_one_freezes_it(tmp_path):
    torch = pytest.importorskip("torch")

    class TinyState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Linear(2, 2, bias=False)
            self.head = torch.nn.Linear(2, 2, bias=False)

        def forward(self, expression, perturbation):
            return self.head(self.transformer(expression))

    class FixedStream:
        def __iter__(self):
            while True:
                yield {
                    "expression": torch.ones(1, 2, 2),
                    "perturbation": torch.zeros(1, 2, 2),
                    "target": torch.zeros(1, 2, 2),
                }

    panel, vocab, splits, model_config = _artifacts()
    for mode in ("Scratch", "Transfer"):
        model = TinyState()
        before_backbone = model.transformer.weight.detach().clone()
        before_head = model.head.weight.detach().clone()
        contract = freeze_training_contract(panel, vocab, splits, model_config, mode, 20260901)
        train_two_phase(model, FixedStream(), FixedStream(), contract, tmp_path / mode,
                        validation_steps=1, stop_after_steps=1)
        backbone_changed = not torch.equal(before_backbone, model.transformer.weight)
        head_changed = not torch.equal(before_head, model.head.weight)
        assert backbone_changed is (mode == "Scratch")
        assert head_changed


def test_resume_requires_same_contract_and_continues_from_last_step(tmp_path):
    torch = pytest.importorskip("torch")

    class TinyState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Linear(2, 2, bias=False)
            self.head = torch.nn.Linear(2, 2, bias=False)

        def forward(self, expression, perturbation):
            return self.head(self.transformer(expression))

    class FixedStream:
        def __iter__(self):
            while True:
                yield {
                    "expression": torch.ones(1, 2, 2),
                    "perturbation": torch.zeros(1, 2, 2),
                    "target": torch.zeros(1, 2, 2),
                }

    panel, vocab, splits, model_config = _artifacts()
    output = tmp_path / "resume"
    contract = freeze_training_contract(panel, vocab, splits, model_config, "Scratch", 20260901)
    train_two_phase(TinyState(), FixedStream(), FixedStream(), contract, output,
                    validation_steps=1, stop_after_steps=2)

    resumed = train_two_phase(TinyState(), FixedStream(), FixedStream(), contract, output,
                              validation_steps=1, stop_after_steps=3, resume=True)
    assert resumed["resumed_from_step"] == 2
    assert resumed["stopped_step"] == 3

    changed = freeze_training_contract(panel, vocab, splits, model_config, "Scratch", 20260902)
    with pytest.raises(ValueError, match="resume checkpoint contract mismatch"):
        train_two_phase(TinyState(), FixedStream(), FixedStream(), changed, output,
                        validation_steps=1, stop_after_steps=3, resume=True)
