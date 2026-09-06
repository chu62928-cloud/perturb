from __future__ import annotations

import pytest

from cd4perturb.state_d2_training import (
    freeze_training_contract,
    train_two_phase,
    validate_frozen_training_contract,
)


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
