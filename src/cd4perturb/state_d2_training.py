from __future__ import annotations

"""Fair Scratch/Transfer training loop for the D2 STATE single-step task."""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class TrainingContract:
    mode: str
    seed: int
    gene_order_hash: str
    perturbation_vocab_hash: str
    split_hash: str
    model_config_hash: str
    batch_size: int = 1
    max_steps: int = 40000
    validation_every: int = 500
    early_stop_patience: int = 8
    selection_metric: str = "validation_mmd"
    d2_responses_used: bool = True

    def __post_init__(self):
        if self.mode not in {"Scratch", "Transfer"}:
            raise ValueError("training mode must be Scratch or Transfer")
        if self.max_steps != 40000:
            raise ValueError("D2 training budget is frozen at 40,000 steps")
        if self.selection_metric != "validation_mmd":
            raise ValueError("D2 checkpoint selection must use validation MMD")

    def as_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


def freeze_training_contract(panel: Mapping, vocab: Mapping, splits: Mapping,
                             model_config: Mapping, mode: str, seed: int) -> TrainingContract:
    """Construct a contract and reject any hidden train/transfer mismatch."""
    gene_hash = panel.get("gene_order_hash")
    vocab_hash = vocab.get("vocab_hash")
    split_hash = splits.get("split_hash")
    if not all(isinstance(value, str) and value for value in (gene_hash, vocab_hash, split_hash)):
        raise ValueError("panel, vocabulary and split hashes are required")
    if len(panel.get("gene_order", [])) != 2000:
        raise ValueError("training panel must contain exactly 2,000 genes")
    return TrainingContract(mode=mode, seed=int(seed), gene_order_hash=gene_hash,
                            perturbation_vocab_hash=vocab_hash, split_hash=split_hash,
                            model_config_hash=model_config.get("config_hash") or
                            hashlib.sha256(json.dumps(model_config, sort_keys=True).encode()).hexdigest())


def set_mmd(prediction, target):
    """A bounded set MMD proxy used for validation checkpoint selection."""
    torch = __import__("torch")
    if prediction.shape != target.shape:
        raise ValueError("MMD inputs must have equal [batch,set,gene] shape")
    # Mean and variance terms avoid an O(S²) distance matrix while preserving
    # the distributional nature of the set objective.
    mean_term = (prediction.mean(dim=1) - target.mean(dim=1)).pow(2).mean()
    var_term = (prediction.var(dim=1, unbiased=False) - target.var(dim=1, unbiased=False)).pow(2).mean()
    result = mean_term + var_term
    if not torch.isfinite(result):
        raise FloatingPointError("MMD is NaN/Inf")
    return result


def _set_phase(model, phase: int):
    for name, parameter in model.named_parameters():
        if "transformer" in name:
            parameter.requires_grad = phase >= 2
        else:
            parameter.requires_grad = True


def train_two_phase(model, train_batches: Iterable[Mapping], validation_batches: Iterable[Mapping],
                    contract: TrainingContract, output_dir: str | Path,
                    phase1_steps: int = 1000, validation_steps: int | None = None,
                    learning_rate: float = 1e-3, finetune_learning_rate: float = 2e-4,
                    log_fn: Callable[[Mapping], None] | None = None) -> dict:
    """Run the identical bounded loop for Scratch and Transfer.

    ``train_batches`` and ``validation_batches`` are factories or re-iterable
    streams yielding ``expression``, ``perturbation`` and ``target`` tensors.
    The function is deliberately agnostic to the source of D2 rows so the
    same loader can be used for both modes.
    """
    torch = __import__("torch")
    if phase1_steps <= 0 or phase1_steps >= contract.max_steps:
        raise ValueError("phase1_steps must leave steps for phase 2")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(contract.seed)
    _set_phase(model, 1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate,
                                  weight_decay=5e-4)
    train_iter = iter(train_batches)
    best = float("inf")
    best_step = 0
    patience = 0
    history = []
    validation_steps = validation_steps or contract.validation_every

    def evaluate(step: int) -> float:
        model.eval()
        values = []
        with torch.no_grad():
            for index, batch in enumerate(validation_batches):
                prediction = model(batch["expression"], batch["perturbation"])
                values.append(float(set_mmd(prediction, batch["target"]).detach().cpu()))
                if index + 1 >= validation_steps:
                    break
        model.train()
        if not values:
            raise ValueError("validation stream is empty")
        return float(sum(values) / len(values))

    for step in range(1, contract.max_steps + 1):
        if step == phase1_steps + 1:
            _set_phase(model, 2)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=finetune_learning_rate, weight_decay=5e-4)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_batches)
            batch = next(train_iter)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["expression"], batch["perturbation"])
        loss = set_mmd(prediction, batch["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % contract.validation_every == 0 or step == contract.max_steps:
            validation_mmd = evaluate(step)
            record = {"step": step, "train_mmd": float(loss.detach().cpu()),
                      "validation_mmd": validation_mmd, "phase": 1 if step <= phase1_steps else 2}
            history.append(record)
            if log_fn:
                log_fn(record)
            if validation_mmd < best:
                best = validation_mmd
                best_step = step
                patience = 0
                torch.save({"state_dict": model.state_dict(), "contract": contract.as_dict(),
                            "best_validation_mmd": best, "step": step}, target_dir / "best.ckpt")
            else:
                patience += 1
                if patience >= contract.early_stop_patience:
                    break
    result = {"version": "d2_state_training_result.v1", "contract": contract.as_dict(),
              "contract_hash": contract.hash(), "best_validation_mmd": best,
              "best_step": best_step, "stopped_step": history[-1]["step"] if history else 0,
              "history": history, "checkpoint": str(target_dir / "best.ckpt"),
              "d2_responses_used": True}
    (target_dir / "training_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

