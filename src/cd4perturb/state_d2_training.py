from __future__ import annotations

"""Fair Scratch/Transfer training loop for the D2 STATE single-step task."""

import hashlib
import json
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, Mapping


FORMAL_MODEL_FIELDS = {
    "architecture", "checkpoint_hparams_hash", "n_genes", "n_perturbations",
    "cell_set_len", "hidden_dim", "transformer_layers", "attention_heads",
    "batch_size", "head_learning_rate", "backbone_learning_rate", "phase1_steps",
}


def formal_model_config(checkpoint_payload: Mapping, n_genes: int, n_perturbations: int,
                        *, batch_size: int = 64, cell_set_len: int = 32,
                        head_learning_rate: float = 1e-3,
                        backbone_learning_rate: float = 2e-4,
                        phase1_steps: int = 1000) -> dict:
    """Describe the exact official architecture used by both formal modes."""
    hparams = dict(checkpoint_payload.get("hyper_parameters", {}))
    transformer = dict(hparams.get("transformer_backbone_kwargs", {}))
    checkpoint_hash = hashlib.sha256(json.dumps(hparams, sort_keys=True).encode()).hexdigest()
    config = {
        "architecture": "official_state_adapter",
        "checkpoint_hparams_hash": checkpoint_hash,
        "n_genes": int(n_genes),
        "n_perturbations": int(n_perturbations),
        "cell_set_len": int(cell_set_len),
        "hidden_dim": int(hparams.get("hidden_dim", transformer.get("hidden_size", 0))),
        "transformer_layers": int(transformer.get("num_hidden_layers", 0)),
        "attention_heads": int(transformer.get("num_attention_heads", 0)),
        "batch_size": int(batch_size),
        "head_learning_rate": float(head_learning_rate),
        "backbone_learning_rate": float(backbone_learning_rate),
        "phase1_steps": int(phase1_steps),
    }
    config["config_hash"] = hashlib.sha256(
        json.dumps({key: config[key] for key in sorted(FORMAL_MODEL_FIELDS)}, sort_keys=True).encode()
    ).hexdigest()
    return config


@dataclass(frozen=True)
class TrainingContract:
    mode: str
    seed: int
    gene_order_hash: str
    perturbation_vocab_hash: str
    split_hash: str
    model_config_hash: str
    architecture: str = "official_state_adapter"
    checkpoint_hparams_hash: str = ""
    n_genes: int = 2000
    n_perturbations: int = 0
    cell_set_len: int = 32
    hidden_dim: int = 0
    transformer_layers: int = 0
    attention_heads: int = 0
    batch_size: int = 64
    max_steps: int = 40000
    validation_every: int = 500
    early_stop_patience: int = 8
    selection_metric: str = "validation_mmd"
    head_learning_rate: float = 1e-3
    backbone_learning_rate: float = 2e-4
    phase1_steps: int = 1000
    d2_responses_used: bool = True

    def __post_init__(self):
        if self.mode not in {"Scratch", "Transfer"}:
            raise ValueError("training mode must be Scratch or Transfer")
        if self.batch_size <= 0:
            raise ValueError("training batch size must be positive")
        if self.max_steps != 40000:
            raise ValueError("D2 training budget is frozen at 40,000 steps")
        if self.selection_metric != "validation_mmd":
            raise ValueError("D2 checkpoint selection must use validation MMD")
        if self.architecture != "official_state_adapter":
            raise ValueError("D2 full training must use the official STATE adapter")
        if self.n_genes != 2000 or self.n_perturbations <= 0 or self.cell_set_len != 32:
            raise ValueError("D2 formal model dimensions are invalid")
        if min(self.hidden_dim, self.transformer_layers, self.attention_heads) <= 0:
            raise ValueError("D2 formal backbone dimensions are required")
        if not self.checkpoint_hparams_hash:
            raise ValueError("D2 checkpoint hyper-parameter hash is required")
        if not (0 < self.phase1_steps < self.max_steps):
            raise ValueError("D2 transfer phase-1 steps must leave a joint-training phase")
        if self.head_learning_rate <= 0 or self.backbone_learning_rate <= 0:
            raise ValueError("D2 learning rates must be positive")

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
    missing = sorted(FORMAL_MODEL_FIELDS.difference(model_config))
    if missing:
        raise ValueError(f"formal model config is missing fields: {missing}")
    normalized_model = {key: model_config[key] for key in sorted(FORMAL_MODEL_FIELDS)}
    model_hash = hashlib.sha256(json.dumps(normalized_model, sort_keys=True).encode()).hexdigest()
    supplied_hash = model_config.get("config_hash")
    if supplied_hash is not None and supplied_hash != model_hash:
        raise ValueError("formal model config hash does not match its fields")
    return TrainingContract(mode=mode, seed=int(seed), gene_order_hash=gene_hash,
                            perturbation_vocab_hash=vocab_hash, split_hash=split_hash,
                            model_config_hash=model_hash,
                            architecture=str(model_config["architecture"]),
                            checkpoint_hparams_hash=str(model_config["checkpoint_hparams_hash"]),
                            n_genes=int(model_config["n_genes"]),
                            n_perturbations=int(model_config["n_perturbations"]),
                            cell_set_len=int(model_config["cell_set_len"]),
                            hidden_dim=int(model_config["hidden_dim"]),
                            transformer_layers=int(model_config["transformer_layers"]),
                            attention_heads=int(model_config["attention_heads"]),
                            batch_size=int(model_config["batch_size"]),
                            head_learning_rate=float(model_config["head_learning_rate"]),
                            backbone_learning_rate=float(model_config["backbone_learning_rate"]),
                            phase1_steps=int(model_config["phase1_steps"]))


def validate_frozen_training_contract(frozen: Mapping, panel: Mapping, vocab: Mapping,
                                      splits: Mapping, model_config: Mapping,
                                      mode: str, seed: int) -> TrainingContract:
    """Reject a launch whose runtime inputs differ from the committed contract."""
    try:
        actual = TrainingContract(**dict(frozen))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid frozen training contract: {exc}") from exc
    expected = freeze_training_contract(panel, vocab, splits, model_config, mode, seed)
    if actual.as_dict() != expected.as_dict():
        differences = {key: (actual.as_dict().get(key), expected.as_dict().get(key))
                       for key in set(actual.as_dict()) | set(expected.as_dict())
                       if actual.as_dict().get(key) != expected.as_dict().get(key)}
        raise ValueError(f"frozen training contract mismatch: {differences}")
    return actual


def select_frozen_training_contract(payload: Mapping, mode: str, seed: int) -> Mapping:
    matches = [row for row in payload.get("contracts", [])
               if row.get("mode") == mode and int(row.get("seed", -1)) == int(seed)]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen contract for {mode} seed {seed}, found {len(matches)}")
    return matches[0]


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


def restore_state_output(output, batch_size: int, set_len: int, n_genes: int):
    """Restore the official model's flattened cell-set output explicitly."""
    expected_cells = int(batch_size) * int(set_len)
    if output.ndim == 3 and tuple(output.shape) == (batch_size, set_len, n_genes):
        return output
    if output.ndim == 3 and tuple(output.shape) == (1, expected_cells, n_genes):
        return output.reshape(batch_size, set_len, n_genes)
    if output.ndim == 2 and tuple(output.shape) == (expected_cells, n_genes):
        return output.reshape(batch_size, set_len, n_genes)
    raise ValueError(
        f"official STATE output shape {tuple(output.shape)} cannot be restored to "
        f"({batch_size}, {set_len}, {n_genes})")


def build_official_state_adapter(checkpoint: str | Path, n_genes: int,
                                 n_perturbations: int, cell_set_len: int = 32):
    """Instantiate the pinned official STATE architecture for full training.

    The pilot uses the lightweight 128-hidden 4/4/8 contract.  Full
    Scratch/Transfer runs use the checkpoint's hidden size and transformer
    shape so compatible attention/FFN/LayerNorm weights can actually be
    transferred; the set length is reduced to 32 for the D2 pilot-scale data.
    """
    torch = __import__("torch")
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    # The official constructor may normalize nested kwargs in place.  Keep
    # the payload immutable so the contract hash is stable before and after
    # model construction.
    hparams = copy.deepcopy(dict(payload["hyper_parameters"]))
    source_hidden = int(hparams["hidden_dim"])
    kwargs = {key: value for key, value in hparams.items()
              if key not in {"input_dim", "hidden_dim", "output_dim", "pert_dim", "batch_dim",
                             "gene_names", "gene_dim", "hvg_dim"}}
    kwargs["cell_set_len"] = int(cell_set_len)
    kwargs["batch_encoder"] = False
    kwargs["batch_predictor"] = False
    base = StateTransitionPerturbationModel(input_dim=int(n_genes), hidden_dim=source_hidden,
                                            output_dim=int(n_genes), pert_dim=int(n_perturbations),
                                            batch_dim=None, gene_dim=int(n_genes), **kwargs)

    class _Adapter(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, expression, perturbation):
            batch = {"ctrl_cell_emb": expression, "pert_emb": perturbation,
                     "batch": torch.zeros(expression.shape[:2], dtype=torch.long,
                                           device=expression.device)}
            output = self.wrapped(batch)
            output = restore_state_output(output, expression.shape[0], expression.shape[1],
                                          expression.shape[2])
            if not torch.isfinite(output).all():
                raise FloatingPointError("official STATE adapter produced NaN/Inf")
            return output

    return _Adapter(base), payload


def initialize_transfer_adapter(adapter, checkpoint_payload: Mapping,
                                target_gene_names, target_perturbation_names,
                                expected_report: Mapping | None = None) -> dict:
    """Apply the audited semantic transfer to an official STATE adapter.

    The checkpoint's perturbation width is intentionally not treated as an
    ordered vocabulary.  When names are unavailable, that projection remains
    at its random target initialization, exactly as recorded by the audit.
    """
    from .state_d2_model import make_transfer_report
    source_hparams = dict(checkpoint_payload.get("hyper_parameters", {}))
    source_genes = list(source_hparams.get("gene_names", []))
    source_perturbations = list(source_hparams.get("perturbation_names", []))
    wrapped = getattr(adapter, "wrapped", adapter)
    transferred, report = make_transfer_report(
        checkpoint_payload["state_dict"], wrapped.state_dict(), source_genes,
        list(target_gene_names), source_perturbations, list(target_perturbation_names))
    if expected_report is not None:
        if expected_report.get("semantic_assertions_pass") is not True:
            raise ValueError("transfer report did not pass semantic assertions")
        if "target_gene_order_hash" in expected_report:
            from .state_d2 import _hash_payload
            if expected_report["target_gene_order_hash"] != _hash_payload(list(target_gene_names)):
                raise ValueError("transfer report mismatch: target gene order")
        if "target_perturbation_vocab_hash" in expected_report:
            from .state_d2 import _hash_payload
            if expected_report["target_perturbation_vocab_hash"] != _hash_payload(list(target_perturbation_names)):
                raise ValueError("transfer report mismatch: target perturbation vocabulary")
        for key in ("gene_overlap", "perturbation_overlap", "copied_key_count", "target_key_count"):
            if key in expected_report and expected_report[key] != report.get(key):
                raise ValueError(f"transfer report mismatch: {key}")
    wrapped.load_state_dict(transferred)
    return report


def _set_phase(model, mode: str, phase: int):
    for name, parameter in model.named_parameters():
        if mode == "Transfer" and "transformer" in name:
            parameter.requires_grad = phase >= 2
        else:
            parameter.requires_grad = True


def _optimizer_for_phase(model, mode: str, phase: int, head_lr: float, backbone_lr: float):
    torch = __import__("torch")
    if mode == "Scratch" or phase == 1:
        parameters = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(parameters, lr=head_lr, weight_decay=5e-4)
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if "transformer" in name else head).append(parameter)
    groups = []
    if head:
        groups.append({"params": head, "lr": head_lr})
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=5e-4)


def _atomic_torch_save(torch, payload: Mapping, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _rng_state(torch) -> dict:
    state = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch, state: Mapping) -> None:
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _training_iterator(stream, completed_steps: int):
    if hasattr(stream, "iter_from"):
        return stream.iter_from(completed_steps)
    iterator = iter(stream)
    for _ in range(completed_steps):
        next(iterator)
    return iterator


def train_two_phase(model, train_batches: Iterable[Mapping], validation_batches: Iterable[Mapping],
                    contract: TrainingContract, output_dir: str | Path,
                    phase1_steps: int | None = None, validation_steps: int | None = None,
                    learning_rate: float | None = None, finetune_learning_rate: float | None = None,
                    stop_after_steps: int | None = None, resume: bool = False,
                    log_fn: Callable[[Mapping], None] | None = None) -> dict:
    """Run the identical bounded loop for Scratch and Transfer.

    ``train_batches`` and ``validation_batches`` are factories or re-iterable
    streams yielding ``expression``, ``perturbation`` and ``target`` tensors.
    The function is deliberately agnostic to the source of D2 rows so the
    same loader can be used for both modes.
    """
    torch = __import__("torch")
    phase1_steps = contract.phase1_steps if phase1_steps is None else int(phase1_steps)
    learning_rate = contract.head_learning_rate if learning_rate is None else float(learning_rate)
    finetune_learning_rate = (contract.backbone_learning_rate if finetune_learning_rate is None
                              else float(finetune_learning_rate))
    if phase1_steps <= 0 or phase1_steps >= contract.max_steps:
        raise ValueError("phase1_steps must leave steps for phase 2")
    if phase1_steps != contract.phase1_steps:
        raise ValueError("phase1_steps differs from the frozen contract")
    if learning_rate != contract.head_learning_rate or finetune_learning_rate != contract.backbone_learning_rate:
        raise ValueError("learning rates differ from the frozen contract")
    run_limit = contract.max_steps if stop_after_steps is None else int(stop_after_steps)
    if run_limit <= 0 or run_limit > contract.max_steps:
        raise ValueError("stop_after_steps must be within the frozen training budget")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    last_path = target_dir / "last.ckpt"
    torch.manual_seed(contract.seed)
    start_step = 0
    best = float("inf")
    best_step = 0
    patience = 0
    history = []
    resumed_from_step = 0
    resume_payload = None
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {last_path}")
        resume_payload = torch.load(last_path, map_location="cpu", weights_only=False)
        if resume_payload.get("contract_hash") != contract.hash():
            raise ValueError("resume checkpoint contract mismatch")
        model.load_state_dict(resume_payload["state_dict"])
        start_step = int(resume_payload["step"])
        resumed_from_step = start_step
        best = float(resume_payload["best_validation_mmd"])
        best_step = int(resume_payload["best_step"])
        patience = int(resume_payload["patience"])
        history = list(resume_payload.get("history", []))
    if start_step >= run_limit:
        raise ValueError("resume checkpoint is already at or beyond stop_after_steps")
    active_phase = 1 if contract.mode == "Transfer" and start_step <= phase1_steps else 2
    _set_phase(model, contract.mode, active_phase)
    optimizer = _optimizer_for_phase(model, contract.mode, active_phase,
                                     learning_rate, finetune_learning_rate)
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        _restore_rng_state(torch, resume_payload.get("rng_state", {}))
    train_iter = _training_iterator(train_batches, start_step)
    validation_steps = 32 if validation_steps is None else int(validation_steps)

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

    for step in range(start_step + 1, run_limit + 1):
        if contract.mode == "Transfer" and step == phase1_steps + 1:
            _set_phase(model, contract.mode, 2)
            optimizer = _optimizer_for_phase(model, contract.mode, 2,
                                             learning_rate, finetune_learning_rate)
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
        if step % contract.validation_every == 0 or step == run_limit:
            validation_mmd = evaluate(step)
            record = {"step": step, "train_mmd": float(loss.detach().cpu()),
                      "validation_mmd": validation_mmd,
                      "phase": ("full" if contract.mode == "Scratch" else
                                ("heads" if step <= phase1_steps else "joint"))}
            history.append(record)
            if log_fn:
                log_fn(record)
            if validation_mmd < best:
                best = validation_mmd
                best_step = step
                patience = 0
                _atomic_torch_save(torch, {
                    "state_dict": model.state_dict(), "contract": contract.as_dict(),
                    "contract_hash": contract.hash(), "best_validation_mmd": best, "step": step,
                }, target_dir / "best.ckpt")
            else:
                patience += 1
            _atomic_torch_save(torch, {
                "state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "contract": contract.as_dict(), "contract_hash": contract.hash(), "step": step,
                "best_validation_mmd": best, "best_step": best_step, "patience": patience,
                "history": history, "rng_state": _rng_state(torch),
            }, last_path)
            if patience >= contract.early_stop_patience:
                break
    result = {"version": "d2_state_training_result.v2", "contract": contract.as_dict(),
              "contract_hash": contract.hash(), "best_validation_mmd": best,
              "best_step": best_step, "stopped_step": history[-1]["step"] if history else 0,
              "history": history, "checkpoint": str(target_dir / "best.ckpt"),
              "last_checkpoint": str(last_path), "resumed_from_step": resumed_from_step,
              "complete": bool(history and (history[-1]["step"] == contract.max_steps or
                                             patience >= contract.early_stop_patience)),
              "d2_responses_used": True}
    (target_dir / "training_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
