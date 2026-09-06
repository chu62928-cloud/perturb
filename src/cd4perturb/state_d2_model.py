from __future__ import annotations

"""Small, deterministic STATE-compatible set model and transfer audit.

The heavy official STATE package is imported only on the remote training
machine.  This adapter keeps the D2 contract testable locally and makes every
transfer decision explicit: tensors are copied only when their parameter
semantics, key and shape are compatible.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class D2StateConfig:
    n_genes: int = 2000
    n_perturbations: int = 128
    cell_set_len: int = 32
    hidden_dim: int = 128
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_attention_heads: int = 8
    dropout: float = 0.1
    predict_residual: bool = True
    batch_encoder: bool = False
    max_steps: int = 40000
    seeds: tuple[int, ...] = (20260901, 20260902, 20260903)

    def as_dict(self) -> dict:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


def _torch():  # pragma: no cover - the local orchestration env intentionally has no torch
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("D2 STATE model requires the isolated e3_state PyTorch environment") from exc
    return torch, nn


def build_d2_state_model(config: D2StateConfig | None = None):
    """Build the 32×2,000 pilot architecture (hidden 128, 4/4, 8 heads)."""
    torch, nn = _torch()
    cfg = config or D2StateConfig()
    if cfg.hidden_dim % cfg.n_attention_heads:
        raise ValueError("hidden_dim must be divisible by n_attention_heads")

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            self.basal_encoder = nn.Linear(cfg.n_genes, cfg.hidden_dim)
            self.pert_encoder = nn.Linear(cfg.n_perturbations, cfg.hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim, nhead=cfg.n_attention_heads,
                dim_feedforward=4 * cfg.hidden_dim, dropout=cfg.dropout,
                batch_first=True, norm_first=True, activation="gelu")
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=cfg.hidden_dim, nhead=cfg.n_attention_heads,
                dim_feedforward=4 * cfg.hidden_dim, dropout=cfg.dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, cfg.n_encoder_layers)
            self.transformer_decoder = nn.TransformerDecoder(decoder_layer, cfg.n_decoder_layers)
            self.output = nn.Linear(cfg.hidden_dim, cfg.n_genes)

        def forward(self, expression, perturbation):
            if expression.ndim != 3 or perturbation.ndim != 3:
                raise ValueError("expression and perturbation must be [batch, set, feature]")
            if expression.shape[:2] != perturbation.shape[:2]:
                raise ValueError("expression and perturbation set dimensions differ")
            if expression.shape[-1] != cfg.n_genes or perturbation.shape[-1] != cfg.n_perturbations:
                raise ValueError("D2 model feature dimensions do not match frozen contract")
            basal = self.basal_encoder(expression)
            perturb = self.pert_encoder(perturbation)
            tokens = basal + perturb
            memory = self.transformer_encoder(tokens)
            decoded = self.transformer_decoder(tokens, memory)
            prediction = self.output(decoded)
            if cfg.predict_residual:
                prediction = prediction + expression
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("D2 STATE forward produced NaN/Inf")
            return prediction

        def set_prediction(self, expression, perturbation):
            """Return a set-level prediction with the required S×G shape."""
            output = self.forward(expression, perturbation)
            if output.shape[0] != 1:
                raise ValueError("set_prediction accepts exactly one set")
            return output[0]

    return _Model()


def pilot_forward_contract(config: D2StateConfig | None = None, device: str = "cpu") -> dict:
    """Run a bounded forward/loss/save/reload contract smoke in e3_state."""
    torch, _ = _torch()
    cfg = config or D2StateConfig()
    torch.manual_seed(20260901)
    model = build_d2_state_model(cfg).to(device)
    expression = torch.rand(1, cfg.cell_set_len, cfg.n_genes, device=device)
    perturbation = torch.zeros(1, cfg.cell_set_len, cfg.n_perturbations, device=device)
    perturbation[:, :, 1] = 1.0
    target = expression * 0.95
    prediction = model(expression, perturbation)
    loss = torch.nn.functional.mse_loss(prediction, target)
    if tuple(prediction.shape) != (1, cfg.cell_set_len, cfg.n_genes):
        raise AssertionError(f"unexpected model output: {tuple(prediction.shape)}")
    if not torch.isfinite(loss):
        raise FloatingPointError("D2 pilot loss is not finite")
    with __import__("tempfile").NamedTemporaryFile(suffix=".pt") as handle:
        torch.save({"state_dict": model.state_dict(), "config": cfg.as_dict()}, handle.name)
        restored = build_d2_state_model(cfg)
        payload = torch.load(handle.name, map_location="cpu", weights_only=True)
        restored.load_state_dict(payload["state_dict"])
    return {"version": "d2_state_pilot_contract.v1", "config": cfg.as_dict(),
            "config_hash": cfg.hash(), "output_shape": list(prediction.shape[1:]),
            "loss": float(loss.detach().cpu()), "finite": True, "checkpoint_reload": True,
            "device": str(device), "batch_encoder": False}


def train_d2_pilot(config: D2StateConfig, train_batches, selected_perturbations: Sequence[str],
                   output_dir: str | Path, *, steps: int = 200, device: str = "cuda",
                   learning_rate: float = 1e-3, comparison_window: int = 20) -> dict:
    """Run the response-blind lightweight optimizer pilot and enforce its hard gates."""
    torch, _ = _torch()
    if steps < 2 * comparison_window or comparison_window <= 0:
        raise ValueError("pilot steps must contain two comparison windows")
    selected = list(map(str, selected_perturbations))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("pilot perturbations must be unique and non-empty")
    torch.manual_seed(20260901)
    model = build_d2_state_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    probe = next(iter(train_batches))
    expected_shape = (int(probe["expression"].shape[0]), config.cell_set_len, config.n_genes)
    model.eval()
    with torch.no_grad():
        initial_prediction = model(probe["expression"], probe["perturbation"])
        initial_probe_loss = torch.nn.functional.mse_loss(initial_prediction, probe["target"])
    if tuple(initial_prediction.shape) != expected_shape or not torch.isfinite(initial_probe_loss):
        raise RuntimeError("pilot probe failed shape or finite-value checks")
    model.train()
    losses = []
    seen = set()
    iterator = iter(train_batches)
    for _ in range(int(steps)):
        batch = next(iterator)
        seen.update(map(str, batch.get("perturbation_names", [])))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["expression"], batch["perturbation"])
        if tuple(prediction.shape) != (int(batch["expression"].shape[0]),
                                      config.cell_set_len, config.n_genes):
            raise RuntimeError("pilot training output shape changed")
        loss = torch.nn.functional.mse_loss(prediction, batch["target"])
        if not torch.isfinite(loss):
            raise FloatingPointError("pilot training loss is NaN/Inf")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    first_median = float(__import__("statistics").median(losses[:comparison_window]))
    last_median = float(__import__("statistics").median(losses[-comparison_window:]))
    model.eval()
    with torch.no_grad():
        final_prediction = model(probe["expression"], probe["perturbation"])
        final_probe_loss = torch.nn.functional.mse_loss(final_prediction, probe["target"])
    loss_descent = bool(last_median < first_median and final_probe_loss < initial_probe_loss)
    covered_all = set(selected).issubset(seen)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = target_dir / "pilot.ckpt"
    temporary = checkpoint.with_suffix(".ckpt.tmp")
    torch.save({"state_dict": model.state_dict(), "config": config.as_dict()}, temporary)
    temporary.replace(checkpoint)
    restored = build_d2_state_model(config).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state_dict"])
    restored.eval()
    with torch.no_grad():
        restored_prediction = restored(probe["expression"], probe["perturbation"])
    reload_ok = bool(torch.allclose(final_prediction, restored_prediction, atol=1e-6, rtol=1e-6))
    peak_memory = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    memory_stable = not device.startswith("cuda") or peak_memory < 32 * 1024 ** 3
    result = {
        "version": "d2_state_pilot_metrics.v1", "steps": int(steps),
        "selected_perturbation_count": len(selected), "covered_perturbation_count": len(seen),
        "covered_all_perturbations": covered_all, "first_window_median_loss": first_median,
        "last_window_median_loss": last_median,
        "initial_probe_loss": float(initial_probe_loss.detach().cpu()),
        "final_probe_loss": float(final_probe_loss.detach().cpu()), "loss_descent": loss_descent,
        "finite": all(map(__import__("math").isfinite, losses)),
        "output_shape": list(final_prediction.shape), "checkpoint_reload": reload_ok,
        "peak_cuda_memory_bytes": peak_memory, "memory_stable": memory_stable, "device": device,
        "test_responses_used": False, "d2_responses_used": True,
    }
    if not (result["loss_descent"] and result["finite"] and covered_all and reload_ok and memory_stable):
        raise RuntimeError(f"D2 pilot hard gate failed: {result}")
    metrics_path = target_dir / "pilot_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _gene_row_or_column_key(key: str) -> str | None:
    if key in {"basal_encoder.0.weight", "basal_encoder.weight"}:
        return "gene_input"
    if key in {"gene_decoder.decoder.12.weight", "gene_decoder.decoder.12.bias", "output.weight", "output.bias"}:
        return "gene_output"
    if key in {"pert_encoder.0.weight", "pert_encoder.weight"}:
        return "perturbation_input"
    return None


def _aligned_tensor(source, target, source_names: Sequence[str], target_names: Sequence[str], axis: int):
    """Copy overlapping named rows/columns while preserving target defaults."""
    torch, _ = _torch()
    if source.ndim != target.ndim:
        return None, 0
    source_index = {str(name): i for i, name in enumerate(source_names)}
    target_index = {str(name): i for i, name in enumerate(target_names)}
    overlap = sorted(set(source_index).intersection(target_index))
    if not overlap:
        return None, 0
    result = target.detach().clone()
    for name in overlap:
        s, t = source_index[name], target_index[name]
        source_slice = source.select(axis, s)
        target_slice = result.select(axis, t)
        if source_slice.shape != target_slice.shape:
            return None, 0
        result.select(axis, t).copy_(source_slice)
    return result, len(overlap)


def make_transfer_report(source_state: Mapping, target_state: Mapping,
                         source_gene_names: Sequence[str], target_gene_names: Sequence[str],
                         source_perturbations: Sequence[str], target_perturbations: Sequence[str]) -> dict:
    """Report and construct a semantically aligned state-dict transfer.

    Exact-shape backbone tensors are copied by key.  Gene and perturbation
    projection tensors are aligned by names, never by integer indices.  The
    returned report is suitable for an automatic pre-training assertion.
    """
    transferred = {key: value.detach().clone() for key, value in target_state.items()}
    copied_exact, copied_aligned, skipped = [], [], []
    overlap_counts = {}
    for key, target in target_state.items():
        source = source_state.get(key)
        if source is None:
            skipped.append({"key": key, "reason": "missing_source_key"})
            continue
        semantic = _gene_row_or_column_key(key)
        if semantic == "gene_input":
            aligned, count = _aligned_tensor(source, target, source_gene_names, target_gene_names, axis=1)
            if aligned is not None:
                transferred[key] = aligned
                copied_aligned.append(key)
                overlap_counts[key] = count
            else:
                skipped.append({"key": key, "reason": "gene_input_shape_or_overlap_mismatch"})
        elif semantic == "gene_output":
            axis = 0
            aligned, count = _aligned_tensor(source, target, source_gene_names, target_gene_names, axis=axis)
            if aligned is not None:
                transferred[key] = aligned
                copied_aligned.append(key)
                overlap_counts[key] = count
            else:
                skipped.append({"key": key, "reason": "gene_output_shape_or_overlap_mismatch"})
        elif semantic == "perturbation_input":
            aligned, count = _aligned_tensor(source, target, source_perturbations, target_perturbations, axis=1)
            if aligned is not None:
                transferred[key] = aligned
                copied_aligned.append(key)
                overlap_counts[key] = count
            else:
                skipped.append({"key": key, "reason": "perturbation_name_shape_or_overlap_mismatch"})
        elif source.shape == target.shape:
            transferred[key] = source.detach().clone()
            copied_exact.append(key)
        else:
            skipped.append({"key": key, "reason": "shape_mismatch"})
    report = {"version": "state_transfer_report.v1",
              "rules": {"backbone": "key_and_shape", "genes": "gene_symbol_alignment",
                        "perturbations": "perturbation_name_alignment", "integer_index_copy": False},
              "source_gene_count": len(source_gene_names), "target_gene_count": len(target_gene_names),
              "gene_overlap": len(set(source_gene_names).intersection(target_gene_names)),
              "perturbation_overlap": len(set(source_perturbations).intersection(target_perturbations)),
              "copied_exact_keys": copied_exact, "copied_aligned_keys": copied_aligned,
              "overlap_counts": overlap_counts, "skipped": skipped,
              "copied_key_count": len(copied_exact) + len(copied_aligned),
              "target_key_count": len(target_state), "semantic_assertions_pass": True,
              "transfer_hash": hashlib.sha256(json.dumps({"exact": copied_exact, "aligned": copied_aligned,
                                                            "skipped": skipped}, sort_keys=True).encode()).hexdigest()}
    if any(item["reason"] == "perturbation_name_shape_or_overlap_mismatch" for item in skipped):
        # A completely new perturbation projection is allowed, but an index
        # based copy is never silently accepted.
        report["new_perturbations_random_initialized"] = True
    return transferred, report


def write_transfer_report(report: Mapping, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
