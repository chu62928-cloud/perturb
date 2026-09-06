from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    rmse_improvement: float = 0.05
    bootstrap_ci_lower: float = 0.0
    program_direction_margin: float = 0.02
    critical_program_tolerance: float = 0.05
    condition_rmse_tolerance: float = 0.10
    support_knn_quantile: float = 0.95
    support_density_quantile: float = 0.05
    support_min_effective_cells: int = 20
    composition_error_amplification: float = 0.10
    composition_support_pass_rate: float = 0.80
    state_distribution_metrics_required: int = 2
    state_proportion_error_tolerance: float = 0.05
    state_region_rmse_tolerance: float = 0.10
    composition_min_valid_triplets: int = 50
    composition_min_second_actions: int = 20
    composition_min_regions: int = 3


@dataclass(frozen=True)
class PilotConfig:
    n_genes: int = 96
    core_genes: int = 64
    challenge_genes: int = 32
    seed: int = 20260901
    max_cells_per_condition: int = 10000
    min_cells_per_guide: int = 100
    min_guides_per_gene: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    project_root: str = "."
    data_root: str = "/root/autodl-tmp/CRISPR_perturb"
    runtime_root: str = "/root/autodl-tmp/CRISPR_perturb_runtime/pipeline"
    shared_gene_count: int = 2000
    latent_dimensions: int = 50
    local_state_resolution: float = 1.0
    continuous_state: dict[str, Any] = field(default_factory=lambda: {
        "n_anchors_per_condition": 10, "k_neighbors": 200, "n_folds": 5,
        "density_quantile": 0.10, "buffer_scale": 1.25,
        "leiden_role": "reporting_only"})
    holdout_gene_fraction: float = 0.20
    holdout_interaction_fraction: float = 0.20
    random_seed: int = 20260901
    development_donor: str = "D2"
    secondary_confirmation_donors: tuple[str, ...] = ("D1",)
    external_test_donors: tuple[str, ...] = ("D3", "D4")
    role_manifest: str = "metadata/donor_roles_v1.json"
    pilot: PilotConfig = field(default_factory=PilotConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    dataset_ids: tuple[str, ...] = (
        "D1_Rest", "D1_Stim8hr", "D1_Stim48hr",
        "D2_Rest", "D2_Stim8hr", "D2_Stim48hr",
        "D3_Rest", "D3_Stim8hr", "D3_Stim48hr",
        "D4_Rest", "D4_Stim8hr", "D4_Stim48hr",
    )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dataset_ids"] = list(self.dataset_ids)
        result["secondary_confirmation_donors"] = list(self.secondary_confirmation_donors)
        result["external_test_donors"] = list(self.external_test_donors)
        return result

    def hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(payload).hexdigest()


def _merge(default: Any, incoming: Any) -> Any:
    if isinstance(default, dict) and isinstance(incoming, dict):
        return {k: _merge(default.get(k), v) for k, v in incoming.items()}
    return incoming


def load_config(path: str | Path) -> PipelineConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    base = PipelineConfig().as_dict()
    merged = _merge(base, raw)
    merged["pilot"] = PilotConfig(**merged.get("pilot", {}))
    merged["gates"] = GateConfig(**merged.get("gates", {}))
    merged["dataset_ids"] = tuple(merged.get("dataset_ids", []))
    merged["secondary_confirmation_donors"] = tuple(merged.get("secondary_confirmation_donors", ["D1"]))
    merged["external_test_donors"] = tuple(merged.get("external_test_donors", ["D3", "D4"]))
    merged["development_donor"] = str(merged.get("development_donor", "D2")).upper()
    return PipelineConfig(**merged)
