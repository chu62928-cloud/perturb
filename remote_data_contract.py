from __future__ import annotations

"""Evidence-based freezing and validation of the D1 data contract."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_optional(root: Path, *relative: str) -> tuple[dict | None, Path | None]:
    for name in relative:
        path = root / name
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8")), path
            except (OSError, json.JSONDecodeError):
                return None, path
    return None, None


def _contains_d2_response(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "d2_responses_used" and bool(item):
                return True
            if _contains_d2_response(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_d2_response(item) for item in value)
    return False


def _canonical_hash(value: dict, field: str, *also_omit: str) -> str:
    """Hash an artifact payload exactly as the producer does."""
    payload = dict(value)
    payload.pop(field, None)
    for name in also_omit:
        payload.pop(name, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _gene_order_hash(order: list[Any]) -> str:
    return hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()


def _finite_effect_rows(effects: dict, n_genes: int) -> bool:
    """Check dimensions and finite values without trusting a manifest flag."""
    gene_effects = effects.get("gene_effects")
    if not isinstance(gene_effects, dict):
        return False
    for by_gene in gene_effects.values():
        if not isinstance(by_gene, dict):
            return False
        for vector in by_gene.values():
            if not isinstance(vector, list) or len(vector) != n_genes:
                return False
            if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in vector):
                return False
    return True


def validate_pilot_contract(primary_path: str | Path, challenge_path: str | Path,
                            registry_path: str | Path, evidence_dir: str | Path | None = None) -> dict:
    """Validate frozen pilot sets plus real A-D evidence artifacts.

    Without ``evidence_dir`` this intentionally remains ``DATA_CONTRACT_READY``
    for compatibility with the pre-evidence structural check.  A true
    ``DATA_VALID`` result requires guide QC, normalization/gene order, D1
    effects, continuous state splits and the immutable pilot manifest.
    """
    primary_path, challenge_path, registry_path = map(Path, (primary_path, challenge_path, registry_path))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    primary_genes = {str(row["gene"]) for row in primary}
    challenge_genes = {str(row["gene"]) for row in challenge}
    reasons: list[str] = []
    if len(primary) != 64 or len(challenge) != 32:
        reasons.append("pilot sizes are not primary_64 plus challenge_32")
    if primary_genes & challenge_genes:
        reasons.append("primary and challenge genes overlap")
    if any(str(row.get("tier")) != "primary_64" for row in primary):
        reasons.append("primary tier label is invalid")
    if any(int(row.get("n_guides", 0)) < 2 or int(row.get("guide_coverage", 0)) < 200 for row in primary):
        reasons.append("primary guide coverage does not prove two guides with 100 cells each")
    if any(row.get("effect_bin", "unknown") != "unknown" or
           row.get("guide_concordance", "unknown") != "unknown" or
           bool(row.get("stimulus_dependent", False)) for row in primary):
        reasons.append("primary set contains response-driven annotations")
    complete = sum(str(row.get("status")) == "complete" for row in registry)
    d1_ids = {"D1_Rest", "D1_Stim8hr", "D1_Stim48hr"}
    d1_records = [row for row in registry if str(row.get("donor_id")) == "D1"]
    d1_complete = sum(str(row.get("status")) == "complete" for row in d1_records)
    stable_files = all(str(row.get("status")) == "complete" and
                       int(row.get("expected_size", 0)) == int(row.get("observed_size", -1)) and
                       bool(row.get("fingerprint")) and bool(row.get("shape"))
                       for row in d1_records) and \
        len(d1_records) == 3 and {str(row.get("dataset_id")) for row in d1_records} == d1_ids
    mapping_rate = None
    gene_order_ok = False
    normalization_ok = False
    split_hash_ok = False
    effects_ok = False
    state_regions_ok = False
    csr_integrity_ok = False
    evidence_paths: dict[str, str] = {}
    if evidence_dir is not None:
        root = Path(evidence_dir)
        guide, guide_path = _load_optional(root, "metadata/guide_qc_v1.json", "guide_qc_v1.json")
        order, order_path = _load_optional(root, "metadata/shared_gene_order_v2.json", "metadata/shared_gene_order_v1.json", "shared_gene_order_v2.json", "shared_gene_order_v1.json")
        effects, effects_path = _load_optional(root, "results/d1_effect_matrix_v2.json", "results/d1_effect_matrix.json")
        split, split_path = _load_optional(root, "metadata/benchmark_splits_v1.json", "metadata/continuous_state_split_v1.json", "metadata/continuous_state_regions_v1.json")
        state_regions, state_regions_path = _load_optional(root, "metadata/continuous_state_regions_v1.json")
        csr_integrity, csr_integrity_path = _load_optional(root, "metadata/d1_csr_integrity_v1.json")
        pilot_manifest, pilot_path = _load_optional(root, "metadata/pilot_manifest_v1.json", "pilot_manifest_v1.json")
        for key, path in (("guide_qc", guide_path), ("gene_order", order_path), ("effects", effects_path),
                          ("state_split", split_path), ("state_regions", state_regions_path),
                          ("csr_integrity", csr_integrity_path),
                          ("pilot_manifest", pilot_path)):
            if path is not None:
                evidence_paths[key] = str(path)
        if guide is None:
            reasons.append("guide QC evidence missing or invalid")
        else:
            mapping_rate = guide.get("unique_targeting_mapping_rate")
            if (mapping_rate is None or not guide.get("unique_observed_targeting_guides", 0) or
                    not math.isfinite(float(mapping_rate)) or not 0 <= float(mapping_rate) <= 1 or
                    int(guide.get("unique_mapped_targeting_guides", 0)) >
                    int(guide.get("unique_observed_targeting_guides", 0))):
                reasons.append("guide mapping rate evidence missing")
        if order is None or len(order.get("gene_order", [])) != 2000:
            reasons.append("shared 2000 gene order evidence missing")
        else:
            expected_gene_hash = _gene_order_hash(order["gene_order"])
            gene_order_ok = (order.get("gene_order_hash") == expected_gene_hash and
                              order.get("d2_responses_used") is False and
                              bool(order.get("artifact_hash")) and
                              _canonical_hash(order, "artifact_hash") == order.get("artifact_hash"))
            normalization_ok = ("CP10K" in str(order.get("algorithm", "")) and
                                bool(order.get("artifact_hash")) and
                                order.get("ntc_sample_size_used") and
                                all(int(x) > 0 for x in order["ntc_sample_size_used"].values()))
            if not gene_order_ok:
                reasons.append("gene order content/hash or D2 guard invalid")
            if not normalization_ok:
                reasons.append("normalization evidence missing")
        if effects is None or len(effects.get("gene_order", [])) != 2000 or not effects.get("effect_hash"):
            reasons.append("D1 effect matrix evidence missing")
        else:
            effects_ok = (effects.get("d2_responses_used") is False and
                          set(effects.get("conditions", [])) == {"D1_Rest", "D1_Stim8hr", "D1_Stim48hr"} and
                          effects.get("gene_order") == (order or {}).get("gene_order") and
                          effects.get("gene_order_hash") == _gene_order_hash(effects["gene_order"]) and
                          _canonical_hash(effects, "effect_hash") == effects.get("effect_hash") and
                          _finite_effect_rows(effects, 2000) and
                          all(str(effects.get("condition_metadata", {}).get(c, {}).get("library_size_field")) ==
                              "total_counts" for c in effects.get("conditions", [])))
            if order is not None and effects.get("gene_order") != order.get("gene_order"):
                effects_ok = False
                reasons.append("D1 effect matrix gene order differs from frozen order")
            if not effects_ok:
                reasons.append("D1 effect matrix content/hash/guard/conditions invalid")
        if state_regions is None:
            reasons.append("continuous state regions evidence missing")
        else:
            state_regions_ok = (state_regions.get("d2_responses_used") is False and
                                bool(state_regions.get("split_hash")) and
                                # state-regions v1 was hashed before the
                                # command-level D2 guard was appended.
                                _canonical_hash(state_regions, "split_hash", "d2_responses_used") == state_regions.get("split_hash") and
                                len(state_regions.get("regions", [])) >= 1 and
                                len(state_regions.get("folds", [])) >= 1)
            if not state_regions_ok:
                reasons.append("continuous state region content/hash/guard invalid")
        if csr_integrity is None:
            reasons.append("D1 CSR integrity evidence missing")
        else:
            csr_integrity_ok = (csr_integrity.get("d2_responses_used") is False and
                                csr_integrity.get("all_valid") is True and
                                bool(csr_integrity.get("artifact_hash")) and
                                _canonical_hash(csr_integrity, "artifact_hash") == csr_integrity.get("artifact_hash"))
            if not csr_integrity_ok:
                reasons.append("D1 CSR integrity evidence invalid")
        if split is None or not split.get("split_hash") or not (split.get("folds") or split.get("state")):
            reasons.append("continuous state split evidence missing")
        else:
            split_hash_ok = (split.get("d2_responses_used", False) is False and
                             _canonical_hash(split, "split_hash") == split.get("split_hash"))
            if "gene" in split and (not split.get("gene", {}).get("train") or not split.get("gene", {}).get("test")):
                split_hash_ok = False
            if "gene_x_state" in split and (not split.get("gene_x_state", {}).get("train") or not split.get("gene_x_state", {}).get("test")):
                split_hash_ok = False
            if state_regions is not None and split.get("state", {}).get("state_split_hash") != state_regions.get("split_hash"):
                split_hash_ok = False
                reasons.append("benchmark split does not reference frozen state regions")
            if not split_hash_ok:
                reasons.append("state split content/hash/guard invalid")
        if pilot_manifest is None:
            reasons.append("pilot manifest evidence missing")
        else:
            if (pilot_manifest.get("primary_count") != 64 or pilot_manifest.get("challenge_count") != 32 or
                    pilot_manifest.get("primary_response_blind") is not True or
                    pilot_manifest.get("primary_sha256") != _sha256(primary_path) or
                    pilot_manifest.get("challenge_sha256") != _sha256(challenge_path)):
                reasons.append("pilot manifest hashes do not match frozen sets")
        all_evidence = [x for x in (guide, order, effects, split, pilot_manifest) if x is not None]
        all_evidence.append(state_regions)
        all_evidence.append(csr_integrity)
        if any(_contains_d2_response(x) for x in all_evidence):
            reasons.append("evidence reports d2_responses_used=true")
    else:
        reasons.append("A-D evidence directory was not supplied")
    ok = (not reasons and complete >= 1 and d1_complete == 3 and stable_files and
          mapping_rate is not None and gene_order_ok and normalization_ok and
          effects_ok and state_regions_ok and csr_integrity_ok and split_hash_ok)
    return {"status": "DATA_VALID" if ok else "DATA_CONTRACT_READY", "valid": ok,
            "primary_count": len(primary), "challenge_count": len(challenge),
            "primary_sha256": _sha256(primary_path), "challenge_sha256": _sha256(challenge_path),
            "registry_sha256": _sha256(registry_path), "d1_complete": d1_complete,
            "stable_files": stable_files, "library_mapping_rate": mapping_rate,
            "library_mapping_rate_ok": mapping_rate is not None,
            "gene_order_hash_present": gene_order_ok, "normalization_hash_present": normalization_ok,
            "effect_matrix_evidence": effects_ok, "split_hash_present": split_hash_ok,
            "state_regions_evidence": state_regions_ok,
            "csr_integrity_evidence": csr_integrity_ok,
            "evidence_paths": evidence_paths, "reasons": reasons,
            "d2_responses_used": False}


def freeze_data_contract(primary_path: str | Path, challenge_path: str | Path,
                         registry_path: str | Path, output: str | Path,
                         evidence_dir: str | Path | None = None) -> dict:
    result = validate_pilot_contract(primary_path, challenge_path, registry_path, evidence_dir)
    result["frozen"] = bool(result["valid"])
    result["contract_version"] = "data_contract.frozen_v2"
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if target.exists():
        old = target.read_text(encoding="utf-8")
        if old != encoded:
            raise FileExistsError(f"refusing to overwrite existing contract: {target}")
    else:
        target.write_text(encoded, encoding="utf-8")
    return result
