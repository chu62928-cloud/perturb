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
                            registry_path: str | Path, evidence_dir: str | Path | None = None,
                            allow_external_csr_patch: bool = False) -> dict:
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
    csr_patch_ok = False
    csr_patch_applied = False
    csr_patch_verify_ok = False
    raw_input_modified = False
    evidence_paths: dict[str, str] = {}
    if evidence_dir is not None:
        root = Path(evidence_dir)
        guide, guide_path = _load_optional(root, "metadata/guide_qc_v1.json", "guide_qc_v1.json")
        order, order_path = _load_optional(root, "metadata/shared_gene_order_v2.json", "metadata/shared_gene_order_v1.json", "shared_gene_order_v2.json", "shared_gene_order_v1.json")
        effects, effects_path = _load_optional(root, "results/d1_effect_matrix_v2.json", "results/d1_effect_matrix.json")
        split, split_path = _load_optional(root, "metadata/benchmark_splits_v1.json", "metadata/continuous_state_split_v1.json", "metadata/continuous_state_regions_v1.json")
        state_regions, state_regions_path = _load_optional(root, "metadata/continuous_state_regions_v1.json")
        csr_integrity, csr_integrity_path = _load_optional(root, "metadata/d1_csr_integrity_v2.json", "metadata/d1_csr_integrity_v1.json")
        csr_patch, csr_patch_path = _load_optional(root, "metadata/d1_rest_csr_patch_v1.json")
        csr_patch_verify, csr_patch_verify_path = _load_optional(root, "metadata/d1_rest_csr_patch_verify_v1.json")
        csr_patch_applied = csr_patch is not None
        csr_patch_verify_ok = (csr_patch_verify is not None and
                               csr_patch_verify.get("status") != "BLOCKED" and
                               csr_patch_verify.get("all_rows_valid") is True and
                               csr_patch_verify.get("d2_responses_used") is False)
        pilot_manifest, pilot_path = _load_optional(root, "metadata/pilot_manifest_v1.json", "pilot_manifest_v1.json")
        for key, path in (("guide_qc", guide_path), ("gene_order", order_path), ("effects", effects_path),
                          ("state_split", split_path), ("state_regions", state_regions_path),
                          ("csr_integrity", csr_integrity_path),
                          ("csr_patch", csr_patch_path),
                          ("csr_patch_verify", csr_patch_verify_path),
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
            if csr_patch_applied:
                effect_csr = effects.get("csr_integrity", {})
                effects_ok = (effects_ok and effect_csr.get("csr_patch_applied") is True and
                              effect_csr.get("raw_input_modified") is False and
                              effect_csr.get("patch_artifact_hash") == (csr_patch or {}).get("artifact_hash"))
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
        if csr_patch is not None:
            csr_patch_applied = True
            csr_patch_ok = (csr_patch.get("version") == "d1_rest_csr_patch.v1" and
                            csr_patch.get("dataset_id") == "D1_Rest" and
                            csr_patch.get("hdf5_dataset") == "X/indptr" and
                            csr_patch.get("raw_input_modified") is False and
                            csr_patch.get("csr_patch_applied") is True and
                            csr_patch.get("d2_responses_used") is False and
                            isinstance(csr_patch.get("chunks"), list) and len(csr_patch["chunks"]) == 2 and
                            all(str(row.get("sha256", "")) and
                                int(row.get("byte_end", -1)) - int(row.get("byte_start", 0)) + 1 == 8008
                                for row in csr_patch["chunks"]) and
                            bool(csr_patch.get("artifact_hash")) and
                            _canonical_hash(csr_patch, "artifact_hash") == csr_patch.get("artifact_hash"))
            if not csr_patch_ok:
                reasons.append("CSR patch provenance/hash/range evidence invalid")
            csr_patch_verify_ok = (csr_patch_verify is not None and
                                   csr_patch_verify.get("status") != "BLOCKED" and
                                   csr_patch_verify.get("all_rows_valid") is True and
                                   csr_patch_verify.get("d2_responses_used") is False)
            if not csr_patch_verify_ok:
                reasons.append("CSR patch boundary/full-value verification missing or blocked")
            if not allow_external_csr_patch:
                reasons.append("external CSR patch requires explicit scientific-policy acceptance")
        if csr_integrity is None:
            reasons.append("D1 CSR integrity evidence missing")
        else:
            csr_patch_applied = csr_patch_applied or bool(csr_integrity.get("csr_patch_applied", False))
            raw_input_modified = bool(csr_integrity.get("raw_input_modified", False))
            csr_integrity_ok = (csr_integrity.get("d2_responses_used") is False and
                                csr_integrity.get("all_valid") is True and
                                bool(csr_integrity.get("artifact_hash")) and
                                _canonical_hash(csr_integrity, "artifact_hash") == csr_integrity.get("artifact_hash"))
            if csr_patch_applied:
                csr_integrity_ok = (csr_integrity_ok and
                                    csr_integrity.get("patched_all_valid") is True and
                                    csr_integrity.get("raw_all_valid") is False and
                                    not raw_input_modified and csr_patch_ok and csr_patch_verify_ok)
            elif "raw_all_valid" in csr_integrity:
                csr_integrity_ok = csr_integrity_ok and csr_integrity.get("raw_all_valid") is True and not raw_input_modified
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
            "csr_patch_evidence": csr_patch_ok,
            "csr_patch_verify_evidence": csr_patch_verify_ok,
            "csr_patch_applied": csr_patch_applied,
            "raw_input_modified": raw_input_modified,
            "external_csr_patch_policy_accepted": bool(allow_external_csr_patch),
            "evidence_paths": evidence_paths, "reasons": reasons,
            "d2_responses_used": False}


def freeze_data_contract(primary_path: str | Path, challenge_path: str | Path,
                         registry_path: str | Path, output: str | Path,
                         evidence_dir: str | Path | None = None,
                         allow_external_csr_patch: bool = False) -> dict:
    result = validate_pilot_contract(primary_path, challenge_path, registry_path, evidence_dir,
                                     allow_external_csr_patch=allow_external_csr_patch)
    result["frozen"] = bool(result["valid"])
    result["contract_version"] = "data_contract.frozen_v3_csr_patch"
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


def validate_development_contract(primary_path: str | Path, challenge_path: str | Path,
                                  registry_path: str | Path, role_manifest_path: str | Path,
                                  evidence_dir: str | Path) -> dict:
    """Validate the versioned D2 development evidence bundle.

    This validator intentionally does not inspect expression values itself;
    the full CSR audit is consumed as an independent evidence artifact.
    """
    from .roles import validate_role_manifest

    primary_path, challenge_path, registry_path = map(Path, (primary_path, challenge_path, registry_path))
    role_path, root = Path(role_manifest_path), Path(evidence_dir)
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    challenge = json.loads(challenge_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    role = json.loads(role_path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    role_ok = validate_role_manifest(role, require_active=True) and role.get("development_donor") == "D2"
    if not role_ok:
        reasons.append("active D2 development role manifest invalid")
    d2_rows = [x for x in registry if x.get("donor_id") == "D2" and x.get("status") == "complete"]
    if len(d2_rows) != 3 or {x.get("condition") for x in d2_rows} != {"Rest", "Stim8hr", "Stim48hr"}:
        reasons.append("D2 development registry is incomplete")
    primary_genes = {str(x.get("gene")) for x in primary}
    challenge_genes = {str(x.get("gene")) for x in challenge}
    if len(primary) != 64 or len(challenge) != 32 or primary_genes & challenge_genes:
        reasons.append("primary/challenge sets are invalid or overlap")
    if any(x.get("tier") != "primary_64" for x in primary):
        reasons.append("primary tier is invalid")
    if any(x.get("effect_bin", "unknown") != "unknown" or
           x.get("guide_concordance", "unknown") != "unknown" or
           bool(x.get("stimulus_dependent", False)) for x in primary):
        reasons.append("primary set is not response blind")
    guide = _load_optional(root, "metadata/guide_qc_v1.json", "guide_qc_v1.json")[0]
    order = _load_optional(root, "metadata/shared_gene_order_v2.json", "metadata/shared_gene_order_v1.json")[0]
    effects = _load_optional(root, "results/d2_effect_matrix_v3.json", "results/effect_matrix_v3.json")[0]
    state = _load_optional(root, "metadata/continuous_state_regions_v1.json")[0]
    split = _load_optional(root, "metadata/benchmark_splits_v1.json")[0]
    csr = _load_optional(root, "metadata/d2_csr_audit_v1/d2_csr_audit_full_summary.json",
                         "metadata/d2_csr_audit_full_summary.json")[0]
    evidence = {"guide_qc": guide, "gene_order": order, "effects": effects,
                "state_regions": state, "splits": split, "csr_audit": csr}
    if any(value is None for value in evidence.values()):
        reasons.append("one or more D2 evidence artifacts are missing")
    for name, value in evidence.items():
        if value is None:
            continue
        if name != "csr_audit" and value.get("role_manifest_hash") != role.get("role_manifest_hash"):
            reasons.append(f"{name} role manifest hash missing or mismatched")
    if guide is not None:
        rate = guide.get("unique_targeting_mapping_rate")
        if guide.get("donor_id") != "D2" or rate is None or not 0 <= float(rate) <= 1:
            reasons.append("D2 guide QC invalid")
    if order is not None:
        genes = order.get("gene_order", [])
        if len(genes) != 2000 or order.get("development_donor") != "D2" or "CP10K" not in str(order.get("algorithm")):
            reasons.append("D2 shared gene order/normalization invalid")
        elif order.get("gene_order_hash") != _gene_order_hash(genes):
            reasons.append("D2 gene order hash mismatch")
    if effects is not None:
        if (effects.get("development_donor") != "D2" or effects.get("response_donors_used") != ["D2"] or
                effects.get("role_manifest_hash") != role.get("role_manifest_hash") or
                set(effects.get("conditions", [])) != {"D2_Rest", "D2_Stim8hr", "D2_Stim48hr"} or
                len(effects.get("gene_order", [])) != 2000 or not effects.get("effect_hash") or
                not _finite_effect_rows(effects, 2000)):
            reasons.append("D2 effect matrix invalid")
        elif effects.get("gene_order") != (order or {}).get("gene_order"):
            reasons.append("D2 effect matrix gene order mismatch")
        elif _canonical_hash(effects, "effect_hash") != effects.get("effect_hash"):
            reasons.append("D2 effect matrix hash mismatch")
    if state is not None:
        if (state.get("development_donor") != "D2" or state.get("response_donors_used") != [] or
                not state.get("split_hash") or not state.get("regions") or not state.get("folds")):
            reasons.append("D2 state region artifact invalid")
    if split is not None:
        if split.get("development_donor") != "D2" or split.get("response_donors_used") != [] or \
                split.get("role_manifest_hash") != role.get("role_manifest_hash"):
            reasons.append("D2 split artifact invalid")
        if _canonical_hash(split, "split_hash") != split.get("split_hash"):
            reasons.append("D2 split hash mismatch")
        for section in ("gene", "gene_x_state", "state_regions"):
            if (not split.get(section, {}).get("train") or
                    not split.get(section, {}).get("validation") or
                    not split.get(section, {}).get("test")):
                reasons.append(f"D2 {section} split missing")
    if csr is not None and (csr.get("status") != "PASS" or csr.get("all_full_csr_read_valid") is not True):
        reasons.append("D2 complete CSR audit did not pass")
    valid = not reasons
    return {
        "status": "DATA_VALID" if valid else "DATA_CONTRACT_READY", "valid": valid,
        "development_donor": "D2", "primary_count": len(primary), "challenge_count": len(challenge),
        "primary_sha256": _sha256(primary_path), "challenge_sha256": _sha256(challenge_path),
        "registry_sha256": _sha256(registry_path), "role_manifest_hash": role.get("role_manifest_hash"),
        "response_donors_used": ["D2"] if effects is not None else [],
        "ntc_donors_used": ["D2"] if order is not None else [],
        "secondary_locked_confirmation": ["D1"], "external_test_donors": ["D3", "D4"],
        "evidence_paths": {name: value for name, value in {
            "guide_qc": "metadata/guide_qc_v1.json", "gene_order": "metadata/shared_gene_order_v2.json",
            "effects": "results/d2_effect_matrix_v3.json", "state_regions": "metadata/continuous_state_regions_v1.json",
            "splits": "metadata/benchmark_splits_v1.json", "csr_audit": "metadata/d2_csr_audit_v1/d2_csr_audit_full_summary.json",
        }.items() if evidence.get(name) is not None},
        "csr_audit_evidence": bool(csr and csr.get("all_full_csr_read_valid") is True),
        "reasons": reasons, "frozen": valid,
        "contract_version": "data_contract.development_D2_v1",
    }


def freeze_development_contract(primary_path: str | Path, challenge_path: str | Path,
                                registry_path: str | Path, role_manifest_path: str | Path,
                                output: str | Path, evidence_dir: str | Path) -> dict:
    result = validate_development_contract(primary_path, challenge_path, registry_path,
                                           role_manifest_path, evidence_dir)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if target.exists() and target.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"refusing to overwrite existing contract: {target}")
    if not target.exists():
        target.write_text(encoded, encoding="utf-8")
    return result
