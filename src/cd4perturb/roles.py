from __future__ import annotations

"""Versioned donor-role protocol for development and locked confirmation."""

import hashlib
import json
import time
from pathlib import Path
from typing import Iterable, Mapping


ROLE_PROTOCOL_VERSION = "donor_roles.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def role_payload(development_donor: str = "D2",
                 secondary_confirmation: Iterable[str] = ("D1",),
                 external_test: Iterable[str] = ("D3", "D4"),
                 *, status: str = "ACTIVE",
                 audit_status: str = "PASS",
                 intent_only: bool = False) -> dict:
    """Build a deterministic role payload; no dataset response is read here."""
    development_donor = str(development_donor).upper()
    secondary = sorted({str(x).upper() for x in secondary_confirmation})
    external = sorted({str(x).upper() for x in external_test})
    if development_donor in secondary or development_donor in external:
        raise ValueError("donor roles must be disjoint")
    if development_donor not in {"D1", "D2", "D3", "D4"}:
        raise ValueError("unsupported development donor")
    return {
        "version": ROLE_PROTOCOL_VERSION,
        "status": str(status),
        "audit_status": str(audit_status),
        "intent_only": bool(intent_only),
        "development_donor": development_donor,
        "secondary_locked_confirmation": secondary,
        "external_test_donors": external,
        "response_donors_used": [],
        "ntc_donors_used": [],
        "scientific_policy": {
            "development_uses": ["guide_qc", "feature_selection", "state_partition",
                                 "effect_matrix", "baseline", "state_adaptation",
                                 "threshold_selection"],
            "secondary_confirmation_rule": "one_time_confirmation_after_model_selection",
            "external_test_rule": "final_confirmation_only",
            "integrity_audit_only_does_not_compute_biological_statistics": True,
        },
    }


def seal_role_manifest(payload: Mapping, *, created_at_epoch: float | None = None) -> dict:
    """Seal a role manifest with a content hash and timestamp."""
    body = dict(payload)
    body.setdefault("created_at_epoch", float(time.time() if created_at_epoch is None else created_at_epoch))
    body.pop("role_manifest_hash", None)
    body["role_manifest_hash"] = _sha256_bytes(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return body


def write_role_manifest(path: str | Path, payload: Mapping) -> dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sealed = seal_role_manifest(payload)
    encoded = json.dumps(sealed, ensure_ascii=False, indent=2)
    if target.exists():
        old = target.read_text(encoding="utf-8")
        if old != encoded:
            raise FileExistsError(f"refusing to overwrite role manifest: {target}")
    else:
        target.write_text(encoded, encoding="utf-8")
    return sealed


def validate_role_manifest(manifest: Mapping, *, require_active: bool = True) -> bool:
    """Validate role disjointness and the immutable content hash."""
    if manifest.get("version") != ROLE_PROTOCOL_VERSION:
        return False
    if require_active and manifest.get("status") != "ACTIVE":
        return False
    development = str(manifest.get("development_donor", "")).upper()
    secondary = {str(x).upper() for x in manifest.get("secondary_locked_confirmation", [])}
    external = {str(x).upper() for x in manifest.get("external_test_donors", [])}
    if not development or development in secondary or development in external or secondary & external:
        return False
    declared = manifest.get("role_manifest_hash")
    if not declared:
        return False
    body = dict(manifest)
    body.pop("role_manifest_hash", None)
    expected = _sha256_bytes(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8"))
    return declared == expected
