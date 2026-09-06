from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_locked_release(registry_path: str | Path, config_hash: str, model_hashes: dict[str, str],
                          output_path: str | Path, donor_id: str = "D2",
                          release_type: str | None = None) -> dict:
    """Create an immutable one-time donor release; never overwrite it."""
    donor_id = str(donor_id).upper()
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"release manifest already exists: {output}")
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    donor_rows = [row for row in registry if row.get("donor_id") == donor_id and row.get("status") == "complete"]
    if len(donor_rows) < 3:
        raise ValueError(f"{donor_id} release requires complete Rest/Stim8hr/Stim48hr records")
    manifest = {"release_type": release_type or f"{donor_id}_locked_eval", "donor_id": donor_id,
                "created_at_epoch": time.time(),
                "config_hash": config_hash, "model_hashes": dict(model_hashes),
                "dataset_registry_sha256": sha256_file(registry_path),
                "datasets": [{"dataset_id": r["dataset_id"], "fingerprint": r["fingerprint"], "role": "locked_eval"} for r in donor_rows],
                "rule": f"After creation, {donor_id} cannot be used for tuning or repeated formal evaluation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
