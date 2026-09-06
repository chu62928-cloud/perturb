from __future__ import annotations

"""Audit D1 CSR pointer integrity without reading expression values."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from cd4perturb.data import _load_csr_patch, _read_indptr_slice_h5


def audit(path: Path, patch_path: Path | None = None) -> dict:
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        if isinstance(x, h5py.Dataset):
            return {"path": str(path), "status": "dense", "raw_valid": True,
                    "patched_valid": True, "d2_responses_used": False}
        raw_pointers = np.asarray(x["indptr"][:], dtype=np.int64)
        raw_decreases = np.flatnonzero(np.diff(raw_pointers) < 0)
        raw_zero_after_start = np.flatnonzero(raw_pointers[1:] == 0) + 1
        data_length = int(x["data"].shape[0])
        raw_valid = bool(len(raw_pointers) and raw_pointers[0] == 0 and not len(raw_decreases) and
                         int(raw_pointers[-1]) == data_length)
        patched = None
        patch_meta = None
        if patch_path is not None:
            patched, patch_meta = _load_csr_patch(patch_path, path)
        pointers = _read_indptr_slice_h5(handle, 0, len(raw_pointers), patched)
        decreases = np.flatnonzero(np.diff(pointers) < 0)
        zero_after_start = np.flatnonzero(pointers[1:] == 0) + 1
        patched_valid = bool(len(pointers) and pointers[0] == 0 and not len(decreases) and
                             int(pointers[-1]) == data_length)
        result = {
            "path": str(path), "shape": [int(v) for v in x.attrs["shape"]],
            "indptr_length": int(len(pointers)), "data_length": data_length,
            "first_pointer": int(pointers[0]) if len(pointers) else None,
            "last_pointer": int(pointers[-1]) if len(pointers) else None,
            "decreasing_pointer_count": int(len(decreases)),
            "first_decreasing_row": int(decreases[0]) if len(decreases) else None,
            "zero_pointer_count_after_start": int(len(zero_after_start)),
            "first_zero_row": int(zero_after_start[0]) if len(zero_after_start) else None,
            "raw_valid": raw_valid,
            "raw_decreasing_pointer_count": int(len(raw_decreases)),
            "raw_first_decreasing_row": int(raw_decreases[0]) if len(raw_decreases) else None,
            "raw_zero_pointer_count_after_start": int(len(raw_zero_after_start)),
            "raw_first_zero_row": int(raw_zero_after_start[0]) if len(raw_zero_after_start) else None,
            # This audit reads only X/indptr and therefore proves pointer
            # structure, not that every X/indices/X/data chunk is readable.
            "raw_pointer_valid": raw_valid,
            "patched_pointer_valid": patched_valid,
            "patched_valid": patched_valid,
            "validation_scope": "X/indptr_structure_only",
            "full_csr_read_valid": None,
            "patch_applied": patch_path is not None,
            "patch_sidecar": str(patch_path) if patch_path is not None else None,
            "patch_artifact_hash": patch_meta.get("artifact_hash") if patch_meta else None,
            # Keep the historical `valid` field explicit and interpretation-safe.
            "valid": patched_valid if patch_path is not None else raw_valid,
            "d2_responses_used": False,
        }
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="JSON list containing D1 paths only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch", default=None, help="explicit D1_Rest CSR patch sidecar JSON")
    args = parser.parse_args()
    paths = [Path(x) for x in json.loads(Path(args.paths).read_text(encoding="utf-8") )]
    if not paths or any("D1" not in path.name for path in paths):
        raise SystemExit("D1 paths only")
    patch = Path(args.patch) if args.patch else None
    rows = [audit(path, patch if path.name.startswith("D1_Rest.") else None) for path in paths]
    payload = {"version": "d1_csr_integrity.v2", "files": rows,
               "raw_all_valid": all(row.get("raw_valid", False) for row in rows),
               "patched_all_valid": all(row.get("patched_pointer_valid", False) for row in rows),
               "patched_pointer_all_valid": all(row.get("patched_pointer_valid", False) for row in rows),
               "full_csr_read_valid": None,
               "all_valid": all(row.get("valid", False) for row in rows),
               "csr_patch_applied": patch is not None,
               "patch_sidecar": str(patch) if patch else None,
               "d2_responses_used": False}
    payload["artifact_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                                           separators=(",", ":")).encode()).hexdigest()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
