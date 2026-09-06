from __future__ import annotations

"""Complete, read-only CSR audit with resumable block evidence.

The source h5ad files are never opened for writing.  Every logical element of
X/indptr, X/indices and X/data is read in bounded blocks; the report records
the source fingerprint and block hashes so a changed source invalidates a
resume.  This is an integrity audit only: it does not calculate perturbation
effects or inspect biological direction.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _file_fingerprint(path: Path, sample_bytes: int = 1024 * 1024) -> dict[str, Any]:
    st = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if st.st_size > sample_bytes:
            handle.seek(max(0, st.st_size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns),
            "sample_sha256": digest.hexdigest()}


def _hash_array(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _combine_hashes(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("ascii"))
    return digest.hexdigest()


def _load_progress(path: Path, source: dict[str, Any], block_rows: int) -> dict[int, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source") != source or int(payload.get("block_rows", -1)) != int(block_rows):
            return {}
        return {int(row["block"]): row for row in payload.get("blocks", [])}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _save_progress(path: Path, source: dict[str, Any], block_rows: int,
                   blocks: dict[int, dict], *, status: str = "IN_PROGRESS") -> None:
    payload = {"version": "csr_audit.progress.v1", "status": status,
               "source": source, "block_rows": int(block_rows),
               "blocks": [blocks[k] for k in sorted(blocks)]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def audit_file(path: str | Path, output: str | Path, *, block_rows: int = 1024,
               progress: str | Path | None = None,
               registered: dict[str, Any] | None = None) -> dict:
    source_path = Path(path)
    output_path = Path(output)
    progress_path = Path(progress) if progress else output_path.with_suffix(output_path.suffix + ".progress.json")
    source_before = _file_fingerprint(source_path)
    blocks: dict[int, dict] = _load_progress(progress_path, source_before, block_rows)
    started = time.time()
    result: dict[str, Any] = {
        "version": "csr_integrity.full.v1", "path": str(source_path),
        "source_before": source_before, "block_rows": int(block_rows),
        "registered": registered,
        "status": "IN_PROGRESS", "full_csr_read_valid": False,
        "validation_scope": "X/indptr_X/indices_X/data_full_stream",
        "failure": None, "blocks_reused": len(blocks), "blocks_read": 0,
    }
    try:
        if registered is not None:
            expected_size = registered.get("observed_size", registered.get("expected_size"))
            expected_mtime = registered.get("mtime")
            expected_fingerprint = registered.get("fingerprint")
            if expected_size is not None and int(expected_size) != source_before["size"]:
                raise ValueError(f"registered size mismatch: {source_before['size']} != {expected_size}")
            if expected_mtime is not None and abs(float(expected_mtime) - source_before["mtime_ns"] / 1e9) > 1.0:
                raise ValueError(f"registered mtime mismatch: {source_before['mtime_ns'] / 1e9} != {expected_mtime}")
            if expected_fingerprint is not None and str(expected_fingerprint) != source_before["sample_sha256"]:
                raise ValueError("registered fingerprint mismatch")
        with h5py.File(source_path, "r") as handle:
            if "X" not in handle or not hasattr(handle["X"], "keys"):
                raise ValueError("X is not a CSR group")
            x = handle["X"]
            required = {"indptr", "indices", "data"}
            missing = sorted(required - set(x.keys()))
            if missing:
                raise ValueError(f"X missing CSR datasets: {missing}")
            shape_attr = x.attrs.get("shape")
            if shape_attr is None or len(shape_attr) != 2:
                raise ValueError("X.shape attribute is missing or invalid")
            n_cells, n_vars = (int(shape_attr[0]), int(shape_attr[1]))
            obs_n = int(handle["obs"]["_index"].shape[0])
            var_n = int(handle["var"]["_index"].shape[0])
            if (n_cells, n_vars) != (obs_n, var_n):
                raise ValueError(f"X shape {(n_cells, n_vars)} differs from obs/var {(obs_n, var_n)}")
            indptr_ds, indices_ds, data_ds = x["indptr"], x["indices"], x["data"]
            if int(indptr_ds.shape[0]) != n_cells + 1:
                raise ValueError(f"indptr length {indptr_ds.shape[0]} != n_cells+1 {n_cells+1}")
            if int(indices_ds.shape[0]) != int(data_ds.shape[0]):
                raise ValueError("indices and data lengths differ")
            raw_indptr_digest = hashlib.sha256()
            indptr_parts: list[str] = []
            previous = None
            for start in range(0, n_cells + 1, max(1, block_rows * 16)):
                stop = min(n_cells + 1, start + max(1, block_rows * 16))
                try:
                    pointers = np.asarray(indptr_ds[start:stop], dtype=np.int64)
                except Exception as exc:
                    raise OSError(f"CSR indptr read failed rows {start}:{stop}: {exc}") from exc
                if not len(pointers):
                    continue
                if previous is not None and int(pointers[0]) < previous:
                    raise ValueError(f"indptr decreases at row {start}")
                decreases = np.flatnonzero(np.diff(pointers) < 0)
                if len(decreases):
                    raise ValueError(f"indptr decreases at row {start + int(decreases[0])}")
                if int(pointers[0]) < 0 or np.any(pointers > int(indices_ds.shape[0])):
                    raise ValueError(f"indptr bounds invalid in rows {start}:{stop}")
                previous = int(pointers[-1])
                raw_indptr_digest.update(np.ascontiguousarray(pointers).tobytes())
                indptr_parts.append(_hash_array(pointers))
            # Re-read exact block pointers to validate all corresponding value slices.
            block_hashes: list[str] = []
            n_blocks = (n_cells + block_rows - 1) // block_rows
            for block in range(n_blocks):
                start = block * block_rows
                stop = min(n_cells, start + block_rows)
                if block in blocks:
                    block_hashes.append(str(blocks[block]["block_sha256"]))
                    result["blocks_reused"] = int(result["blocks_reused"]) + 0
                    continue
                try:
                    pointers = np.asarray(indptr_ds[start:stop + 1], dtype=np.int64)
                    if len(pointers) != stop - start + 1 or np.any(np.diff(pointers) < 0):
                        raise ValueError(f"indptr invalid in rows {start}:{stop}")
                    raw_start, raw_stop = int(pointers[0]), int(pointers[-1])
                    if raw_start < 0 or raw_stop < raw_start or raw_stop > int(indices_ds.shape[0]):
                        raise ValueError(f"CSR bounds invalid rows {start}:{stop}: {raw_start}:{raw_stop}")
                    indices = np.asarray(indices_ds[raw_start:raw_stop])
                    values = np.asarray(data_ds[raw_start:raw_stop])
                except Exception as exc:
                    raise OSError(
                        f"CSR read failed in block {block} rows {start}:{stop} "
                        f"raw={locals().get('raw_start', '?')}:{locals().get('raw_stop', '?')}: {exc}"
                    ) from exc
                if len(indices) != raw_stop - raw_start or len(values) != raw_stop - raw_start:
                    raise ValueError(f"CSR slice length mismatch rows {start}:{stop}")
                if len(indices) and (np.any(indices < 0) or np.any(indices >= n_vars)):
                    raise ValueError(f"indices out of range rows {start}:{stop}")
                if len(values) and (not np.isfinite(values).all() or np.any(values < 0)):
                    raise ValueError(f"data contains non-finite/negative values rows {start}:{stop}")
                block_digest = hashlib.sha256()
                for array in (pointers, indices, values):
                    block_digest.update(np.ascontiguousarray(array).tobytes(order="C"))
                block_record = {"block": block, "row_start": start, "row_end": stop,
                                "raw_start": raw_start, "raw_end": raw_stop,
                                "nnz": raw_stop - raw_start,
                                "block_sha256": block_digest.hexdigest()}
                blocks[block] = block_record
                block_hashes.append(block_record["block_sha256"])
                result["blocks_read"] = int(result["blocks_read"]) + 1
                if block % 16 == 0 or block == n_blocks - 1:
                    _save_progress(progress_path, source_before, block_rows, blocks)
            if int(indptr_ds[0]) != 0:
                raise ValueError("indptr first value is not zero")
            if int(indptr_ds[-1]) != int(indices_ds.shape[0]):
                raise ValueError("indptr final value does not equal indices/data length")
            result.update({
                "shape": [n_cells, n_vars],
                "indptr_length": int(indptr_ds.shape[0]),
                "indices_length": int(indices_ds.shape[0]),
                "data_length": int(data_ds.shape[0]),
                "indptr_sha256": raw_indptr_digest.hexdigest(),
                "block_hash_sha256": _combine_hashes(block_hashes),
                "indptr_chunk_hash_sha256": _combine_hashes(indptr_parts),
            })
        source_after = _file_fingerprint(source_path)
        if source_after != source_before:
            raise ValueError("source file changed during audit")
        result["source_after"] = source_after
        result["status"] = "PASS"
        result["full_csr_read_valid"] = True
    except Exception as exc:  # include HDF5 errors as auditable failures
        result["status"] = "BLOCKED_D2_CSR"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        result["source_after"] = _file_fingerprint(source_path) if source_path.exists() else None
    result["elapsed_seconds"] = time.time() - started
    result["artifact_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True,
                                                         separators=(",", ":")).encode()).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, output_path)
    _save_progress(progress_path, source_before, block_rows, blocks,
                   status=result["status"])
    return result


def audit_many(paths: list[str | Path], output_dir: str | Path, *, block_rows: int = 1024,
               registered: dict[str, dict[str, Any]] | None = None,
               summary_name: str = "d2_csr_audit_full_summary.json") -> dict:
    root = Path(output_dir)
    rows = []
    for path in paths:
        source = Path(path)
        report = root / f"{source.stem}.csr_audit_full.json"
        progress = root / f"{source.stem}.csr_audit_full.progress.json"
        rows.append(audit_file(source, report, block_rows=block_rows, progress=progress,
                                registered=(registered or {}).get(source.name)))
    summary = {"version": "csr_integrity.full.summary.v1", "files": rows,
               "all_full_csr_read_valid": all(r.get("full_csr_read_valid") is True for r in rows),
               "status": "PASS" if all(r.get("full_csr_read_valid") is True for r in rows) else "BLOCKED_D2_CSR",
               "integrity_audit_only": True, "biological_statistics_computed": False}
    summary["artifact_hash"] = hashlib.sha256(json.dumps(summary, sort_keys=True,
                                                           separators=(",", ":")).encode()).hexdigest()
    target = root / summary_name
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="JSON list of D2 h5ad paths")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-rows", type=int, default=1024)
    args = parser.parse_args()
    paths = [Path(x) for x in json.loads(Path(args.paths).read_text(encoding="utf-8"))]
    if len(paths) != 3 or any("D2_" not in p.name for p in paths):
        raise SystemExit("audit-csr requires exactly three D2 paths")
    result = audit_many(paths, args.output_dir, block_rows=args.block_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["all_full_csr_read_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
