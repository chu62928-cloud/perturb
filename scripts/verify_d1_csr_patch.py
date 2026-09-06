from __future__ import annotations

"""Verify selected D1_Rest rows through the explicit indptr patch overlay."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from cd4perturb.data import _load_csr_patch, _read_indptr_slice_h5


DEFAULT_ROWS = tuple(
    list(range(450448, 450455)) +
    list(range(451448, 451455)) +
    list(range(452448, 452455))
)


def verify(path: Path, patch_path: Path, rows: tuple[int, ...] = DEFAULT_ROWS) -> dict:
    overlays, patch = _load_csr_patch(patch_path, path)
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        shape = tuple(int(v) for v in x.attrs["shape"])
        if any(row < 0 or row >= shape[0] for row in rows):
            raise ValueError("requested row is outside D1_Rest/X")
        lo, hi = min(rows), max(rows)
        pointers = _read_indptr_slice_h5(handle, lo, hi + 2, overlays)
        if np.any(np.diff(pointers) < 0):
            raise ValueError("patched pointer window is decreasing")
        records = []
        for row in rows:
            local = row - lo
            start, stop = int(pointers[local]), int(pointers[local + 1])
            if start < 0 or stop < start or stop > int(x["data"].shape[0]):
                raise ValueError(f"patched row {row} has invalid data bounds {start}:{stop}")
            try:
                indices = np.asarray(x["indices"][start:stop])
            except Exception as exc:
                raise RuntimeError(f"X/indices read failed at row {row}, data slice {start}:{stop}: {exc}") from exc
            try:
                values = np.asarray(x["data"][start:stop], dtype=np.float32)
            except Exception as exc:
                raise RuntimeError(f"X/data read failed at row {row}, data slice {start}:{stop}: {exc}") from exc
            if len(indices) != stop - start or len(values) != stop - start:
                raise ValueError(f"row {row} CSR arrays disagree")
            if len(indices) and (np.min(indices) < 0 or np.max(indices) >= shape[1]):
                raise ValueError(f"row {row} has an out-of-range index")
            if not np.isfinite(values).all():
                raise ValueError(f"row {row} has non-finite values")
            # Construct the one-row CSR object only after all bounds checks.
            matrix = csr_matrix((values, indices, np.array([0, stop - start], dtype=np.int64)),
                                shape=(1, shape[1]))
            records.append({"row": row, "start": start, "stop": stop,
                            "nnz": int(matrix.nnz), "indices_min": int(np.min(indices)) if len(indices) else None,
                            "indices_max": int(np.max(indices)) if len(indices) else None,
                            "finite": bool(np.isfinite(matrix.data).all())})
    return {"version": "d1_rest_csr_patch_verify.v1", "path": str(path),
            "patch_sidecar": str(patch_path), "patch_artifact_hash": patch.get("artifact_hash"),
            "rows": records, "all_rows_valid": True, "d2_responses_used": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = verify(Path(args.path), Path(args.patch))
    except Exception as exc:
        result = {"version": "d1_rest_csr_patch_verify.v1", "path": args.path,
                  "patch_sidecar": args.patch, "all_rows_valid": False,
                  "status": "BLOCKED", "error_type": type(exc).__name__,
                  "error": str(exc), "d2_responses_used": False}
        if args.output:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
