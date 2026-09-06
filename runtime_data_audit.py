#!/usr/bin/env python3
"""Read-only audit for the currently available CD4 Perturb-seq h5ad files."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anndata as ad


EXPECTED_BYTES = {
    "D1_Rest.assigned_guide.h5ad": 142_828_875_662,
    "D1_Stim8hr.assigned_guide.h5ad": 154_774_003_188,
    "D1_Stim48hr.assigned_guide.h5ad": 156_246_167_022,
    "D2_Rest.assigned_guide.h5ad": 140_574_804_978,
    "D2_Stim8hr.assigned_guide.h5ad": 172_796_432_870,
    "D2_Stim48hr.assigned_guide.h5ad": 137_554_355_000,
    "D3_Rest.assigned_guide.h5ad": 138_149_946_574,
    "D3_Stim8hr.assigned_guide.h5ad": 142_656_069_026,
    "D3_Stim48hr.assigned_guide.h5ad": 137_830_572_362,
    "D4_Rest.assigned_guide.h5ad": 118_609_904_492,
    "D4_Stim8hr.assigned_guide.h5ad": 134_582_395_766,
    "D4_Stim48hr.assigned_guide.h5ad": 159_231_588_926,
}


def row_probe(x, row: int, n_vars: int):
    values = x[row, : min(3, n_vars)]
    if hasattr(values, "toarray"):
        values = values.toarray()
    if hasattr(values, "tolist"):
        values = values.tolist()
    return values


def audit_one(path: Path) -> dict:
    result = {"name": path.name, "path": str(path), "size_bytes": path.stat().st_size}
    result["expected_size_bytes"] = EXPECTED_BYTES.get(path.name)
    result["size_matches_expected"] = result["expected_size_bytes"] == result["size_bytes"]
    obj = None
    try:
        obj = ad.read_h5ad(path, backed="r")
        n_obs, n_vars = obj.shape
        first = 0
        middle = n_obs // 2
        last = n_obs - 1
        result.update(
            {
                "hdf5_open_ok": True,
                "shape": [int(n_obs), int(n_vars)],
                "obs_columns": list(obj.obs.columns),
                "var_columns": list(obj.var.columns),
                "layers": list(obj.layers.keys()),
                "obsm": list(obj.obsm.keys()),
                "first_middle_last_probe": {
                    "first": row_probe(obj.X, first, n_vars),
                    "middle": row_probe(obj.X, middle, n_vars),
                    "last": row_probe(obj.X, last, n_vars),
                },
                "obs_names_unique": bool(obj.obs_names.is_unique),
                "var_names_unique": bool(obj.var_names.is_unique),
                "var_first_three": [str(v) for v in obj.var_names[:3]],
            }
        )
    except Exception as exc:  # record and continue so one file cannot hide others
        result["hdf5_open_ok"] = False
        result["read_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if obj is not None and getattr(obj, "file", None) is not None:
            obj.file.close()
    return result


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/CRISPR_perturb")
    output = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else "/root/autodl-tmp/CRISPR_perturb_runtime/data_audit/current.json"
    )
    files = sorted(root.glob("D[1-4]_(Rest|Stim8hr|Stim48hr).assigned_guide.h5ad"))
    # glob does not interpret alternation; filter explicitly for portability.
    files = sorted(
        p
        for p in root.glob("D*.assigned_guide.h5ad")
        if p.name in EXPECTED_BYTES
    )
    report = {
        "checked_at_epoch": __import__("time").time(),
        "root": str(root),
        "expected_count": len(EXPECTED_BYTES),
        "present_count": len(files),
        "files": [audit_one(p) for p in files],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"present_count": len(files), "expected_count": len(EXPECTED_BYTES), "output": str(output)}))
    return 0 if all(f.get("hdf5_open_ok") and f.get("size_matches_expected") for f in report["files"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
