from __future__ import annotations

import json
from pathlib import Path


FILES = {
    "sgrna_library_metadata.suppl_table.csv": "guide_mapping",
    "guide_kd_efficiency.suppl_table.csv": "guide_efficiency_audit_only",
    "DE_stats.suppl_table.csv": "effect_summary_audit_only",
    "Th1Th2bulkRNAseq_DESeq2_results.csv.gz": "external_reference_audit",
}


def audit_public_metadata(root: str | Path, commit: str) -> dict:
    import pandas as pd

    root = Path(root)
    result = {"commit": commit, "files": []}
    for name, role in FILES.items():
        path = root / name
        entry = {"name": name, "path": str(path), "role": role, "exists": path.exists()}
        if path.exists():
            head = pd.read_csv(path, nrows=0, compression="infer")
            # Counting rows is linear but these summaries are at most 44 MB and
            # are intentionally separate from raw H5AD processing.
            rows = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=100_000, compression="infer"))
            entry.update({"size_bytes": path.stat().st_size, "rows": int(rows), "columns": head.columns.tolist()})
        result["files"].append(entry)
    return result


def write_public_audit(result: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

