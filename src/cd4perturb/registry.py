from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DATASET_RE = re.compile(r"^(D[1-4])_(Rest|Stim8hr|Stim48hr)\.assigned_guide\.h5ad$")


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    donor_id: str
    condition: str
    path: str
    expected_size: int | None
    observed_size: int
    status: str
    role: str
    mtime: float
    fingerprint: str
    shape: tuple[int, int] | None = None
    obs_columns: tuple[str, ...] = ()
    var_columns: tuple[str, ...] = ()
    notes: str = ""


def fingerprint_file(path: str | Path, sample_bytes: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    size = p.stat().st_size
    with p.open("rb") as fh:
        h.update(fh.read(sample_bytes))
        if size > sample_bytes:
            fh.seek(max(0, size - sample_bytes))
            h.update(fh.read(sample_bytes))
    return h.hexdigest()


def infer_role(donor_id: str, complete: bool, locked: bool = False) -> str:
    if not complete:
        return "incomplete"
    if donor_id == "D1":
        return "development"
    if donor_id == "D2":
        return "locked_eval" if locked else "provisional_external"
    return "external_test"


def build_registry(
    data_root: str | Path,
    expected_sizes: dict[str, int] | None = None,
    audit: dict | None = None,
    locked_d2: bool = False,
) -> list[DatasetRecord]:
    root = Path(data_root)
    expected_sizes = expected_sizes or {}
    by_name = {x.get("name"): x for x in (audit or {}).get("files", [])}
    records: list[DatasetRecord] = []
    for path in sorted(root.glob("D*.assigned_guide.h5ad")):
        match = DATASET_RE.match(path.name)
        if not match:
            continue
        donor, condition = match.groups()
        observed = path.stat().st_size
        expected = expected_sizes.get(path.name) or by_name.get(path.name, {}).get("expected_size_bytes")
        complete = bool(expected is not None and observed == expected)
        audit_row = by_name.get(path.name, {})
        status = "complete" if complete and audit_row.get("hdf5_open_ok", True) else "incomplete_or_unreadable"
        records.append(DatasetRecord(
            dataset_id=f"{donor}_{condition}", donor_id=donor, condition=condition,
            path=str(path), expected_size=expected, observed_size=observed,
            status=status, role=infer_role(donor, status == "complete", locked_d2),
            mtime=path.stat().st_mtime, fingerprint=fingerprint_file(path),
            shape=tuple(audit_row["shape"]) if audit_row.get("shape") else None,
            obs_columns=tuple(audit_row.get("obs_columns", [])),
            var_columns=tuple(audit_row.get("var_columns", [])),
            notes="D2 formal evaluation is locked after first release" if donor == "D2" else "",
        ))
    return records


def write_registry(records: Iterable[DatasetRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2), encoding="utf-8")

