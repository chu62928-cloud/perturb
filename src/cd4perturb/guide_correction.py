from __future__ import annotations

"""Deterministic correction and quality control for assigned guides.

The fixed public library is the authority for guide identity.  Cell-level
labels are retained as provenance, but they are never silently used to fill a
missing library mapping.  In particular, ``targeting single sgRNA`` is the
only guide group accepted for the single-guide benchmark; multi-guide and
no-guide cells are reported and excluded.
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping


_NTC = re.compile(r"ntc|non[-_ ]?target|control|negative", re.I)
_TARGET = re.compile(r"target", re.I)
SINGLE_GUIDE_GROUP = "targeting single sgRNA"


def _text(value) -> str:
    if value is None:
        return ""
    try:
        if value != value:  # NaN
            return ""
    except Exception:
        pass
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null", "na", "n/a"} else value


def _flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "t", "yes", "y"}


def load_guide_library(path: str | Path) -> list[dict]:
    """Load the fixed CSV library without changing its values."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _library_index(library_rows: Iterable[Mapping]) -> dict[str, list[Mapping]]:
    index: dict[str, list[Mapping]] = defaultdict(list)
    for row in library_rows:
        key = _text(row.get("sgRNA", row.get("guide_id", "")))
        if key:
            index[key].append(row)
    return index


def _correction(original: str, row: Mapping, index: Mapping[str, list[Mapping]]) -> dict:
    matches = list(index.get(original, ()))
    target_ids = {_text(x.get("target_gene_id")) for x in matches}
    target_names = {_text(x.get("target_gene_name")) for x in matches}
    target_ids.discard("")
    target_names.discard("")
    reasons: list[str] = []
    # ``correct_guides`` is also used on compact unit-test rows that predate
    # the explicit group column; real AnnData rows always carry it.
    group = _text(row.get("guide_group", SINGLE_GUIDE_GROUP))
    guide_type = _text(row.get("guide_type"))
    low_quality = _flag(row.get("low_quality", False))
    if group and group != SINGLE_GUIDE_GROUP:
        reasons.append("guide_group_not_single")
    if low_quality:
        reasons.append("low_quality")
    if not matches:
        reasons.append("guide_not_in_fixed_library")
    if len(target_ids) > 1 or len(target_names) > 1:
        reasons.append("library_target_conflict")
    mapped_id = sorted(target_ids)[0] if len(target_ids) == 1 else ""
    mapped_name = sorted(target_names)[0] if len(target_names) == 1 else ""
    flagged = any(_flag(x.get("flag")) or "off-target" in _text(x.get("note")).lower()
                  or "off target" in _text(x.get("note")).lower() for x in matches)
    if flagged:
        reasons.append("library_off_target_flag")
    is_ntc = bool(_NTC.search(guide_type) or _NTC.search(original)) and not (mapped_id or mapped_name)
    if bool(_NTC.search(guide_type) or _NTC.search(original)) and (mapped_id or mapped_name):
        reasons.append("ntc_type_has_target_mapping")
    is_targeting = bool(_TARGET.search(guide_type)) and not bool(_NTC.search(guide_type))
    raw_id = _text(row.get("perturbed_gene_id"))
    raw_name = _text(row.get("perturbed_gene_name"))
    raw_mismatch = bool(mapped_id and raw_id and raw_id != mapped_id) or bool(mapped_name and raw_name and raw_name != mapped_name)
    # A raw-label mismatch is deliberately not a fatal mapping reason: the
    # fixed library is the correction source, while the discrepancy is audited.
    eligible = (group == SINGLE_GUIDE_GROUP and not low_quality and
                not any(x in reasons for x in ("guide_not_in_fixed_library",
                                                "library_target_conflict",
                                                "library_off_target_flag",
                                                "ntc_type_has_target_mapping")) and
                bool(mapped_id or mapped_name or is_ntc))
    return {"original_guide_id": original,
            "corrected_target_gene_id": mapped_id,
            "corrected_target_gene_name": mapped_name,
            "raw_perturbed_gene_id": raw_id,
            "raw_perturbed_gene_name": raw_name,
            "raw_label_mismatch": raw_mismatch,
            "library_rows": len(matches),
            "off_target": flagged,
            "is_ntc": is_ntc,
            "is_targeting": is_targeting,
            "guide_group": group,
            "guide_type": guide_type,
            "low_quality": low_quality,
            "conflict_reason": ";".join(reasons),
            "eligible": bool(eligible)}


def correct_guides(obs_rows: Iterable[Mapping], library_rows: Iterable[Mapping]) -> list[dict]:
    """Attach corrected identity to cell rows while preserving raw labels."""
    index = _library_index(library_rows)
    result = []
    for source in obs_rows:
        row = dict(source)
        original = _text(row.get("guide_id", row.get("sgRNA", "")))
        result.append({**row, **_correction(original, row, index)})
    return result


def audit_guides(paths: Iterable[str | Path], library_rows: Iterable[Mapping],
                 donor_id: str = "D1") -> dict:
    """Audit guide mapping over one donor's conditions using metadata reads.

    Only ``obs`` is read.  The returned report contains one row per observed
    single-guide label and compact condition counts; it does not contain gene
    expression or any D2-derived response.
    """
    import anndata as ad

    index = _library_index(library_rows)
    guide_rows: dict[str, dict] = {}
    condition_counts: dict[str, dict] = {}
    exceptions: dict[str, list[str]] = defaultdict(list)
    donor_id = str(donor_id).upper()
    if donor_id not in {"D1", "D2", "D3", "D4"}:
        raise ValueError(f"unsupported donor_id: {donor_id}")
    paths = [Path(x) for x in paths]
    if not paths or any(not x.name.startswith(f"{donor_id}_") for x in paths):
        raise ValueError(f"guide QC accepts {donor_id} paths only")
    for raw_path in paths:
        path = Path(raw_path)
        obj = ad.read_h5ad(path, backed="r")
        try:
            obs = obj.obs
            required = {"guide_id", "guide_type", "guide_group", "low_quality",
                        "perturbed_gene_name", "perturbed_gene_id"}
            missing = sorted(required.difference(obs.columns))
            if missing:
                raise ValueError(f"{path.name} missing columns: {missing}")
            counts = Counter()
            columns = ["guide_id", "guide_type", "guide_group", "low_quality",
                       "perturbed_gene_name", "perturbed_gene_id"]
            for guide, typ, group, low, raw_name, raw_id in obs[columns].itertuples(index=False, name=None):
                guide = _text(guide)
                if not guide:
                    counts["no_guide"] += 1
                    continue
                item = {"guide_id": guide, "guide_type": typ,
                        "guide_group": group, "low_quality": low,
                        "perturbed_gene_name": raw_name,
                        "perturbed_gene_id": raw_id}
                corrected = _correction(guide, item, index)
                counts["total_assigned"] += 1
                if corrected["guide_group"] != SINGLE_GUIDE_GROUP:
                    counts["excluded_non_single"] += 1
                else:
                    counts["single_guide"] += 1
                    if corrected["low_quality"]:
                        counts["excluded_low_quality"] += 1
                    elif corrected["is_ntc"]:
                        counts["ntc"] += 1
                    elif corrected["is_targeting"]:
                        counts["targeting"] += 1
                        if corrected["corrected_target_gene_id"] or corrected["corrected_target_gene_name"]:
                            counts["mapped_targeting"] += 1
                        else:
                            counts["unmapped_targeting"] += 1
                    if corrected["eligible"]:
                        counts["eligible_single"] += 1
                if guide not in guide_rows:
                    guide_rows[guide] = corrected
                else:
                    old = guide_rows[guide]
                    if (old["corrected_target_gene_id"], old["corrected_target_gene_name"]) != \
                       (corrected["corrected_target_gene_id"], corrected["corrected_target_gene_name"]):
                        exceptions["observed_correction_inconsistent"].append(guide)
                    old["raw_label_mismatch"] |= corrected["raw_label_mismatch"]
                if corrected["raw_label_mismatch"]:
                    exceptions["raw_label_mismatch"].append(f"{path.name}:{guide}")
                if corrected["conflict_reason"]:
                    for reason in corrected["conflict_reason"].split(";"):
                        exceptions[reason].append(f"{path.name}:{guide}")
            total_target = counts["targeting"] + counts["unmapped_targeting"]
            condition_counts[path.stem.split(".")[0]] = {
                "total_cells": int(len(obs)),
                **{k: int(v) for k, v in counts.items()},
                "targeting_cell_mapping_rate": (counts["mapped_targeting"] / total_target
                                                  if total_target else None),
            }
        finally:
            if getattr(obj, "file", None) is not None:
                obj.file.close()
    target_guides = [x for x in guide_rows.values() if x["is_targeting"]]
    mapped_guides = [x for x in target_guides if x["corrected_target_gene_id"] or x["corrected_target_gene_name"]]
    exceptions = {k: sorted(set(v)) for k, v in sorted(exceptions.items())}
    return {
        "version": "guide_qc.v1",
        "guide_group_rule": SINGLE_GUIDE_GROUP,
        "library_target_authority": "fixed_submission_library",
        "guide_rows": [guide_rows[k] for k in sorted(guide_rows)],
        "unique_observed_targeting_guides": len(target_guides),
        "unique_mapped_targeting_guides": len(mapped_guides),
        "unique_targeting_mapping_rate": (len(mapped_guides) / len(target_guides)
                                           if target_guides else None),
        "condition_counts": condition_counts,
        "exceptions": exceptions,
        "donor_id": donor_id,
        "conditions": sorted(condition_counts),
    }


def audit_d1_guides(paths: Iterable[str | Path], library_rows: Iterable[Mapping]) -> dict:
    """Backward-compatible D1 wrapper for the donor-generic guide audit."""
    report = audit_guides(paths, library_rows, donor_id="D1")
    report["d1_conditions"] = list(report["conditions"])
    report["d2_responses_used"] = False
    return report
