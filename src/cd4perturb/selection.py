from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GeneCandidate:
    gene: str
    tier: str
    expression_bin: str
    guide_coverage: int = 0
    n_guides: int = 0
    effect_bin: str = "unknown"
    guide_concordance: str = "unknown"
    functional_class: str = "unknown"
    stimulus_dependent: bool = False
    selection_reason: str = ""


@dataclass(frozen=True)
class PilotSelection:
    """Two disjoint strata with an explicit response-blind primary set."""
    primary: tuple[GeneCandidate, ...]
    challenge: tuple[GeneCandidate, ...]
    version: str = "pilot_96.frozen_v1"
    primary_response_blind: bool = True

    @property
    def all(self) -> tuple[GeneCandidate, ...]:
        return self.primary + self.challenge


def _row_key(row: dict) -> tuple:
    return (str(row.get("expression_bin", "unknown")),
            -float(row.get("guide_coverage", 0)), str(row["gene"]))


def select_primary_64(rows: Iterable[dict], n_genes: int = 64, donor_id: str = "D1") -> list[GeneCandidate]:
    """Select primary genes using only pre-perturbation/design annotations.

    Effect, stimulus-response and guide-concordance fields are deliberately
    ignored, even when present, so this set remains valid for a blind gate.
    """
    clean = [dict(r) for r in rows if r.get("gene")]
    clean.sort(key=_row_key)
    bins: dict[str, list[dict]] = {}
    for row in clean:
        bins.setdefault(str(row.get("expression_bin", "unknown")), []).append(row)
    chosen: list[dict] = []
    for name in sorted(bins):
        bucket = bins[name]
        take = max(1, round(n_genes * len(bucket) / max(1, len(clean))))
        chosen.extend(bucket[:take])
    chosen = chosen[:n_genes]
    if len(chosen) < n_genes:
        used = {r["gene"] for r in chosen}
        chosen.extend(r for r in clean if r["gene"] not in used)
    chosen = chosen[:n_genes]
    if len(chosen) < n_genes:
        raise ValueError(f"only {len(chosen)} response-blind genes available, need {n_genes}")
    return [GeneCandidate(gene=r["gene"], tier="primary_64",
                          expression_bin=str(r.get("expression_bin", "unknown")),
                          guide_coverage=int(r.get("guide_coverage", 0)), n_guides=int(r.get("n_guides", 0)),
                          functional_class=str(r.get("functional_class", "unknown")),
                          selection_reason=f"{str(donor_id).upper()} pre-perturbation expression/coverage/function strata only")
            for r in chosen]


def select_challenge_32(rows: Iterable[dict], excluded: Iterable[str] = (), n_genes: int = 32,
                        donor_id: str = "D1") -> list[GeneCandidate]:
    """Select a separately reported stress set using development responses."""
    excluded = set(map(str, excluded))
    clean = [dict(r) for r in rows if r.get("gene") and str(r["gene"]) not in excluded]
    clean.sort(key=_row_key)
    chosen: list[dict] = []
    used: set[str] = set()
    # One representative per difficulty axis first; remaining slots are
    # deterministic and remain outside all primary statistics.
    axes = [("effect_bin", "strong"), ("effect_bin", "mid"), ("effect_bin", "weak"),
            ("stimulus_dependent", True), ("guide_concordance", "concordant"),
            ("guide_concordance", "discordant")]
    for field, value in axes:
        for row in clean:
            if row.get(field) == value and str(row["gene"]) not in used:
                chosen.append(row); used.add(str(row["gene"])); break
    for row in clean:
        if len(chosen) >= n_genes:
            break
        if str(row["gene"]) not in used:
            chosen.append(row); used.add(str(row["gene"]))
    if len(chosen) < n_genes:
        raise ValueError(f"only {len(chosen)} challenge genes available, need {n_genes}")
    return [GeneCandidate(gene=r["gene"], tier="challenge_32",
                          expression_bin=str(r.get("expression_bin", "unknown")),
                          guide_coverage=int(r.get("guide_coverage", 0)), n_guides=int(r.get("n_guides", 0)),
                          effect_bin=str(r.get("effect_bin", "unknown")),
                          guide_concordance=str(r.get("guide_concordance", "unknown")),
                          functional_class=str(r.get("functional_class", "unknown")),
                          stimulus_dependent=bool(r.get("stimulus_dependent", False)),
                          selection_reason=f"{str(donor_id).upper()} response/context/guide challenge stress test")
            for r in chosen]


def select_challenge_pool(rows: Iterable[dict], excluded: Iterable[str] = (),
                          n_genes: int = 256, donor_id: str = "D1") -> list[dict]:
    """Freeze a response-blind pool before effect responses are opened."""
    excluded = set(map(str, excluded))
    clean = [dict(r) for r in rows if r.get("gene") and str(r["gene"]) not in excluded]
    clean.sort(key=_row_key)
    if len(clean) < n_genes:
        raise ValueError(f"only {len(clean)} challenge-pool genes available, need {n_genes}")
    return clean[:n_genes]


def select_challenge_from_effects(pool_rows: Iterable[dict], gene_qc: Iterable[dict],
                                  excluded: Iterable[str] = (), n_genes: int = 32,
                                  donor_id: str = "D2") -> tuple[list[GeneCandidate], list[dict]]:
    """Annotate the frozen response-blind pool and select the stress set.

    The pool itself is never changed.  Once the development effect matrix is
    available, compact gene-level QC fields are derived deterministically and
    fed to the predeclared axis-first selector (strong/mid/weak, concordant /
    discordant and stimulus-dependent).  This keeps response access after the
    pool freeze while making the final ``challenge_32`` auditable.
    """
    excluded = set(map(str, excluded))
    rows = [dict(x) for x in pool_rows if x.get("gene") and str(x["gene"]) not in excluded]
    qc = [dict(x) for x in gene_qc]
    per_gene: dict[str, list[dict]] = {}
    stimulus: dict[str, dict] = {}
    for row in qc:
        gene = str(row.get("gene", ""))
        if not gene:
            continue
        if row.get("condition") == "across_conditions":
            stimulus[gene] = row
        else:
            per_gene.setdefault(gene, []).append(row)
    norms = []
    for row in rows:
        gene = str(row["gene"])
        records = per_gene.get(gene, [])
        norm = float(sum(float(x.get("effect_l2", 0.0)) for x in records) / max(len(records), 1))
        consistency = float(sum(float(x.get("guide_consistency", 0.0)) for x in records) / max(len(records), 1))
        stim = stimulus.get(gene, {})
        enriched = {**row, "effect_norm": norm,
                    "guide_concordance": "concordant" if consistency >= 0.75 else "discordant",
                    "stimulus_dependent": bool(float(stim.get("stimulation_dependence_cv", 0.0)) >= 0.5)}
        row.clear(); row.update(enriched)
        norms.append(norm)
    finite = np.asarray([x for x in norms if np.isfinite(x)], dtype=float) if norms else np.zeros(0)
    q1, q2 = (np.quantile(finite, [1 / 3, 2 / 3]) if finite.size else (0.0, 0.0))
    for row in rows:
        value = float(row.get("effect_norm", 0.0))
        row["effect_bin"] = "weak" if value <= q1 else "strong" if value >= q2 else "mid"
    chosen = select_challenge_32(rows, excluded=excluded, n_genes=n_genes, donor_id=donor_id)
    return chosen, rows


def make_pilot_selection(rows: Iterable[dict], primary_genes: int = 64,
                         challenge_genes: int = 32, donor_id: str = "D1") -> PilotSelection:
    rows = [dict(r) for r in rows]
    primary = select_primary_64(rows, primary_genes, donor_id=donor_id)
    challenge = select_challenge_32(rows, (x.gene for x in primary), challenge_genes, donor_id=donor_id)
    if {x.gene for x in primary} & {x.gene for x in challenge}:
        raise ValueError("primary and challenge sets overlap")
    return PilotSelection(tuple(primary), tuple(challenge))


def stratified_pilot(
    rows: Iterable[dict],
    n_genes: int = 96,
    core_genes: int = 64,
    challenge_genes: int = 32,
    donor_id: str = "D1",
) -> list[GeneCandidate]:
    """Deterministic, target-direction-blind pilot selection.

    Core genes are selected from expression/coverage strata only.  Challenge
    genes may use effect strength and guide concordance, but are explicitly
    labelled so primary metrics can report them separately.
    """
    rows = [dict(r) for r in rows if r.get("gene")]
    if n_genes != core_genes + challenge_genes:
        raise ValueError("n_genes must equal core_genes + challenge_genes")
    chosen = make_pilot_selection(rows, core_genes, challenge_genes, donor_id=donor_id)
    # Keep the historical API labels for callers that consume the flat list;
    # the new on-disk split uses the unambiguous primary_64/challenge_32 tiers.
    result = [GeneCandidate(**{**asdict(x), "tier": "core" if x.tier == "primary_64" else "challenge"})
              for x in chosen.all]
    if len(result) < n_genes:
        raise ValueError(f"only {len(result)} eligible genes, need {n_genes}")
    return result[:n_genes]


def write_pilot(rows: Iterable[GeneCandidate], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([asdict(x) for x in rows], ensure_ascii=False, indent=2), encoding="utf-8")


def write_pilot_selection(selection: PilotSelection, root: str | Path, *, donor_id: str = "D1",
                          role_manifest_hash: str | None = None) -> dict[str, str]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    paths = {"primary": root / "pilot_64_primary_v1.json",
             "challenge": root / "pilot_32_challenge_v1.json",
             "manifest": root / "pilot_manifest_v1.json"}
    write_pilot(selection.primary, paths["primary"])
    write_pilot(selection.challenge, paths["challenge"])
    manifest = {"version": selection.version, "primary_count": len(selection.primary),
                "challenge_count": len(selection.challenge), "primary_response_blind": True,
                "primary_gate_scope": "primary_64_only", "challenge_scope": "stress_test_only",
                "primary_sha256": _sha256(paths["primary"]), "challenge_sha256": _sha256(paths["challenge"]),
                "development_donor": str(donor_id).upper(), "response_donors_used": [],
                "ntc_donors_used": [str(donor_id).upper()]}
    if role_manifest_hash is not None:
        manifest["role_manifest_hash"] = str(role_manifest_hash)
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()
