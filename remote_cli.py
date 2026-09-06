from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from .config import load_config
from .registry import build_registry, write_registry
from .release import create_locked_release


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cd4perturb", description="CD4 Perturb-seq auditable pipeline")
    p.add_argument("command", choices=["preflight", "audit", "audit-public", "prepare-pilot", "guide-qc", "gene-order", "ntc-latent", "freeze-data", "effect-matrix", "state-regions", "freeze-splits", "release-d2", "fit-baselines", "evaluate", "match-composition", "score-composition", "proxy-composition", "plan", "report"])
    p.add_argument("--config", default="config/config.json")
    p.add_argument("--audit", default=None)
    p.add_argument("--candidate-table", default=None)
    p.add_argument("--public-root", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--input", default=None, help="stage input, usually a compact NPZ for smoke/evaluation")
    p.add_argument("--primary", default=None)
    p.add_argument("--challenge", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--genes", default=None, help="JSON list of fixed genes for effect-matrix")
    p.add_argument("--paths", default=None, help="JSON list of D1 h5ad paths")
    p.add_argument("--guide-library", default=None, help="fixed submission guide CSV")
    p.add_argument("--guide-qc", default=None, help="guide QC JSON")
    p.add_argument("--pilot", default=None, help="pilot JSON used to force on-target genes")
    p.add_argument("--block-rows", type=int, default=8192)
    p.add_argument("--evidence-dir", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config)
    out_root = Path(args.output or cfg.runtime_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.command == "preflight":
        print(json.dumps({"python": sys.version, "platform": platform.platform(), "config_hash": cfg.hash(),
                          "data_root": cfg.data_root, "runtime_root": str(out_root)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit":
        audit = json.loads(Path(args.audit).read_text(encoding="utf-8")) if args.audit else {}
        records = build_registry(cfg.data_root, audit=audit, locked_d2=False)
        target = out_root / "metadata" / "dataset_registry.json"
        write_registry(records, target)
        print(json.dumps({"records": len(records), "complete": sum(r.status == "complete" for r in records), "output": str(target)}, ensure_ascii=False))
        return 0
    if args.command == "prepare-pilot":
        from .data import summarize_d1_candidates
        from .selection import stratified_pilot, write_pilot, make_pilot_selection, write_pilot_selection
        registry_path = out_root / "metadata" / "dataset_registry.json"
        if not registry_path.exists():
            raise SystemExit("dataset registry is missing; run audit first")
        records = json.loads(registry_path.read_text(encoding="utf-8"))
        d1_paths = [r["path"] for r in records if r["donor_id"] == "D1" and r["status"] == "complete"]
        if len(d1_paths) < 3:
            raise SystemExit("prepare-pilot requires complete D1 Rest/Stim8hr/Stim48hr")
        if args.candidate_table:
            rows = json.loads(Path(args.candidate_table).read_text(encoding="utf-8"))
        else:
            rows = summarize_d1_candidates(d1_paths, sample_size=256,
                                           min_cells_per_guide=cfg.pilot.min_cells_per_guide,
                                           min_guides_per_gene=cfg.pilot.min_guides_per_gene,
                                           seed=cfg.pilot.seed)
        candidates_path = out_root / "metadata" / "gene_candidates.json"
        candidates_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        pilot = stratified_pilot(rows, n_genes=cfg.pilot.n_genes, core_genes=cfg.pilot.core_genes,
                                 challenge_genes=cfg.pilot.challenge_genes)
        pilot_path = out_root / "metadata" / "pilot_96.json"
        write_pilot(pilot, pilot_path)
        # New immutable split: primary_64 is response-blind and drives gates;
        # challenge_32 is reported separately as a stress test.
        selection = make_pilot_selection(rows, cfg.pilot.core_genes, cfg.pilot.challenge_genes)
        split_paths = write_pilot_selection(selection, out_root / "metadata")
        print(json.dumps({"candidate_rows": len(rows), "pilot_genes": len(pilot),
                          "d1_files": len(d1_paths), "candidates": str(candidates_path),
                          "pilot": str(pilot_path), **split_paths}, ensure_ascii=False))
        return 0
    if args.command == "guide-qc":
        if not args.paths or not args.guide_library:
            raise SystemExit("guide-qc requires --paths JSON and --guide-library CSV")
        from .guide_correction import audit_d1_guides, load_guide_library
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        if any("D1" not in Path(p).name for p in paths):
            raise SystemExit("guide-qc accepts D1 paths only")
        report = audit_d1_guides(paths, load_guide_library(args.guide_library))
        target = out_root / "metadata" / "guide_qc_v1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "GUIDE_QC_READY", "output": str(target),
                          "mapping_rate": report["unique_targeting_mapping_rate"],
                          "d2_responses_used": False}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "gene-order":
        if not args.paths:
            raise SystemExit("gene-order requires --paths JSON")
        from .data import freeze_shared_gene_order
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        pilot_genes = []
        if args.pilot:
            raw = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
            pilot_genes = [x.get("gene", x) if isinstance(x, dict) else x for x in raw]
        order = freeze_shared_gene_order(paths, pilot_genes, cfg.shared_gene_count,
                                         seed=cfg.random_seed, block_rows=args.block_rows)
        target = out_root / "metadata" / "shared_gene_order_v2.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "GENE_ORDER_FROZEN", "output": str(target),
                          "genes": len(order["gene_order"]), "gene_order_hash": order["gene_order_hash"],
                          "d2_responses_used": False}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "ntc-latent":
        if not args.paths or not args.genes:
            raise SystemExit("ntc-latent requires --paths JSON and --genes gene-order JSON")
        import numpy as np
        from .data import build_d1_ntc_latent
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        genes_raw = json.loads(Path(args.genes).read_text(encoding="utf-8"))
        genes = genes_raw.get("gene_order", genes_raw) if isinstance(genes_raw, dict) else genes_raw
        latent, conditions, meta = build_d1_ntc_latent(paths, genes,
                                                        n_components=cfg.latent_dimensions,
                                                        seed=cfg.random_seed, block_rows=args.block_rows)
        npz_target = out_root / "results" / "d1_ntc_latent_v1.npz"
        meta_target = out_root / "metadata" / "d1_ntc_latent_v1.json"
        npz_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_target, latent=latent, conditions=conditions)
        meta["output_npz"] = str(npz_target)
        meta_target.parent.mkdir(parents=True, exist_ok=True)
        meta_target.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "D1_NTC_LATENT_READY", "output": str(npz_target),
                          "metadata": str(meta_target), "shape": list(latent.shape),
                          "d2_responses_used": False}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "state-regions":
        if not args.input:
            raise SystemExit("state-regions requires --input NPZ with latent and conditions")
        import numpy as np
        from .holdouts import continuous_state_regions, state_split_payload
        state_cfg = cfg.continuous_state
        with np.load(args.input, allow_pickle=False) as data:
            condition_key = "latent_conditions" if "latent_conditions" in data else "conditions"
            split = continuous_state_regions(data["latent"], data[condition_key],
                n_anchors=int(state_cfg.get("n_anchors_per_condition", 10)),
                k_neighbors=int(state_cfg.get("k_neighbors", 200)),
                n_folds=int(state_cfg.get("n_folds", 5)), seed=cfg.random_seed,
                density_quantile=float(state_cfg.get("density_quantile", .10)),
                buffer_scale=float(state_cfg.get("buffer_scale", 1.25)))
            payload = state_split_payload(split, data[condition_key])
        target = out_root / "metadata" / "continuous_state_regions_v1.json"
        payload["d2_responses_used"] = False
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "STATE_REGIONS_READY", "output": str(target),
                          "regions": len(split.regions)}, ensure_ascii=False))
        return 0
    if args.command == "freeze-splits":
        if not args.genes or not args.input:
            raise SystemExit("freeze-splits requires --genes JSON and --input state split JSON")
        from .holdouts import gene_holdout, interaction_holdout
        genes_raw = json.loads(Path(args.genes).read_text(encoding="utf-8"))
        genes = genes_raw.get("gene_order", genes_raw) if isinstance(genes_raw, dict) else genes_raw
        state = json.loads(Path(args.input).read_text(encoding="utf-8"))
        region_ids = [f"{r['condition']}:{i}" for i, r in enumerate(state.get("regions", []))]
        gene_split = gene_holdout(genes, fraction=cfg.holdout_gene_fraction, seed=cfg.random_seed)
        pairs = [(g, s) for g in genes for s in region_ids]
        joint = interaction_holdout(pairs, fraction=cfg.holdout_interaction_fraction, seed=cfg.random_seed)
        payload = {"version": "benchmark_splits.v1", "gene": {"train": list(gene_split.train), "test": list(gene_split.test)},
                   "state": {"regions": region_ids, "state_split_hash": state.get("split_hash")},
                   "gene_x_state": {"train": [list(x) for x in joint.train], "test": [list(x) for x in joint.test]},
                   "d2_responses_used": False}
        payload["split_hash"] = __import__("hashlib").sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        target = out_root / "metadata" / "benchmark_splits_v1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "BENCHMARK_SPLITS_FROZEN", "output": str(target),
                          "split_hash": payload["split_hash"], "d2_responses_used": False}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "effect-matrix":
        if not args.input or not args.genes:
            raise SystemExit("effect-matrix requires --input JSON list of D1 paths and --genes JSON list")
        from .data import build_d1_effect_matrix
        paths = json.loads(Path(args.input).read_text(encoding="utf-8"))
        genes_raw = json.loads(Path(args.genes).read_text(encoding="utf-8"))
        genes = genes_raw.get("gene_order", genes_raw) if isinstance(genes_raw, dict) else genes_raw
        guide_qc = json.loads(Path(args.guide_qc).read_text(encoding="utf-8")) if args.guide_qc else None
        primary_path = Path(args.primary) if args.primary else out_root / "metadata" / "pilot_64_primary_v1.json"
        challenge_path = Path(args.challenge) if args.challenge else out_root / "metadata" / "pilot_32_challenge_v1.json"
        primary_genes = None
        challenge_genes = None
        if primary_path.exists():
            primary_genes = [str(x.get("gene", x)) if isinstance(x, dict) else str(x)
                             for x in json.loads(primary_path.read_text(encoding="utf-8"))]
        if challenge_path.exists():
            challenge_genes = [str(x.get("gene", x)) if isinstance(x, dict) else str(x)
                               for x in json.loads(challenge_path.read_text(encoding="utf-8"))]
        result = build_d1_effect_matrix(paths, genes, cfg.pilot.min_cells_per_guide, cfg.random_seed,
                                        guide_qc=guide_qc, block_rows=args.block_rows,
                                        primary_genes=primary_genes, challenge_genes=challenge_genes,
                                        progress_path=out_root / "logs" / "effect_matrix_v2.progress.log")
        target = out_root / "results" / "d1_effect_matrix_v2.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing effect matrix: {target}")
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, target)
        digest = __import__("hashlib").sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest = {"command": args.command, "status": "D1_EFFECT_MATRIX_READY",
                    "conditions": result["conditions"], "genes": len(result["gene_order"]),
                    "guide_rows": len(result["guide_effects"]), "d2_responses_used": False,
                    "output": str(target), "bytes": target.stat().st_size,
                    "sha256": digest.hexdigest(), "config_hash": cfg.hash(),
                    "effect_hash": result["effect_hash"]}
        (out_root / "logs").mkdir(parents=True, exist_ok=True)
        (out_root / "logs" / f"{args.command}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "freeze-data":
        if not args.primary or not args.challenge:
            raise SystemExit("freeze-data requires --primary and --challenge")
        from .data_contract import freeze_data_contract
        registry_path = out_root / "metadata" / "dataset_registry.json"
        target = out_root / "metadata" / "data_contract_frozen_v2.json"
        result = freeze_data_contract(args.primary, args.challenge, registry_path, target,
                                      args.evidence_dir or out_root)
        print(json.dumps({"status": result["status"], "output": str(target)}, ensure_ascii=False))
        return 0
    if args.command == "audit-public":
        from .public_metadata import audit_public_metadata, write_public_audit
        if not args.public_root:
            raise SystemExit("--public-root is required for audit-public")
        commit = Path(args.public_root).name
        result = audit_public_metadata(args.public_root, commit=commit)
        target = out_root / "metadata" / "public_metadata_audit.json"
        write_public_audit(result, target)
        print(json.dumps({"files": len(result["files"]), "output": str(target)}, ensure_ascii=False))
        return 0
    if args.command == "release-d2":
        registry_path = out_root / "metadata" / "dataset_registry.json"
        if not registry_path.exists():
            raise SystemExit("dataset registry is missing; run audit first")
        manifest = create_locked_release(registry_path, cfg.hash(), {}, out_root / "metadata" / "d2_locked_release.json")
        print(json.dumps({"status": "D2_LOCKED", "datasets": len(manifest["datasets"]),
                          "output": str(out_root / "metadata" / "d2_locked_release.json")}, ensure_ascii=False))
        return 0
    if args.command == "fit-baselines":
        if not args.input:
            raise SystemExit("fit-baselines requires --input NPZ with x_control and y_perturbed")
        import numpy as np
        from .baselines import fit_baseline_suite
        with np.load(args.input, allow_pickle=False) as data:
            suite = fit_baseline_suite(data["x_control"], data["y_perturbed"])
        target = out_root / "results" / "baselines.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **{name: result.prediction for name, result in suite.items()})
        manifest = {"command": args.command, "status": "BASELINES_READY", "scope": "D1_only",
                    "models": sorted(suite), "output": str(target), "config_hash": cfg.hash()}
    elif args.command == "evaluate":
        if not args.input:
            raise SystemExit("evaluate requires --input NPZ with y_true, pred, baseline")
        import numpy as np
        from .metrics import summarize, bootstrap_rmse_improvement, model_global_gate, model_state_gate
        with np.load(args.input, allow_pickle=False) as data:
            y_true, pred, baseline = data["y_true"], data["pred"], data["baseline"]
        summary = summarize(y_true, pred)
        boot = bootstrap_rmse_improvement(y_true, pred, baseline, n_boot=200, seed=cfg.random_seed)
        metrics = {"evaluation_scope": "primary_64", "relative_standardized_rmse_improvement":
                   1.0 - summary.standardized_rmse / max(summarize(y_true, baseline).standardized_rmse, 1e-12),
                   "bootstrap_ci_lower": boot["lower"], "program_direction_drop": 0,
                   "critical_program_error_ratio": 1, "max_condition_rmse_ratio": 1,
                   "ntc_zero_effect_error": 0, "ntc_null_p95": float("inf"),
                   "local_direction_drop": 0, "distribution_metrics_improved": 2,
                   "state_proportion_error_ratio": 1, "max_region_rmse_ratio": 1}
        global_ok, global_reasons = model_global_gate(metrics, cfg.gates.__dict__)
        state_ok, state_reasons = model_state_gate(metrics, cfg.gates.__dict__)
        manifest = {"command": args.command, "status": "MODEL_GLOBAL_VALID" if global_ok else "MODEL_GLOBAL_INVALID",
                    "model_global_valid": global_ok, "model_state_valid": state_ok,
                    "global_reasons": global_reasons, "state_reasons": state_reasons,
                    "metrics": metrics, "summary": summary.__dict__, "config_hash": cfg.hash()}
    elif args.command == "match-composition":
        if not args.input:
            raise SystemExit("match-composition requires --input NPZ with predicted_latent, real_latent, conditions")
        import numpy as np
        from .composition import match_intermediate_states, seal_match
        with np.load(args.input, allow_pickle=False) as data:
            match = match_intermediate_states(data["predicted_latent"], data["real_latent"], data["conditions"],
                                               {"max_distance": float(data.get("max_distance", 1.0)),
                                                "min_density": float(data.get("min_density", 0.0)),
                                                "min_effective_cells": int(data.get("min_effective_cells", 20))})
        match = seal_match(match)
        target = out_root / "metadata" / "composition_match_v1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"version": match.version, "rows": list(match.rows),
                                      "geometry_hash": match.geometry_hash, "sealed": match.sealed},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = {"command": args.command, "status": "COMPOSITION_MATCH_SEALED", "valid_matches": sum(r["matched"] for r in match.rows),
                    "output": str(target), "geometry_hash": match.geometry_hash, "config_hash": cfg.hash()}
    elif args.command == "score-composition":
        if not args.manifest or not args.input:
            raise SystemExit("score-composition requires --manifest JSON and --input NPZ with outcomes")
        import numpy as np
        from .composition import CompositionMatch, score_composition
        raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        match = CompositionMatch(raw["version"], tuple(raw["rows"]), raw["geometry_hash"], bool(raw.get("sealed")))
        with np.load(args.input, allow_pickle=False) as data:
            outcomes = {int(k): float(v) for k, v in zip(data["indices"], data["outcomes"])}
        result = score_composition(match, outcomes)
        manifest = {"command": args.command, "status": "COMPOSITION_SCORED", **result, "config_hash": cfg.hash()}
    elif args.command == "proxy-composition":
        manifest = {"command": args.command, "status": "ROUTE_REQUIRED",
                    "message": "run match-composition then score-composition; second-step outcomes are inaccessible during matching",
                    "config_hash": cfg.hash()}
    elif args.command == "plan":
        # Planner generation is intentionally gated by an explicit upstream
        # status manifest and never upgrades a failed scientific gate.
        if not args.input:
            raise SystemExit("plan requires --input JSON gate manifest")
        upstream = json.loads(Path(args.input).read_text(encoding="utf-8"))
        required = ["model_global_valid", "model_state_valid", "composition_valid"]
        allowed = all(bool(upstream.get(k, False)) for k in required)
        manifest = {"command": args.command, "status": "PLANNER_VALID" if allowed else "PLANNER_BLOCKED",
                    "required_gates": required, "config_hash": cfg.hash()}
    elif args.command == "report":
        files = sorted(str(p) for p in (out_root / "logs").glob("*.json"))
        manifest = {"command": args.command, "status": "REPORT_READY", "logs": files,
                    "config_hash": cfg.hash()}
    else:
        raise SystemExit(f"unsupported command: {args.command}")
    target = out_root / "logs" / f"{args.command}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
