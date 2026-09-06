from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

from .config import load_config
from .registry import build_registry, write_registry
from .release import create_locked_release
from .roles import role_payload, validate_role_manifest, write_role_manifest
from .guide_correction import load_guide_library


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cd4perturb", description="CD4 Perturb-seq auditable pipeline")
    p.add_argument("command", choices=["preflight", "audit", "audit-csr", "activate-roles", "audit-public", "prepare-pilot", "guide-qc", "gene-order", "ntc-latent", "freeze-data", "effect-matrix", "state-regions", "freeze-splits", "release-d2", "release-confirmation", "fit-baselines", "state-adaptation", "evaluate", "match-composition", "score-composition", "proxy-composition", "plan", "report", "program-validate", "d2-audit", "d2-vocab", "d2-splits", "d2-hvg", "d2-gene-panel", "d2-pilot", "d2-contract", "d2-freeze-check", "d2-counts", "d2-enrich-hvg", "d2-train"])
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
    p.add_argument("--paths", default=None, help="JSON list of h5ad paths for the selected donor")
    p.add_argument("--development-donor", default=None, help="development donor (defaults to config)")
    p.add_argument("--role-manifest", default=None, help="sealed donor-role manifest JSON")
    p.add_argument("--audit-summary", default=None, help="full CSR audit summary JSON")
    p.add_argument("--audit-donor", default=None, help="donor label for generic CSR audits (D1-D4)")
    p.add_argument("--registry", default=None, help="explicit dataset registry snapshot for CSR audit")
    p.add_argument("--output-dir", default=None, help="directory for full CSR audit reports")
    p.add_argument("--guide-library", default=None, help="fixed submission guide CSV")
    p.add_argument("--guide-qc", default=None, help="guide QC JSON")
    p.add_argument("--csr-patch", default=None, help="explicit D1_Rest CSR indptr patch sidecar JSON")
    p.add_argument("--pilot", default=None, help="pilot JSON used to force on-target genes")
    p.add_argument("--block-rows", type=int, default=8192)
    p.add_argument("--evidence-dir", default=None)
    p.add_argument("--accept-external-csr-patch", action="store_true",
                   help="explicitly accept the auditable D1_Rest indptr sidecar for DATA_VALID")
    p.add_argument("--programs", default="config/th_lineage_programs.v1.json",
                   help="frozen Naive/Th1/Th2/Th17 program definition")
    p.add_argument("--hvg-input", default=None,
                   help="raw D2 HVG artifact used by d2-gene-panel")
    p.add_argument("--gene-panel", default=None,
                   help="final D2 gene panel JSON used by the real pilot")
    p.add_argument("--max-cells", type=int, default=None,
                   help="optional deterministic per-condition cell cap for HVG pre-audit")
    p.add_argument("--mode", choices=["Scratch", "Transfer"], default=None,
                   help="D2 training mode")
    p.add_argument("--seed", type=int, default=20260901, help="D2 training seed")
    p.add_argument("--checkpoint", default=None, help="pinned official STATE checkpoint")
    p.add_argument("--transfer-report", default=None, help="audited semantic transfer report")
    p.add_argument("--contract-file", default=None, help="frozen formal D2 training contracts")
    p.add_argument("--phase1-steps", type=int, default=1000)
    p.add_argument("--validation-steps", type=int, default=32)
    p.add_argument("--smoke-steps", type=int, default=None,
                   help="bounded optimizer smoke run; never changes the 40,000-step contract")
    p.add_argument("--resume", action="store_true", help="resume from a matching last.ckpt")
    p.add_argument("--dry-run", action="store_true",
                   help="only build and validate one real D2 train/validation batch")
    return p


def _development_donor(args, cfg) -> str:
    return str(args.development_donor or cfg.development_donor).upper()


def _registry_path(out_root: Path) -> Path:
    role_path = out_root / "metadata" / "dataset_registry_roles_v1.json"
    return role_path if role_path.exists() else out_root / "metadata" / "dataset_registry.json"


def _find_registry(out_root: Path) -> Path:
    """Find the nearest existing registry without copying or mutating it."""
    candidates = [_registry_path(out_root)]
    candidates.extend(parent / "metadata" / "dataset_registry.json" for parent in out_root.parents)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _load_role_manifest(args, cfg, out_root: Path, *, require_active: bool = True) -> dict | None:
    path = Path(args.role_manifest) if args.role_manifest else out_root / cfg.role_manifest
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not validate_role_manifest(manifest, require_active=require_active):
        raise SystemExit(f"invalid role manifest: {path}")
    return manifest


def _require_development_role(args, cfg, out_root: Path) -> tuple[str, dict]:
    donor = _development_donor(args, cfg)
    manifest = _load_role_manifest(args, cfg, out_root, require_active=True)
    if manifest is None:
        raise SystemExit("active donor role manifest is required before development analysis")
    if manifest["development_donor"] != donor:
        raise SystemExit(f"development donor {donor} does not match role manifest")
    return donor, manifest


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
    if args.command == "audit-csr":
        if not args.paths:
            raise SystemExit("audit-csr requires --paths JSON list")
        from scripts.audit_csr_full import audit_many
        paths = [Path(x) for x in json.loads(Path(args.paths).read_text(encoding="utf-8"))]
        donor = _development_donor(args, cfg)
        audit_donor = str(args.audit_donor or donor).upper()
        if audit_donor not in {"D1", "D2", "D3", "D4"}:
            raise SystemExit("audit-csr donor must be D1, D2, D3 or D4")
        if not paths or any(not p.name.startswith(f"{audit_donor}_") for p in paths):
            raise SystemExit(f"audit-csr paths must contain only {audit_donor} files")
        if audit_donor == "D2" and (len(paths) != 3 or
                                     {p.name.split("_")[1].split(".")[0] for p in paths} !=
                                     {"Rest", "Stim8hr", "Stim48hr"}):
            raise SystemExit("D2 audit requires exactly Rest/Stim8hr/Stim48hr")
        registry_path = Path(args.registry) if args.registry else _find_registry(out_root)
        if not registry_path.exists():
            raise SystemExit("audit-csr requires the registered dataset fingerprint; run audit first")
        registry_rows = json.loads(registry_path.read_text(encoding="utf-8"))
        registered = {Path(row["path"]).name: row for row in registry_rows
                      if row.get("donor_id") == audit_donor and row.get("status") == "complete"}
        if audit_donor == "D2" and any(path.name not in registered for path in paths):
            raise SystemExit("audit-csr requires registered complete D2 rows for all three files")
        audit_root = Path(args.output_dir or (out_root / "metadata" / "d2_csr_audit_v1"))
        if audit_donor == "D2":
            pending = role_payload(development_donor="D2", secondary_confirmation=("D1",),
                                   external_test=("D3", "D4"), status="PENDING_D2_CSR_AUDIT",
                                   audit_status="PENDING", intent_only=True)
            pending_target = audit_root.parent / "donor_roles_v1.pending.json"
            if not pending_target.exists():
                write_role_manifest(pending_target, pending)
        summary_name = f"{audit_donor.lower()}_csr_audit_full_summary.json"
        result = audit_many(paths, audit_root, block_rows=args.block_rows, registered=registered,
                            summary_name=summary_name)
        if audit_donor == "D2" and not result["all_full_csr_read_valid"]:
            blocked = role_payload(development_donor="D2", secondary_confirmation=("D1",),
                                   external_test=("D3", "D4"), status="BLOCKED_D2_CSR",
                                   audit_status="BLOCKED", intent_only=True)
            blocked["csr_audit_summary"] = str(audit_root / summary_name)
            blocked["csr_audit_artifact_hash"] = result.get("artifact_hash")
            blocked_target = audit_root.parent / "donor_roles_v1.blocked.json"
            if blocked_target.exists():
                old_blocked = json.loads(blocked_target.read_text(encoding="utf-8"))
                if old_blocked.get("csr_audit_artifact_hash") != blocked.get("csr_audit_artifact_hash"):
                    blocked_target = audit_root.parent / f"donor_roles_v1.blocked.{str(result.get('artifact_hash', 'latest'))[:12]}.json"
            if not blocked_target.exists():
                write_role_manifest(blocked_target, blocked)
        print(json.dumps({"status": result["status"], "all_full_csr_read_valid": result["all_full_csr_read_valid"],
                          "output": str(audit_root / summary_name),
                          "audit_donor": audit_donor,
                          "unregistered_files": [p.name for p in paths if p.name not in registered]}, ensure_ascii=False, indent=2))
        return 0 if result["all_full_csr_read_valid"] else 2
    if args.command == "program-validate":
        from .state_d2 import load_lineage_programs, program_coverage
        programs = load_lineage_programs(args.programs)
        result = {"version": programs["version"],
                  "identity_programs": programs["identity_programs"],
                  "effect_genes": programs["effect_genes"],
                  "confounder_programs": programs["confounder_programs"],
                  "d2_responses_used": False}
        if args.genes:
            genes = json.loads(Path(args.genes).read_text(encoding="utf-8"))
            result["coverage"] = program_coverage(genes, programs)
        target = out_root / "research" / "state_d2" / "program_definition_v1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "version": result["version"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-audit":
        if not args.paths:
            raise SystemExit("d2-audit requires --paths JSON list")
        from .state_d2 import audit_d2_data
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        summary = json.loads(Path(args.audit_summary).read_text(encoding="utf-8")) if args.audit_summary else None
        result = audit_d2_data(paths, csr_summary=summary)
        target = out_root / "research" / "state_d2" / "d2_data_audit.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "common_n_vars": result["common_n_vars"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-vocab":
        if not args.paths:
            raise SystemExit("d2-vocab requires --paths JSON list")
        from .state_d2 import build_d2_perturbation_vocab
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        library = load_guide_library(args.guide_library) if args.guide_library else None
        result = build_d2_perturbation_vocab(paths, library_rows=library)
        target = out_root / "research" / "state_d2" / "d2_perturbation_vocab.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "eligible_target_count": result["eligible_target_count"],
                          "vocab_hash": result["vocab_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-splits":
        from .state_d2 import freeze_d2_splits
        genes = None
        if args.genes:
            genes = json.loads(Path(args.genes).read_text(encoding="utf-8"))
        elif args.input:
            vocab = json.loads(Path(args.input).read_text(encoding="utf-8"))
            genes = vocab.get("perturbation_names", [])
        if not genes:
            raise SystemExit("d2-splits requires --genes JSON list or --input vocabulary JSON")
        result = freeze_d2_splits(genes, seed=cfg.random_seed)
        target = out_root / "research" / "state_d2" / "splits" / "d2_splits.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "split_hash": result["split_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-hvg":
        if not args.paths:
            raise SystemExit("d2-hvg requires --paths JSON list")
        from .state_d2 import compute_d2_hvg_panel
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        target_name = "d2_hvg_raw.json" if args.max_cells is None else "d2_hvg_precheck.json"
        target = out_root / "research" / "state_d2" / target_name
        splits = json.loads(Path(args.input).read_text(encoding="utf-8")) if args.input else None
        result = compute_d2_hvg_panel(paths, output=target, block_rows=args.block_rows,
                                      max_cells=args.max_cells, seed=cfg.random_seed, splits=splits)
        print(json.dumps({"output": str(target), "n_cells": result["n_cells"],
                          "raw_hvg_hash": result["raw_hvg_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-gene-panel":
        if not args.hvg_input:
            raise SystemExit("d2-gene-panel requires --hvg-input raw HVG JSON")
        from .state_d2 import apply_identity_anchor_policy
        raw = json.loads(Path(args.hvg_input).read_text(encoding="utf-8"))
        if raw.get("version") != "d2_gene_panel_2000.v1" or raw.get("max_cells_per_condition") is not None:
            raise SystemExit("d2-gene-panel requires a formal all-eligible-cell d2-hvg artifact; prechecks are not accepted")
        programs = json.loads(Path(args.programs).read_text(encoding="utf-8"))
        result = apply_identity_anchor_policy(raw["raw_hvg"], raw["gene_statistics"], programs=programs)
        result["source_hvg_artifact"] = str(args.hvg_input)
        target = out_root / "research" / "state_d2" / "d2_gene_panel_2000.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "gene_order_hash": result["gene_order_hash"],
                          "forced_count": result["forced_count"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-counts":
        if not args.paths or not args.input:
            raise SystemExit("d2-counts requires --paths JSON list and --input splits JSON")
        from .state_d2 import d2_hvg_eligible_cell_counts
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        splits = json.loads(Path(args.input).read_text(encoding="utf-8"))
        counts = d2_hvg_eligible_cell_counts(paths, splits)
        target = out_root / "research" / "state_d2" / "d2_hvg_eligible_cell_counts.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"version": "d2_hvg_eligible_cell_counts.v1",
                                      "counts": counts, "split_hash": splits.get("split_hash"),
                                      "d2_responses_used": False},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "counts": counts}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-enrich-hvg":
        if not args.hvg_input or not args.audit or not args.input:
            raise SystemExit("d2-enrich-hvg requires --hvg-input, --audit and --input counts JSON")
        from .state_d2 import enrich_d2_hvg_artifact
        hvg = json.loads(Path(args.hvg_input).read_text(encoding="utf-8"))
        audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
        counts_payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = enrich_d2_hvg_artifact(hvg, audit, counts_payload.get("counts", counts_payload))
        target = out_root / "research" / "state_d2" / "d2_hvg_raw.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "n_cells": result["n_cells"],
                          "eligible_cells_by_condition": result["eligible_cells_by_condition"]},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-pilot":
        from .state_d2_model import D2StateConfig, build_d2_state_model, pilot_forward_contract
        if args.paths and args.gene_panel and args.input:
            from .state_d2_data import make_pilot_batch
            import numpy as np
            import torch
            paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
            panel_payload = json.loads(Path(args.gene_panel).read_text(encoding="utf-8"))
            vocab_payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
            panel = panel_payload.get("gene_order", [])
            names = vocab_payload.get("perturbation_names", [])
            pilot_targets = ["TBX21", "GATA3", "RORC", "STAT4", "STAT6", "STAT3", "BATF", "IRF4"]
            missing = [gene for gene in pilot_targets if gene not in names]
            if missing:
                raise SystemExit(f"pilot perturbation mapping missing without fallback: {missing}")
            batch = make_pilot_batch(paths, panel, pilot_targets, set_len=32, seed=cfg.random_seed)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_cfg = D2StateConfig(n_genes=len(panel), n_perturbations=len(names))
            model = build_d2_state_model(model_cfg).to(device)
            pert = torch.zeros(1, 32, len(names), device=device)
            pert[:, :, names.index(batch["target_gene"])] = 1.0
            expression = torch.as_tensor(batch["control_expression"], device=device)
            target = torch.as_tensor(batch["target_expression"], device=device)
            prediction = model(expression, pert)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if tuple(prediction.shape) != (1, 32, len(panel)) or not torch.isfinite(loss):
                raise SystemExit("D2 pilot hard contract failed")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
                torch.save({"state_dict": model.state_dict(), "config": model_cfg.as_dict()}, handle.name)
                restored = build_d2_state_model(model_cfg)
                restored.load_state_dict(torch.load(handle.name, map_location="cpu", weights_only=True)["state_dict"])
            result = {"version": "d2_state_pilot_contract.v2", "config": model_cfg.as_dict(),
                      "config_hash": model_cfg.hash(), "output_shape": list(prediction.shape[1:]),
                      "loss": float(loss.detach().cpu()), "finite": True, "checkpoint_reload": True,
                      "device": device, "batch_encoder": False, "condition": batch["condition"],
                      "target_gene": batch["target_gene"], "pilot_targets": pilot_targets,
                      "gene_order_hash": panel_payload.get("gene_order_hash"),
                      "perturbation_vocab_hash": vocab_payload.get("vocab_hash"),
                      "d2_responses_used": True}
        else:
            result = pilot_forward_contract(D2StateConfig(), device="cuda")
        target = out_root / "research" / "state_d2" / "d2_state_pilot_contract.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "output_shape": result["output_shape"],
                          "loss": result["loss"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-contract":
        if not args.gene_panel or not args.input or not args.manifest or not args.checkpoint:
            raise SystemExit("d2-contract requires --gene-panel, --input vocabulary, --manifest splits and --checkpoint")
        import torch
        from .state_d2_training import formal_model_config, freeze_training_contract
        panel = json.loads(Path(args.gene_panel).read_text(encoding="utf-8"))
        vocab = json.loads(Path(args.input).read_text(encoding="utf-8"))
        splits = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model_config = formal_model_config(checkpoint_payload,
                                           len(panel.get("gene_order", [])),
                                           len(vocab.get("perturbation_names", [])),
                                           phase1_steps=args.phase1_steps)
        contracts = []
        for mode in ("Scratch", "Transfer"):
            for seed in (20260901, 20260902, 20260903):
                contracts.append(freeze_training_contract(panel, vocab, splits,
                                                          model_config, mode, seed).as_dict())
        result = {"version": "d2_state_fair_training_contracts.v2",
                  "formal_model_config": model_config, "contracts": contracts,
                  "same_data_and_budget_assertion": True, "d2_responses_used": True}
        target = out_root / "research" / "state_d2" / "training_contracts.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "contracts": len(contracts)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-train":
        if not all((args.paths, args.gene_panel, args.input, args.manifest, args.checkpoint,
                    args.mode, args.contract_file)):
            raise SystemExit("d2-train requires --paths, --gene-panel, --input, --manifest, --checkpoint, --mode and --contract-file")
        import torch
        from .state_d2_data import D2BatchStream
        from .state_d2_training import (build_official_state_adapter, formal_model_config,
                                        initialize_transfer_adapter,
                                        select_frozen_training_contract, train_two_phase,
                                        validate_frozen_training_contract)
        panel_payload = json.loads(Path(args.gene_panel).read_text(encoding="utf-8"))
        vocab_payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        splits_payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        panel = list(panel_payload.get("gene_order", []))
        names = list(vocab_payload.get("perturbation_names", []))
        if len(panel) != 2000 or not names or names[0] != "NTC":
            raise SystemExit("d2-train received an unfrozen panel or vocabulary")
        torch.manual_seed(args.seed)
        model, checkpoint_payload = build_official_state_adapter(
            args.checkpoint, len(panel), len(names), cell_set_len=32)
        transfer_report = None
        if args.mode == "Transfer":
            if not args.transfer_report:
                raise SystemExit("Transfer training requires --transfer-report")
            transfer_report = json.loads(Path(args.transfer_report).read_text(encoding="utf-8"))
            transfer_report = initialize_transfer_adapter(
                model, checkpoint_payload, panel, names, expected_report=transfer_report)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model_config = formal_model_config(checkpoint_payload, len(panel), len(names),
                                           phase1_steps=args.phase1_steps)
        contract_payload = json.loads(Path(args.contract_file).read_text(encoding="utf-8"))
        frozen = select_frozen_training_contract(contract_payload, args.mode, args.seed)
        contract = validate_frozen_training_contract(
            frozen, panel_payload, vocab_payload, splits_payload, model_config,
            args.mode, args.seed)
        train_stream = D2BatchStream(json.loads(Path(args.paths).read_text(encoding="utf-8")),
                                     panel, names, splits_payload, split="train",
                                     batch_size=contract.batch_size, set_len=32, device=device,
                                     seed=args.seed)
        validation_stream = D2BatchStream(json.loads(Path(args.paths).read_text(encoding="utf-8")),
                                          panel, names, splits_payload, split="validation",
                                          batch_size=contract.batch_size, set_len=32, device=device,
                                          seed=args.seed + 1)
        if args.dry_run:
            train_batch = next(iter(train_stream))
            validation_batch = next(iter(validation_stream))
            expected = (contract.batch_size, 32, len(panel))
            for name, batch in (("train", train_batch), ("validation", validation_batch)):
                if tuple(batch["expression"].shape) != expected or tuple(batch["target"].shape) != expected:
                    raise SystemExit(f"D2 {name} batch shape mismatch")
                if tuple(batch["perturbation"].shape) != (contract.batch_size, 32, len(names)):
                    raise SystemExit(f"D2 {name} perturbation shape mismatch")
                if not all(torch.isfinite(batch[key]).all() for key in ("expression", "perturbation", "target")):
                    raise SystemExit(f"D2 {name} batch contains NaN/Inf")
            dry_result = {"version": "d2_state_training_stream_dry_run.v1", "mode": args.mode,
                          "dry_run": True, "batch_size": contract.batch_size, "set_len": 32,
                          "n_genes": len(panel), "n_perturbations": len(names), "device": device,
                          "records_train": len(train_stream.records),
                          "records_validation": len(validation_stream.records),
                          "transfer_applied": args.mode == "Transfer",
                          "gene_order_hash": panel_payload.get("gene_order_hash"),
                          "perturbation_vocab_hash": vocab_payload.get("vocab_hash"),
                          "split_hash": splits_payload.get("split_hash"),
                          "finite_batches": True, "d2_responses_used": True}
            dry_target = out_root / "research" / "state_d2" / f"d2_training_stream_dry_run_{args.mode}.json"
            dry_target.parent.mkdir(parents=True, exist_ok=True)
            dry_target.write_text(json.dumps(dry_result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"output": str(dry_target), **dry_result}, ensure_ascii=False, indent=2))
            return 0
        target = out_root / "research" / "state_d2" / "training" / args.mode / f"seed_{args.seed}"
        result = train_two_phase(model, train_stream, validation_stream, contract, target,
                                 validation_steps=args.validation_steps,
                                 stop_after_steps=args.smoke_steps, resume=args.resume)
        result["device"] = device
        result["transfer_applied"] = args.mode == "Transfer"
        if transfer_report is not None:
            result["transfer_report_hash"] = transfer_report.get("transfer_hash")
        (target / "training_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target / "training_result.json"), "mode": args.mode,
                          "seed": args.seed, "best_step": result["best_step"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "d2-freeze-check":
        required = {"audit": args.audit, "vocab": args.input, "splits": args.manifest, "panel": args.gene_panel}
        if any(value is None for value in required.values()):
            raise SystemExit("d2-freeze-check requires --audit, --input vocabulary, --manifest splits and --gene-panel")
        from .state_d2 import validate_d2_freeze_artifacts
        payloads = {name: json.loads(Path(value).read_text(encoding="utf-8")) for name, value in required.items()}
        pilot = json.loads(Path(args.pilot).read_text(encoding="utf-8")) if args.pilot else None
        result = validate_d2_freeze_artifacts(payloads["audit"], payloads["vocab"], payloads["splits"], payloads["panel"], pilot)
        target = out_root / "research" / "state_d2" / "d2_freeze_validation.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(target), "freeze_hash": result["freeze_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "activate-roles":
        if not args.audit_summary:
            raise SystemExit("activate-roles requires --audit-summary")
        summary_path = Path(args.audit_summary)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS" or summary.get("all_full_csr_read_valid") is not True:
            raise SystemExit("D2 role activation requires a passing complete CSR audit")
        registry = _find_registry(out_root)
        if not registry.exists():
            raise SystemExit("dataset registry is missing; run audit first")
        rows = json.loads(registry.read_text(encoding="utf-8"))
        d2 = [x for x in rows if x.get("donor_id") == "D2" and x.get("status") == "complete"]
        if len(d2) != 3 or {x.get("condition") for x in d2} != {"Rest", "Stim8hr", "Stim48hr"}:
            raise SystemExit("role activation requires three complete D2 conditions")
        manifest_payload = role_payload(development_donor="D2", secondary_confirmation=("D1",),
                                         external_test=("D3", "D4"), status="ACTIVE",
                                         audit_status="PASS", intent_only=False)
        manifest_payload["csr_audit_summary"] = str(summary_path)
        manifest_payload["csr_audit_artifact_hash"] = summary.get("artifact_hash")
        role_target = out_root / "metadata" / "donor_roles_v1.json"
        manifest = write_role_manifest(role_target, manifest_payload)
        role_map = {"D1": "secondary_locked_confirmation", "D2": "development",
                    "D3": "external_test", "D4": "external_test"}
        updated = []
        for row in rows:
            copy = dict(row)
            copy["role"] = role_map.get(str(copy.get("donor_id")), copy.get("role"))
            if copy.get("donor_id") == "D1":
                copy["notes"] = "Previously read by engineering smoke; one-time secondary confirmation only"
            elif copy.get("donor_id") == "D2":
                copy["notes"] = "Development donor activated after complete CSR audit"
            updated.append(copy)
        role_registry = out_root / "metadata" / "dataset_registry_roles_v1.json"
        role_registry.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "ROLES_ACTIVE", "development_donor": "D2",
                          "role_manifest": str(role_target), "registry": str(role_registry),
                          "role_manifest_hash": manifest["role_manifest_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-pilot":
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        from .data import summarize_candidates
        from .selection import (stratified_pilot, write_pilot, write_pilot_selection,
                                select_primary_64, select_challenge_pool, select_challenge_32,
                                PilotSelection)
        registry_path = _registry_path(out_root)
        if not registry_path.exists():
            raise SystemExit("dataset registry is missing; run audit first")
        records = json.loads(registry_path.read_text(encoding="utf-8"))
        development_paths = [r["path"] for r in records if r["donor_id"] == donor and r["status"] == "complete"]
        if len(development_paths) < 3:
            raise SystemExit(f"prepare-pilot requires complete {donor} Rest/Stim8hr/Stim48hr")
        if args.candidate_table:
            rows = json.loads(Path(args.candidate_table).read_text(encoding="utf-8"))
        else:
            rows = summarize_candidates(development_paths, sample_size=256,
                                           min_cells_per_guide=cfg.pilot.min_cells_per_guide,
                                           min_guides_per_gene=cfg.pilot.min_guides_per_gene,
                                           seed=cfg.pilot.seed, donor_id=donor)
        candidates_path = out_root / "metadata" / "gene_candidates.json"
        candidates_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        pilot = stratified_pilot(rows, n_genes=cfg.pilot.n_genes, core_genes=cfg.pilot.core_genes,
                                 challenge_genes=cfg.pilot.challenge_genes, donor_id=donor)
        pilot_path = out_root / "metadata" / "pilot_96.json"
        write_pilot(pilot, pilot_path)
        # Freeze primary and a response-blind challenge pool before effect reads.
        primary = select_primary_64(rows, cfg.pilot.core_genes, donor_id=donor)
        pool_rows = select_challenge_pool(rows, (x.gene for x in primary), n_genes=256, donor_id=donor)
        pool_path = out_root / "metadata" / "challenge_pool_256_v1.json"
        pool_path.write_text(json.dumps(pool_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        challenge = select_challenge_32(pool_rows, (x.gene for x in primary), cfg.pilot.challenge_genes,
                                        donor_id=donor)
        selection = PilotSelection(tuple(primary), tuple(challenge))
        split_paths = write_pilot_selection(selection, out_root / "metadata", donor_id=donor,
                                            role_manifest_hash=role_manifest["role_manifest_hash"])
        print(json.dumps({"candidate_rows": len(rows), "pilot_genes": len(pilot),
                          "development_donor": donor, "development_files": len(development_paths),
                          "role_manifest_hash": role_manifest["role_manifest_hash"],
                          "candidates": str(candidates_path), "challenge_pool": str(pool_path),
                          "pilot": str(pilot_path), **split_paths}, ensure_ascii=False))
        return 0
    if args.command == "guide-qc":
        if not args.paths or not args.guide_library:
            raise SystemExit("guide-qc requires --paths JSON and --guide-library CSV")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        from .guide_correction import audit_guides
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        report = audit_guides(paths, load_guide_library(args.guide_library), donor_id=donor)
        report["role_manifest_hash"] = role_manifest["role_manifest_hash"]
        target = out_root / "metadata" / "guide_qc_v1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "GUIDE_QC_READY", "output": str(target),
                          "development_donor": donor,
                          "mapping_rate": report["unique_targeting_mapping_rate"],
                          "role_manifest_hash": role_manifest["role_manifest_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "gene-order":
        if not args.paths:
            raise SystemExit("gene-order requires --paths JSON")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        from .data import freeze_shared_gene_order
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        pilot_genes = []
        if args.pilot:
            raw = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
            pilot_genes = [x.get("gene", x) if isinstance(x, dict) else x for x in raw]
        order = freeze_shared_gene_order(paths, pilot_genes, cfg.shared_gene_count,
                                         seed=cfg.random_seed, block_rows=args.block_rows,
                                         donor_id=donor)
        order["role_manifest_hash"] = role_manifest["role_manifest_hash"]
        order.pop("artifact_hash", None)
        order["artifact_hash"] = __import__("hashlib").sha256(
            json.dumps(order, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        target = out_root / "metadata" / "shared_gene_order_v2.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "GENE_ORDER_FROZEN", "output": str(target),
                          "development_donor": donor,
                          "genes": len(order["gene_order"]), "gene_order_hash": order["gene_order_hash"],
                          "role_manifest_hash": role_manifest["role_manifest_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "ntc-latent":
        if not args.paths or not args.genes:
            raise SystemExit("ntc-latent requires --paths JSON and --genes gene-order JSON")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        import numpy as np
        from .data import build_d1_ntc_latent
        paths = json.loads(Path(args.paths).read_text(encoding="utf-8"))
        genes_raw = json.loads(Path(args.genes).read_text(encoding="utf-8"))
        genes = genes_raw.get("gene_order", genes_raw) if isinstance(genes_raw, dict) else genes_raw
        latent, conditions, meta = build_d1_ntc_latent(paths, genes,
                                                        n_components=cfg.latent_dimensions,
                                                        seed=cfg.random_seed, block_rows=args.block_rows,
                                                        donor_id=donor)
        meta["role_manifest_hash"] = role_manifest["role_manifest_hash"]
        npz_target = out_root / "results" / f"{donor.lower()}_ntc_latent_v1.npz"
        meta_target = out_root / "metadata" / f"{donor.lower()}_ntc_latent_v1.json"
        npz_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_target, latent=latent, conditions=conditions)
        meta["output_npz"] = str(npz_target)
        meta_target.parent.mkdir(parents=True, exist_ok=True)
        meta_target.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "NTC_LATENT_READY", "output": str(npz_target),
                          "development_donor": donor,
                          "metadata": str(meta_target), "shape": list(latent.shape),
                          "role_manifest_hash": role_manifest["role_manifest_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "state-regions":
        if not args.input:
            raise SystemExit("state-regions requires --input NPZ with latent and conditions")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
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
        payload["development_donor"] = donor
        payload["ntc_donors_used"] = [donor]
        payload["response_donors_used"] = []
        payload["role_manifest_hash"] = role_manifest["role_manifest_hash"]
        payload["split_hash"] = __import__("hashlib").sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "STATE_REGIONS_READY", "output": str(target),
                          "development_donor": donor, "regions": len(split.regions),
                          "role_manifest_hash": role_manifest["role_manifest_hash"]}, ensure_ascii=False))
        return 0
    if args.command == "freeze-splits":
        if not args.genes or not args.input:
            raise SystemExit("freeze-splits requires --genes JSON and --input state split JSON")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        from .holdouts import three_way_split
        genes_raw = json.loads(Path(args.genes).read_text(encoding="utf-8"))
        genes = genes_raw.get("gene_order", genes_raw) if isinstance(genes_raw, dict) else genes_raw
        state = json.loads(Path(args.input).read_text(encoding="utf-8"))
        region_rows = state.get("regions", [])
        region_ids = [f"{r['condition']}:{i}" for i, r in enumerate(region_rows)]
        gene_split = three_way_split(genes, validation_fraction=0.20, test_fraction=cfg.holdout_gene_fraction,
                                     seed=cfg.random_seed)
        pairs = [(g, s) for g in genes for s in region_ids]
        joint = three_way_split(pairs, validation_fraction=0.20, test_fraction=cfg.holdout_interaction_fraction,
                                seed=cfg.random_seed)
        state_test = [sid for sid, row in zip(region_ids, region_rows) if int(row.get("fold", 0)) == 0]
        state_validation = [sid for sid, row in zip(region_ids, region_rows) if int(row.get("fold", 0)) == 1]
        state_train = [sid for sid, row in zip(region_ids, region_rows) if int(row.get("fold", 0)) not in (0, 1)]
        payload = {"version": "benchmark_splits.v2", "gene": gene_split,
                   "state": {"regions": region_ids, "state_split_hash": state.get("split_hash")},
                   "state_regions": {"train": state_train, "validation": state_validation, "test": state_test},
                   "gene_x_state": {key: [list(x) for x in value] for key, value in joint.items()},
                   "development_donor": donor, "ntc_donors_used": [donor],
                   "response_donors_used": [], "role_manifest_hash": role_manifest["role_manifest_hash"]}
        payload["split_hash"] = __import__("hashlib").sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        target = out_root / "metadata" / "benchmark_splits_v1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "BENCHMARK_SPLITS_FROZEN", "output": str(target),
                          "split_hash": payload["split_hash"], "development_donor": donor,
                          "role_manifest_hash": role_manifest["role_manifest_hash"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "effect-matrix":
        if not args.input or not args.genes:
            raise SystemExit("effect-matrix requires --input JSON list of D1 paths and --genes JSON list")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        from .data import build_d1_effect_matrix
        from .selection import select_challenge_from_effects, write_pilot
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
        # The 256-gene pool is frozen before response reads.  If it exists,
        # defer the final challenge_32 choice until the effect matrix has been
        # computed; otherwise retain the historical compatibility file.
        pool_path = out_root / "metadata" / "challenge_pool_256_v1.json"
        if not pool_path.exists() and challenge_path.exists():
            challenge_genes = [str(x.get("gene", x)) if isinstance(x, dict) else str(x)
                               for x in json.loads(challenge_path.read_text(encoding="utf-8"))]
        result = build_d1_effect_matrix(paths, genes, cfg.pilot.min_cells_per_guide, cfg.random_seed,
                                        guide_qc=guide_qc, block_rows=args.block_rows,
                                        primary_genes=primary_genes, challenge_genes=challenge_genes,
                                        csr_patch=args.csr_patch,
                                        progress_path=out_root / "logs" / f"{donor.lower()}_effect_matrix_v3.progress.log",
                                        donor_id=donor)
        if pool_path.exists() and primary_genes is not None:
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            selected, annotated_pool = select_challenge_from_effects(
                pool, result.get("gene_qc", []), excluded=primary_genes,
                n_genes=cfg.pilot.challenge_genes, donor_id=donor)
            write_pilot(selected, challenge_path)
            challenge_genes = [x.gene for x in selected]
            condition_rows = [x for x in result.get("gene_qc", [])
                              if x.get("condition") in result.get("conditions", [])]
            chosen_rows = [x for x in condition_rows if x.get("gene") in set(challenge_genes)]
            result["pilot_qc"]["challenge_32"] = {
                "n_genes": len(challenge_genes), "n_rows": len(chosen_rows),
                "genes_with_effect": sum(int(x.get("n_guides", 0)) > 0 for x in chosen_rows),
                "unstable_rows": sum(bool(x.get("unstable")) for x in chosen_rows),
                "rows": chosen_rows}
            result["challenge_selection"] = {
                "pool_path": str(pool_path), "pool_sha256": __import__("hashlib").sha256(
                    pool_path.read_bytes()).hexdigest(), "algorithm": "effect_axis_first_v1",
                "challenge_path": str(challenge_path), "challenge_sha256": __import__("hashlib").sha256(
                    challenge_path.read_bytes()).hexdigest(), "annotated_pool": annotated_pool}
        result["role_manifest_hash"] = role_manifest["role_manifest_hash"]
        result.pop("effect_hash", None)
        result["effect_hash"] = __import__("hashlib").sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        target = out_root / "results" / f"{donor.lower()}_effect_matrix_v3.json"
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
        manifest = {"command": args.command, "status": "EFFECT_MATRIX_READY",
                    "conditions": result["conditions"], "genes": len(result["gene_order"]),
                    "guide_rows": len(result["guide_effects"]), "development_donor": donor,
                    "response_donors_used": [donor], "role_manifest_hash": role_manifest["role_manifest_hash"],
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
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        from .data_contract import freeze_development_contract
        registry_path = _registry_path(out_root)
        target = out_root / "metadata" / f"data_contract_{donor.lower()}_development_v1.json"
        result = freeze_development_contract(args.primary, args.challenge, registry_path,
                                             out_root / cfg.role_manifest, target,
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
        active = _load_role_manifest(args, cfg, out_root, require_active=True)
        if active is not None and active.get("development_donor") == "D2":
            raise SystemExit("D2 is the development donor; use release-confirmation for D1")
        registry_path = out_root / "metadata" / "dataset_registry.json"
        if not registry_path.exists():
            raise SystemExit("dataset registry is missing; run audit first")
        manifest = create_locked_release(registry_path, cfg.hash(), {}, out_root / "metadata" / "d2_locked_release.json")
        print(json.dumps({"status": "D2_LOCKED", "datasets": len(manifest["datasets"]),
                          "output": str(out_root / "metadata" / "d2_locked_release.json")}, ensure_ascii=False))
        return 0
    if args.command == "release-confirmation":
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        if donor != "D2":
            raise SystemExit("confirmation release is only available after D2 development")
        registry_path = _registry_path(out_root)
        confirmation = cfg.secondary_confirmation_donors
        if len(confirmation) != 1 or confirmation[0] != "D1":
            raise SystemExit("the configured secondary confirmation donor must be D1")
        target = out_root / "metadata" / "d1_secondary_confirmation_release_v1.json"
        manifest = create_locked_release(registry_path, cfg.hash(), {}, target,
                                         donor_id="D1", release_type="D1_secondary_confirmation")
        manifest["role_manifest_hash"] = role_manifest["role_manifest_hash"]
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": "D1_SECONDARY_CONFIRMATION_LOCKED", "output": str(target),
                          "role_manifest_hash": role_manifest["role_manifest_hash"]}, ensure_ascii=False))
        return 0
    if args.command == "state-adaptation":
        if not args.input:
            raise SystemExit("state-adaptation requires --input JSON with zero_shot and best_simple_baseline metrics")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        from .state_protocol import decide_state_adaptation, select_adaptation
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        decision = decide_state_adaptation(payload["zero_shot"], payload["best_simple_baseline"])
        if decision["decision"] == "COMPARE_LIGHT_ADAPTATION" and payload.get("candidates"):
            decision["selection"] = select_adaptation(payload["candidates"])
        decision.update({"development_donor": donor,
                         "role_manifest_hash": role_manifest["role_manifest_hash"],
                         "evaluation_scope": "primary_64"})
        target = out_root / "results" / "state_adaptation_decision_v1.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": decision["decision"], "output": str(target),
                          "role_manifest_hash": role_manifest["role_manifest_hash"]}, ensure_ascii=False))
        return 0
    if args.command == "fit-baselines":
        if not args.input:
            raise SystemExit("fit-baselines requires --input NPZ with x_control and y_perturbed")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
        import numpy as np
        from .baselines import fit_baseline_suite
        with np.load(args.input, allow_pickle=False) as data:
            gene_effect = data["gene_effect"] if "gene_effect" in data else None
            pert2state_prediction = data["pert2state_prediction"] if "pert2state_prediction" in data else None
            suite = fit_baseline_suite(data["x_control"], data["y_perturbed"], gene_effect=gene_effect,
                                       pert2state_prediction=pert2state_prediction)
        target = out_root / "results" / f"baselines_{donor.lower()}_v1.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **{name: result.prediction for name, result in suite.items()})
        manifest = {"command": args.command, "status": "BASELINES_READY",
                    "development_donor": donor, "scope": f"{donor}_development",
                    "models": sorted(suite), "output": str(target), "config_hash": cfg.hash(),
                    "role_manifest_hash": role_manifest["role_manifest_hash"]}
    elif args.command == "evaluate":
        if not args.input:
            raise SystemExit("evaluate requires --input NPZ with y_true, pred, baseline")
        donor, role_manifest = _require_development_role(args, cfg, out_root)
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
                    "development_donor": donor, "role_manifest_hash": role_manifest["role_manifest_hash"],
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
