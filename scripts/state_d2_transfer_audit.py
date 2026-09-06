#!/usr/bin/env python3
"""Build a semantic Transfer report against the pinned official STATE ckpt.

Run this script only in the isolated e3_state environment.  It writes JSON
metadata, never a model weight, and refuses to infer the official perturbation
vocabulary when its names are not available in the checkpoint/data package.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    from cd4perturb.state_d2_model import make_transfer_report

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_hparams = dict(checkpoint["hyper_parameters"])
    panel = json.loads(Path(args.panel).read_text(encoding="utf-8"))
    vocab = json.loads(Path(args.vocab).read_text(encoding="utf-8"))
    target_genes = list(panel["gene_order"])
    target_perturbations = list(vocab["perturbation_names"])
    source_genes = list(source_hparams.get("gene_names", []))
    if len(source_genes) != 2000 or len(target_genes) != 2000:
        raise ValueError("both official and D2 panels must contain exactly 2,000 genes")

    # The release checkpoint stores perturbation width but not its ordered
    # names.  Unknown source names are deliberately left empty so no index
    # based transfer can occur.
    source_perturbations: list[str] = []
    kwargs = copy.deepcopy(source_hparams)
    for key in ("input_dim", "hidden_dim", "output_dim", "pert_dim", "batch_dim",
                "gene_names", "hvg_dim", "gene_dim"):
        kwargs.pop(key, None)
    kwargs["cell_set_len"] = 32
    kwargs["batch_encoder"] = False
    kwargs["batch_predictor"] = False
    kwargs["gene_names"] = target_genes
    target = StateTransitionPerturbationModel(
        input_dim=2000, hidden_dim=int(source_hparams["hidden_dim"]), output_dim=2000,
        pert_dim=len(target_perturbations), batch_dim=None, gene_dim=2000,
        **kwargs)
    transferred, report = make_transfer_report(
        checkpoint["state_dict"], target.state_dict(), source_genes, target_genes,
        source_perturbations, target_perturbations)
    report.update({"checkpoint": str(args.checkpoint), "checkpoint_hparams_hash":
                   __import__("hashlib").sha256(json.dumps(source_hparams, sort_keys=True).encode()).hexdigest(),
                   "source_perturbation_names_verified": False,
                   "source_perturbation_name_reason": "official checkpoint stores perturbation width but no ordered name vocabulary",
                   "target_gene_order_hash": panel.get("gene_order_hash"),
                   "target_perturbation_vocab_hash": vocab.get("vocab_hash"),
                   "weights_written": False})
    target_path = Path(args.output)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(target_path), "copied_key_count": report["copied_key_count"],
                      "gene_overlap": report["gene_overlap"],
                      "source_perturbation_names_verified": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

