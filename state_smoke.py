#!/usr/bin/env python3
"""Load an official STATE checkpoint and run one small GPU forward pass."""

import json
import time
from pathlib import Path

import torch
from state.tx.models.state_transition import StateTransitionPerturbationModel


ckpt = Path(
    "/root/autodl-tmp/CRISPR_perturb_runtime/models/huggingface/"
    "ST-HVG-Replogle/fewshot/hepg2/checkpoints/last.ckpt"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("STATE smoke test requires CUDA")

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
model = StateTransitionPerturbationModel.load_from_checkpoint(
    str(ckpt), map_location="cpu", weights_only=False
)
model.eval().to(device)

batch_size = 1
sentence_len = int(model.cell_sentence_len)
input_dim = int(model.input_dim)
pert_dim = int(model.pert_dim)
batch = {
    "pert_emb": torch.zeros(batch_size, sentence_len, pert_dim, device=device),
    "ctrl_cell_emb": torch.rand(batch_size, sentence_len, input_dim, device=device),
    "batch": torch.zeros(batch_size, sentence_len, dtype=torch.long, device=device),
}
with torch.inference_mode():
    output = model(batch)
torch.cuda.synchronize()

result = {
    "model": "STATE",
    "checkpoint": str(ckpt),
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "parameters": sum(p.numel() for p in model.parameters()),
    "input_shape": [batch_size, sentence_len, input_dim],
    "output_shape": list(output.shape),
    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    "elapsed_seconds": time.perf_counter() - start,
}
print(json.dumps(result))
