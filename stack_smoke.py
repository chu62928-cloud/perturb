import json
import time

import torch
from stack.model_loading import load_model_from_checkpoint

checkpoint = '/root/autodl-tmp/CRISPR_perturb_runtime/models/huggingface/Stack-Large-Aligned/bc_large_aligned.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
t0 = time.perf_counter()
model = load_model_from_checkpoint(checkpoint, model_class='ICLFinetunedModel', device=device)
load_seconds = time.perf_counter() - t0
features = torch.poisson(torch.full((1, 8, 15012), 0.2, device=device))
t1 = time.perf_counter()
with torch.inference_mode():
    result = model(features, features, n_kept_cell=4, mask_genes=False, return_loss=False)
forward_seconds = time.perf_counter() - t1
summary = {
    'model': 'STACK-Large-Aligned',
    'checkpoint': checkpoint,
    'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
    'torch': torch.__version__,
    'input_shape': list(features.shape),
    'output_shape': list(result['nb_mean'].shape),
    'cell_embedding_shape': list(result['final_cell_embeddings'].shape),
    'parameters': sum(p.numel() for p in model.parameters()),
    'load_seconds': load_seconds,
    'forward_seconds': forward_seconds,
    'peak_gpu_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
}
print(json.dumps(summary))
