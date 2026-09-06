import json
import time
from pathlib import Path

import torch
from primeflow.modelcore.nn.dynamics import DynamicsMLP

torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = {
    'gene_expression_input_dim': 8,
    'condition_input_dim': 3,
    'time_input_dim': 16,
    'hidden_dim': 32,
    'n_layers_gene_expression': 1,
    'n_layers_condition': 1,
    'n_layers_time': 1,
    'n_layers_decoding': 1,
    'output_dim': 8,
}
model = DynamicsMLP(**config).to(device).eval()
x = torch.randn(4, 8, device=device)
c = torch.randn(4, 3, device=device)
t = torch.linspace(0.1, 0.9, 4, device=device)
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
t0 = time.perf_counter()
with torch.inference_mode():
    y = model(x, c, t)['gex_out']
elapsed = time.perf_counter() - t0
path = Path('/root/autodl-tmp/CRISPR_perturb_runtime/smoke/primeflow_tiny.pt')
torch.save({'state_dict': model.state_dict(), 'config': config}, path)
payload = torch.load(path, map_location='cpu', weights_only=False)
reloaded = DynamicsMLP(**payload['config']).to(device).eval()
reloaded.load_state_dict(payload['state_dict'])
with torch.inference_mode():
    y2 = reloaded(x, c, t)['gex_out']
result = {
    'model': 'PRiMeFlow DynamicsMLP (tiny smoke; not pretrained)',
    'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
    'torch': torch.__version__,
    'jax': __import__('jax').__version__,
    'input_shape': list(x.shape),
    'output_shape': list(y.shape),
    'checkpoint': str(path),
    'reload_max_abs_diff': float((y - y2).abs().max().cpu()),
    'parameters': sum(p.numel() for p in model.parameters()),
    'elapsed_seconds': elapsed,
    'peak_gpu_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
}
print(json.dumps(result))
