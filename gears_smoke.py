import json
import time
from pathlib import Path

import torch
from torch_geometric.data import Data
from gears.model import GEARS_Model

torch.manual_seed(11)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_genes = 8
edge = torch.tensor([[i for i in range(n_genes)], [(i + 1) % n_genes for i in range(n_genes)]], dtype=torch.long)
weight = torch.ones(edge.shape[1], dtype=torch.float32)
edge_go = torch.tensor([[0], [0]], dtype=torch.long)
weight_go = torch.ones(1, dtype=torch.float32)
config = {
    'num_genes': n_genes,
    'num_perts': 1,
    'hidden_size': 16,
    'num_go_gnn_layers': 1,
    'num_gene_gnn_layers': 1,
    'decoder_hidden_size': 8,
    'num_similar_genes_go_graph': 2,
    'num_similar_genes_co_express_graph': 2,
    'coexpress_threshold': 0.4,
    'uncertainty': False,
    'uncertainty_reg': 1,
    'G_go': edge_go,
    'G_go_weight': weight_go,
    'G_coexpress': edge,
    'G_coexpress_weight': weight,
    'device': device,
    'no_perturb': False,
}
model = GEARS_Model(config).to(device).eval()
data = Data(
    x=torch.randn(n_genes, 1, device=device),
    pert_idx=[[0]],
    batch=torch.zeros(n_genes, dtype=torch.long, device=device),
)
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
t0 = time.perf_counter()
with torch.inference_mode():
    pred = model(data)
elapsed = time.perf_counter() - t0
save_dir = Path('/root/autodl-tmp/CRISPR_perturb_runtime/smoke/gears_tiny')
save_dir.mkdir(parents=True, exist_ok=True)
torch.save({'config': {**config, 'device': 'cuda' if device.type == 'cuda' else 'cpu'}, 'state_dict': model.state_dict()}, save_dir / 'model.pt')
model2 = GEARS_Model(config).to(device).eval()
model2.load_state_dict(torch.load(save_dir / 'model.pt', map_location=device, weights_only=False)['state_dict'])
with torch.inference_mode():
    pred2 = model2(data)
result = {
    'model': 'GEARS',
    'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
    'torch': torch.__version__,
    'input_shape': list(data.x.shape),
    'output_shape': list(pred.shape),
    'single_gene_perturbation_index': 0,
    'checkpoint': str(save_dir / 'model.pt'),
    'reload_max_abs_diff': float((pred - pred2).abs().max().cpu()),
    'parameters': sum(p.numel() for p in model.parameters()),
    'elapsed_seconds': elapsed,
    'peak_gpu_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
}
print(json.dumps(result))
