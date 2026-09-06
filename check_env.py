import importlib.util
import torch
print(torch.__version__, torch.cuda.is_available())
print({n: bool(importlib.util.find_spec(n)) for n in ['torch_geometric','scanpy','anndata','numpy','pandas','scipy','dgl','cell_gears','gears']})
