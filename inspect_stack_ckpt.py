import torch

p = '/root/autodl-tmp/CRISPR_perturb_runtime/models/huggingface/Stack-Large-Aligned/bc_large_aligned.ckpt'
x = torch.load(p, map_location='cpu', weights_only=False)
print(type(x), list(x.keys()))
print(x.get('hyper_parameters', {}))
for k, v in x.items():
    if hasattr(v, 'shape'):
        print(k, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, 'dict_keys', list(v)[:20])
