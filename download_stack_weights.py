from huggingface_hub import hf_hub_download
from pathlib import Path

root = Path('/root/autodl-tmp/CRISPR_perturb_runtime/models/huggingface/Stack-Large-Aligned')
root.mkdir(parents=True, exist_ok=True)
for name in ('bc_large_aligned.ckpt', 'basecount_1000per_15000max.pkl', 'MODEL_LICENSE.md', 'MODEL_ACCEPTABLE_USE_POLICY.md', 'README.md'):
    path = hf_hub_download(
        repo_id='arcinstitute/Stack-Large-Aligned',
        filename=name,
        local_dir=str(root),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(name, path)
