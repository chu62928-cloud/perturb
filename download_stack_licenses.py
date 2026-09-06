from huggingface_hub import hf_hub_download
from pathlib import Path

root = Path('/root/autodl-tmp/CRISPR_perturb_runtime/models/huggingface/Stack-Large-Aligned')
for name in ['MODEL_LICENSE.md', 'MODEL_ACCEPTABLE_USE_POLICY.md', 'README.md']:
    print(hf_hub_download(repo_id='arcinstitute/Stack-Large-Aligned', filename=name, local_dir=str(root)))
