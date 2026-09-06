from pathlib import Path

from huggingface_hub import snapshot_download


target = Path("/root/autodl-tmp/CRISPR_perturb_runtime/models/huggingface/ST-HVG-Replogle")
target.mkdir(parents=True, exist_ok=True)
path = snapshot_download(
    repo_id="arcinstitute/ST-HVG-Replogle",
    repo_type="model",
    local_dir=str(target),
    resume_download=True,
)
print(path)
