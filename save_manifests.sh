#!/usr/bin/env bash
set -euo pipefail
R=/root/autodl-tmp/CRISPR_perturb_runtime
for E in e3_state e3_stack e3_primeflow e3_gears e3-th-actuator; do
  /root/miniconda3/bin/conda env export -n "$E" --no-builds > "$R/env_manifests/${E}.conda.yml"
  /root/miniconda3/bin/conda list -n "$E" --explicit > "$R/env_manifests/${E}.conda.explicit.txt"
  /root/miniconda3/bin/conda run -n "$E" python -m pip freeze > "$R/env_manifests/${E}.requirements.lock.txt"
  /root/miniconda3/bin/conda run -n "$E" python -m pip check > "$R/env_manifests/${E}.pip_check.txt" 2>&1
done
echo manifests_done
