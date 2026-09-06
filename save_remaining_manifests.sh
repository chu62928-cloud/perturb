#!/usr/bin/env bash
R=/root/autodl-tmp/CRISPR_perturb_runtime
for E in e3_primeflow e3_gears e3-th-actuator; do
  echo "saving $E"
  /root/miniconda3/bin/conda env export -n "$E" --no-builds > "$R/env_manifests/${E}.conda.yml" 2> "$R/env_manifests/${E}.conda.stderr.txt"; echo "conda_export=$?"
  /root/miniconda3/bin/conda list -n "$E" --explicit > "$R/env_manifests/${E}.conda.explicit.txt"; echo "conda_list=$?"
  /root/miniconda3/bin/conda run -n "$E" python -m pip freeze > "$R/env_manifests/${E}.requirements.lock.txt"; echo "pip_freeze=$?"
  /root/miniconda3/bin/conda run -n "$E" python -m pip check > "$R/env_manifests/${E}.pip_check.txt" 2>&1; echo "pip_check=$?"
done
