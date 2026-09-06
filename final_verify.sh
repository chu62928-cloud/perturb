#!/usr/bin/env bash
set -u
echo '=== disk ==='
df -h /root/autodl-tmp | tail -1
echo '=== gpu ==='
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
echo '=== envs ==='
/root/miniconda3/bin/conda env list | grep -E 'e3_state|e3_stack|e3_primeflow|e3_gears|e3-th-actuator'
echo '=== pip checks ==='
for E in e3_state e3_stack e3_primeflow e3_gears e3-th-actuator; do
  printf '%s: ' "$E"
  /root/miniconda3/bin/conda run -n "$E" python -m pip check 2>&1 | tail -1
done
echo '=== audit ==='
python -c "import json; d=json.load(open('/root/autodl-tmp/CRISPR_perturb_runtime/data_audit/current.json')); print(d['present_count'], d['expected_count'], d['missing_total_bytes'])"
echo '=== models ==='
find /root/autodl-tmp/CRISPR_perturb_runtime/models/huggingface -maxdepth 2 -type f -printf '%p %s\n'
