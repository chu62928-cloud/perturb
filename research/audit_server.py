"""Read-only audit; imports the existing SSH configuration without printing it."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connect_server import create_ssh_client

REMOTE = r'''
import gc, json, os, pathlib, subprocess
import h5py, anndata as ad, numpy as np, pandas as pd
from datetime import datetime

base = pathlib.Path('/root/autodl-tmp/CRISPR_perturb')
vfs = os.statvfs(base)
report = {'checked_at': datetime.now().isoformat(), 'files': [],
          'disk_available_bytes': vfs.f_bavail*vfs.f_frsize,
          'gpu_report': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'], text=True).strip()}
limit = pathlib.Path('/sys/fs/cgroup/memory.max')
report['cgroup_memory_max'] = limit.read_text().strip() if limit.exists() else None
var_indexes = {}
for path in sorted(base.glob('*.h5ad')):
    item = {'name': path.name, 'size_bytes': path.stat().st_size}
    try:
        with h5py.File(path, 'r') as f:
            item['shape'] = [int(x) for x in f['X'].attrs['shape']]
            item['stored_entries'] = int(f['X/data'].shape[0])
            item['hdf5_open_ok'] = True
        a = ad.read_h5ad(path, backed='r')
        obs = a.obs
        good = ~obs['low_quality'].astype(bool)
        single = obs['guide_group'].astype(str).str.contains('single', case=False)
        ntc = good & single & obs['guide_type'].eq('non-targeting')
        target = good & single & obs['guide_type'].eq('targeting') & obs['perturbed_gene_id'].notna()
        item['eligible_ntc_cells'] = int(ntc.sum())
        item['eligible_target_cells_before_curation'] = int(target.sum())
        item['eligible_target_gene_categories_before_curation'] = int(obs.loc[target, 'perturbed_gene_id'].nunique())
        item['guide_group_type_counts'] = [
            {'guide_group': str(k[0]), 'guide_type': str(k[1]), 'n': int(v)}
            for k,v in obs.groupby(['guide_group','guide_type'], observed=True, dropna=False).size().items()]
        idx = sorted(set([0, a.n_obs//2, a.n_obs-1]))
        sample = a.X[idx, :]
        item['first_middle_last_row_read_ok'] = True
        item['sample_row_sums_match_total_counts'] = bool(np.allclose(np.asarray(sample.sum(axis=1)).ravel(), obs.iloc[idx]['total_counts']))
        item['sample_values_nonnegative_integer_like'] = bool(np.all(sample.data >= 0) and np.allclose(sample.data, np.rint(sample.data)))
        item['obs_names_unique'] = bool(a.obs_names.is_unique)
        item['var_names_unique'] = bool(a.var_names.is_unique)
        item['var_first_three'] = a.var_names[:3].tolist()
        var_indexes[path.name] = a.var_names.tolist()
        a.file.close()
        del a, obs, sample
        gc.collect()
    except Exception as exc:
        item['read_error'] = f'{type(exc).__name__}: {exc}'
    report['files'].append(item)
keys = list(var_indexes)
report['feature_alignment'] = [
    {'a': keys[0], 'b': k, 'same_order': var_indexes[keys[0]] == var_indexes[k],
     'same_set': set(var_indexes[keys[0]]) == set(var_indexes[k])}
    for k in keys[1:]] if keys else []
print(json.dumps(report, ensure_ascii=True))
'''

ssh = create_ssh_client()
try:
    stdin, stdout, stderr = ssh.exec_command('/root/miniconda3/envs/sc_fate_env/bin/python -u -')
    stdin.write(REMOTE)
    stdin.flush()
    stdin.channel.shutdown_write()
    raw = stdout.read().decode('utf-8')
    errors = stderr.read().decode('utf-8', errors='replace')
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f'Remote exit {status}: {errors}')
    report = json.loads(raw)
    output = ROOT / 'research' / 'server_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
finally:
    ssh.close()
