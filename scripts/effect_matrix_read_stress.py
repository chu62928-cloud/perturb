from __future__ import annotations

"""Bounded real-file CSR reader stress probe; reports the first bad block."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csr_matrix

from cd4perturb.data import _h5_strings


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--genes", required=True)
    p.add_argument("--block-rows", type=int, default=512)
    p.add_argument("--blocks", type=int, default=1000)
    args = p.parse_args()
    raw = json.loads(Path(args.genes).read_text(encoding="utf-8"))
    genes = raw["gene_order"] if isinstance(raw, dict) else raw
    with h5py.File(args.path, "r") as f:
        ids = _h5_strings(f["var"]["gene_ids"])
        lookup = {x: i for i, x in enumerate(ids)}
        cols = [lookup[x] for x in genes]
        shape = tuple(int(x) for x in f["X"].attrs["shape"])
        indptr_ds, indices_ds, data_ds = f["X/indptr"], f["X/indices"], f["X/data"]
        for block_number, start in enumerate(range(0, shape[0], args.block_rows), 1):
            if block_number > args.blocks:
                break
            stop = min(shape[0], start + args.block_rows)
            indptr = np.asarray(indptr_ds[start:stop + 1], dtype=np.int64)
            raw_start, raw_stop = int(indptr[0]), int(indptr[-1])
            indices = np.asarray(indices_ds[raw_start:raw_stop])
            values = np.asarray(data_ds[raw_start:raw_stop], dtype=np.float32)
            indptr -= raw_start
            matrix = csr_matrix((values, indices, indptr), shape=(stop - start, shape[1]))
            projected = matrix[:, cols].toarray().astype(np.float32, copy=False)
            if not np.isfinite(projected).all():
                raise RuntimeError(f"non-finite block {block_number}")
            del projected, matrix, values, indices, indptr
            if block_number == 1 or block_number % 64 == 0:
                print(f"block={block_number} stop={stop} raw_nnz={raw_stop-raw_start}", flush=True)
    print(f"completed={min(args.blocks, (shape[0] + args.block_rows - 1) // args.block_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
