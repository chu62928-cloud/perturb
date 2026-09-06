from __future__ import annotations

"""Read real D1 CSR pointer/data slices without SciPy projection."""

import argparse
import h5py
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--block-rows", type=int, default=512)
    p.add_argument("--blocks", type=int, default=1000)
    args = p.parse_args()
    with h5py.File(args.path, "r") as f:
        shape = tuple(int(x) for x in f["X"].attrs["shape"])
        ip, ix, val = f["X/indptr"], f["X/indices"], f["X/data"]
        for n, start in enumerate(range(0, shape[0], args.block_rows), 1):
            if n > args.blocks:
                break
            stop = min(shape[0], start + args.block_rows)
            pointers = np.asarray(ip[start:stop + 1], dtype=np.int64)
            raw_start, raw_stop = int(pointers[0]), int(pointers[-1])
            indices = np.asarray(ix[raw_start:raw_stop])
            values = np.asarray(val[raw_start:raw_stop])
            if len(indices) != len(values) or len(indices) != raw_stop - raw_start:
                raise RuntimeError(f"CSR length mismatch block={n} start={start} stop={stop} "
                                   f"raw={raw_start}:{raw_stop} index_len={len(indices)} value_len={len(values)}")
            if n == 1 or n % 64 == 0:
                print(f"block={n} stop={stop} raw_nnz={raw_stop-raw_start}", flush=True)
            del values, indices, pointers
    print(f"completed={min(args.blocks, (shape[0] + args.block_rows - 1) // args.block_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
