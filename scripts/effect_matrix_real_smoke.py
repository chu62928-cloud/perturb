from __future__ import annotations

"""Read one bounded block from a real D1 CSR file for effect-reader smoke."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cd4perturb.data import _h5_obs_field, _h5_strings, _read_csr_block_columns_h5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--block-rows", type=int, default=8192)
    args = parser.parse_args()
    order = json.loads(Path(args.genes).read_text(encoding="utf-8"))
    genes = order["gene_order"] if isinstance(order, dict) else order
    with h5py.File(args.path, "r") as handle:
        gene_ids = _h5_strings(handle["var"]["gene_ids"])
        lookup = {gene: index for index, gene in enumerate(gene_ids)}
        columns = [lookup[gene] for gene in genes]
        x = handle["X"]
        shape = tuple(int(x) for x in x.attrs["shape"])
        stop = min(shape[0], args.block_rows)
        block = _read_csr_block_columns_h5(handle, 0, stop, columns, shape[1])
        guide_categories, guide_dataset = _h5_obs_field(handle, "guide_id")
        group_categories, group_dataset = _h5_obs_field(handle, "guide_group")
        type_categories, type_dataset = _h5_obs_field(handle, "guide_type")
        guide_codes = np.asarray(guide_dataset[:stop], dtype=np.int64)
        group_codes = np.asarray(group_dataset[:stop], dtype=np.int64)
        type_codes = np.asarray(type_dataset[:stop], dtype=np.int64)
        print(json.dumps({
            "path": Path(args.path).name,
            "block_shape": list(block.shape),
            "finite": bool(np.isfinite(block).all()),
            "raw_sum": float(block.sum()),
            "guide_categories": len(guide_categories),
            "group_categories": group_categories,
            "type_categories": type_categories,
            "unique_guides_in_block": int(np.unique(guide_codes[guide_codes >= 0]).size),
            "single_group_cells": int(np.sum([group_categories[x] == "targeting single sgRNA" for x in group_codes])),
            "non_targeting_type_cells": int(np.sum(["target" not in type_categories[x].lower() or "non-target" in type_categories[x].lower() for x in type_codes])),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
