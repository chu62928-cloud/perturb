from __future__ import annotations

"""Fetch and record the two independently auditable D1_Rest indptr chunks.

This script never opens the H5AD for writing and never requests a full object.
The byte ranges are intentionally fixed to the chunks audited on 2026-09-02.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SOURCE_URL = "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/D1_Rest.assigned_guide.h5ad"
CHUNKS = (
    {"row_start": 450450, "row_end": 451450, "byte_start": 20762308416, "byte_end": 20762316423},
    {"row_start": 451451, "row_end": 452451, "byte_start": 20810454136, "byte_end": 20810462143},
)


def _fetch(url: str, start: int, end: int) -> tuple[bytes, str, int]:
    request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    try:
        with urlopen(request, timeout=120) as response:
            payload = response.read()
            content_range = response.headers.get("Content-Range", "")
            status = int(getattr(response, "status", 200))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"failed HTTP range {start}-{end}: {exc}") from exc
    expected = end - start + 1
    if len(payload) != expected:
        raise RuntimeError(f"HTTP range returned {len(payload)} bytes, expected {expected}")
    if status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
        raise RuntimeError(f"HTTP range was not independently bounded: status={status}, content-range={content_range!r}")
    return payload, content_range, status


def create(original_path: Path, output: Path, reuse_existing: bool = False) -> dict:
    if "D2" in original_path.name.upper() or original_path.name != "D1_Rest.assigned_guide.h5ad":
        raise ValueError("CSR patch creation accepts only D1_Rest.assigned_guide.h5ad")
    if not original_path.exists():
        raise FileNotFoundError(original_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing patch metadata: {output}")
    patch_root = output.parent / "patches"
    patch_root.mkdir(parents=True, exist_ok=True)
    downloaded_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    records = []
    for chunk in CHUNKS:
        name = f"d1_rest_x_indptr_{chunk['row_start']}_{chunk['row_end']}.bin"
        target = patch_root / name
        if target.exists():
            if not reuse_existing:
                raise FileExistsError(f"refusing to overwrite existing patch bytes: {target}")
            payload = target.read_bytes()
            content_range, status = "existing-sidecar-bytes", 0
        else:
            payload, content_range, status = _fetch(SOURCE_URL, chunk["byte_start"], chunk["byte_end"])
            target.write_bytes(payload)
        if len(payload) != chunk["byte_end"] - chunk["byte_start"] + 1:
            raise RuntimeError(f"existing patch byte count mismatch for {target}")
        records.append({
            **chunk,
            "row_end_inclusive": True,
            "row_count": chunk["row_end"] - chunk["row_start"] + 1,
            "byte_count": len(payload),
            "patch_file": f"patches/{name}",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "downloaded_at": downloaded_at,
            "http_status": status,
            "content_range": content_range,
        })
    payload = {
        "version": "d1_rest_csr_patch.v1",
        "dataset_id": "D1_Rest",
        "original_path": str(original_path.resolve()),
        "source_url": SOURCE_URL,
        "hdf5_dataset": "X/indptr",
        "dtype": "<i8",
        "chunk_shape": [1001],
        "chunks": records,
        "raw_input_modified": False,
        "csr_patch_applied": True,
        "d2_responses_used": False,
        "provenance": "HTTP Range reads of two pre-audited uncompressed HDF5 X/indptr chunks; no full object download",
    }
    payload["artifact_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-existing", action="store_true",
                        help="reuse already downloaded chunk bytes without another HTTP request")
    args = parser.parse_args()
    payload = create(Path(args.original_path), Path(args.output), reuse_existing=args.reuse_existing)
    print(json.dumps({"output": args.output, "chunks": payload["chunks"], "d2_responses_used": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
