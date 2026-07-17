#!/usr/bin/env python
"""Download the public CoNIC 2022 Hugging Face release with verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

REPO = "MedOtter/CoNIC2022"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "cpath-demos-conic/1.0"})
    with urlopen(request) as response:
        return response.read()


def download(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    start = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "cpath-demos-conic/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = Request(url, headers=headers)
    with urlopen(request) as response, temporary.open("ab" if start else "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise RuntimeError(f"{destination.name}: expected {expected_size} bytes, got {temporary.stat().st_size}")
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(fetch(API))
    entries = [x for x in manifest if x.get("path", "").startswith("data/") and x.get("path", "").endswith(".parquet")]
    entries.sort(key=lambda x: x["path"])
    if args.limit_files:
        entries = entries[: args.limit_files]
    (args.outdir / "dataset_manifest.json").write_text(json.dumps({"repo": REPO, "entries": entries}, indent=2))
    (args.outdir / "README.md").write_bytes(fetch(BASE + "README.md"))
    for entry in entries:
        path = args.outdir / Path(entry["path"]).name
        print(f"downloading {entry['path']} ({entry.get('size', 0) / 1e6:.1f} MB)", flush=True)
        download(BASE + entry["path"], path, entry.get("lfs", {}).get("size", entry.get("size")))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = entry.get("lfs", {}).get("oid")
        if expected and digest != expected:
            raise RuntimeError(f"checksum mismatch for {path}: {digest} != {expected}")
        print(f"verified {path.name} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
