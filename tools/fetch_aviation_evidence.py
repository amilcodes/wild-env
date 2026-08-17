#!/usr/bin/env python3
"""Fetch and checksum the public aviation evidence cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.request
from pathlib import Path

import certifi


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/aviation/evidence_registry_v1.json"),
    )
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        help="Fetch only this document_id; may be repeated.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.registry.read_text(encoding="utf-8"))
    selected = set(args.document)
    context = ssl.create_default_context(cafile=certifi.where())
    rows: list[dict[str, str]] = []
    for document in payload["documents"]:
        document_id = str(document["document_id"])
        cache_value = document.get("local_cache_path")
        if cache_value is None or (selected and document_id not in selected):
            continue
        destination = Path(cache_value)
        expected = document.get("sha256")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or args.force:
            request = urllib.request.Request(
                str(document["url"]),
                headers={"User-Agent": "Aeolus aviation-evidence research fetch/1.0"},
            )
            with urllib.request.urlopen(request, context=context, timeout=90) as response:
                content = response.read()
            destination.write_bytes(content)
        actual = _sha256(destination)
        if expected is not None and actual != expected:
            raise RuntimeError(f"checksum mismatch for {document_id}: expected {expected}, got {actual}")
        rows.append(
            {
                "document_id": document_id,
                "path": str(destination),
                "sha256": actual,
                "status": "verified",
            }
        )
    print(json.dumps({"verified_documents": rows}, indent=2))


if __name__ == "__main__":
    main()
