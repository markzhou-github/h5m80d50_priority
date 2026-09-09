#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for name, expected in manifest["files"].items():
        path = HERE / name
        if not path.exists():
            failures.append(f"missing: {name}")
        elif sha256(path) != expected:
            failures.append(f"checksum: {name}")
    if failures:
        raise RuntimeError("Bundle verification failed:\n" + "\n".join(failures))
    print(f"Bundle verified: {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
