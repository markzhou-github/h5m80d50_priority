from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the frozen 15-member ensemble from retrain_robust_v5b.")
    parser.add_argument("--source", type=Path, default=ROOT / "retrain_robust_v5b/work/final_oos/models")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    manifest = []
    for family, spec in cfg["families"].items():
        feature_source = ROOT / "retrain_robust_v5b/work/final_oos/datasets" / spec["source_name"] / "numeric_features.txt"
        if not feature_source.exists():
            raise FileNotFoundError(feature_source)
        feature_target = HERE / "models" / family / "features.txt"
        feature_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(feature_source, feature_target)
        features = [x for x in feature_target.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(features) != int(spec["feature_count"]):
            raise ValueError(f"{family}: expected {spec['feature_count']} features, found {len(features)}")

        for seed in cfg["seeds"]:
            source = args.source / spec["source_name"] / f"seed{seed}" / "model.txt"
            target = HERE / "models" / family / f"seed{seed}" / "model.txt"
            if not source.exists():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not args.overwrite and sha256(target) != sha256(source):
                raise FileExistsError(f"Different model already exists: {target}; pass --overwrite")
            shutil.copy2(source, target)
            manifest.append({
                "family": family, "seed": seed,
                "path": target.relative_to(HERE).as_posix(),
                "bytes": target.stat().st_size, "sha256": sha256(target),
            })

    (HERE / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[installed] models={len(manifest)} bytes={sum(x['bytes'] for x in manifest):,}")
    print(f"[manifest] {HERE / 'model_manifest.json'}")


if __name__ == "__main__":
    main()
