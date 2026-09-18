from __future__ import annotations

import hashlib
import json
from pathlib import Path

import lightgbm as lgb


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "model_manifest.json").read_text(encoding="utf-8"))
    entries = {(x["family"], int(x["seed"])): x for x in manifest}
    policy = cfg.get("signal_policy", {})
    required_policy = {
        "production_tag", "family", "top_k", "gate_source_feature",
        "gate_feature", "gate_operator", "gate_threshold", "null_policy",
    }
    missing_policy = sorted(required_policy - set(policy))
    if missing_policy:
        raise ValueError(f"Missing production signal policy fields: {missing_policy}")
    if policy["family"] != "F220" or int(policy["top_k"]) != 1:
        raise ValueError("Production policy must use F220 Top1")
    if policy["gate_operator"] != "<=" or policy["null_policy"] != "fail_gate":
        raise ValueError("Unexpected production gate semantics")
    if not 0 < float(policy["gate_threshold"]) < 1:
        raise ValueError("Gate threshold must be a percentile in (0, 1)")
    checked = 0
    for family, spec in cfg["families"].items():
        features = [x for x in (HERE / "models" / family / "features.txt").read_text(encoding="utf-8").splitlines() if x]
        if len(features) != int(spec["feature_count"]):
            raise ValueError(f"{family}: feature count mismatch")
        for seed in cfg["seeds"]:
            item = entries[(family, int(seed))]
            path = HERE / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            if lgb.Booster(model_file=str(path)).feature_name() != features:
                raise ValueError(f"Feature mismatch: {family}/seed{seed}")
            checked += 1
    print(f"package ok: families={len(cfg['families'])} models={checked}")


if __name__ == "__main__":
    main()
