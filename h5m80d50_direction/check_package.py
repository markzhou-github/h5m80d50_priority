#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb


HERE = Path(__file__).resolve().parent


def main() -> None:
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    json.loads((HERE / "calibration.json").read_text(encoding="utf-8"))
    count = 0
    for kind, groups, seeds in [
        ("stage1", cfg["stage1_families"].keys(), cfg["stage1_seeds"]),
        ("competing", ["up", "down", "intensity", "direction"], cfg["competing_seeds"]),
    ]:
        for group in groups:
            root = HERE / "models" / kind / group
            features = [x for x in (root / "features.txt").read_text(encoding="utf-8").splitlines() if x]
            for seed in seeds:
                model = lgb.Booster(model_file=str(root / f"seed{seed}" / "model.txt"))
                if model.feature_name() != features:
                    raise ValueError(f"Schema mismatch: {kind}/{group}/seed{seed}")
                count += 1
    print(f"[OK] package={HERE.name} models={count}")


if __name__ == "__main__":
    main()
