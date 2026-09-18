#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (20260801, 20260811, 20260821)
EXPERTS = {
    "top1500_lr20_l2_12": "top1500_stable",
    "top1600_leaf600": "top1600_stable",
}


def copy(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble frozen H3M55D15 production bundle")
    parser.add_argument(
        "--model-work", type=Path,
        default=ROOT / "complete_features_h3m55d15/work/top1500_top1600_locked_oos",
    )
    parser.add_argument(
        "--catalog", type=Path,
        default=ROOT / "complete_features_h3m55d15/work/top1500_top1600_multiseed_sweep/catalog",
    )
    parser.add_argument(
        "--risk-work", type=Path,
        default=ROOT / "complete_features_h3m55d15/work/day_risk_locked_oos",
    )
    args = parser.parse_args()

    feature_sets = json.loads((args.catalog / "feature_sets.json").read_text(encoding="utf-8"))
    required = [args.model_work / 'models' / expert / f'seed_{seed}' / 'model.txt'
                for expert in EXPERTS for seed in SEEDS]
    required += [HERE / 'risk' / name for name in ('day_risk_logit.joblib',
        'day_risk_features.txt', 'dated_history.csv', 'market_loss_logit.joblib', 'market_protocol.json')]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing package inputs:\n' + '\n'.join(missing))
    manifest: dict[str, object] = {
        "name": "h3m55d15_consensus", "target": "h3m55d15",
        "current": "(pooled Top3 minus pooled Top1) AND market raw-loss gate",
        "P1": "earlier keep50 Top3 INTERSECTION current",
        "P2": "(earlier keep60 Top3 UNION current) MINUS P1",
        "seeds": list(SEEDS), "models": {}, "files": {},
    }
    for expert, feature_set in EXPERTS.items():
        feature_path = HERE / "features" / f"{expert}.txt"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_path.write_text("\n".join(feature_sets[feature_set]) + "\n", encoding="utf-8")
        manifest["models"][expert] = []
        for seed in SEEDS:
            source = args.model_work / "models" / expert / f"seed_{seed}" / "model.txt"
            target = HERE / "models" / expert / f"seed_{seed}" / "model.txt"
            copy(source, target)
            manifest["models"][expert].append(str(target.relative_to(HERE)).replace("\\", "/"))

    copy(ROOT / 'complete_features_h3m55d15/raw_loss_filter/priority_benchmark/summary.csv',
         HERE / 'reports/priority_benchmark.csv')
    import subprocess
    import sys
    (HERE / 'environment_freeze.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    for name in ("locked_oos_consensus_summary.csv", "locked_oos_day_quality.csv"):
        copy(args.model_work / "reports" / name, HERE / "reports" / name)
    copy(
        args.risk_work / "locked_oos_day_risk_summary.csv",
        HERE / "reports" / "locked_oos_day_risk_summary.csv",
    )
    for path in sorted(HERE.rglob("*")):
        if (
            path.is_file()
            and path.name != "manifest.json"
            and "__pycache__" not in path.parts
            and (path.parent == HERE or path.relative_to(HERE).parts[0] in ('models', 'risk', 'features', 'reports'))
        ):
            manifest["files"][str(path.relative_to(HERE)).replace("\\", "/")] = sha256(path)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[models] {len(SEEDS) * len(EXPERTS)}")
    print(f"[files] {len(manifest['files'])}")
    print(f"[save] {HERE}")


if __name__ == "__main__":
    main()
