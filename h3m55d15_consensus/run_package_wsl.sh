#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
python complete_features_h3m55d15/18_evaluate_day_risk_locked_oos.py
python production/h3m55d15_consensus/build_day_risk_artifacts.py
python production/h3m55d15_consensus/package_production.py
python production/h3m55d15_consensus/verify_bundle.py
