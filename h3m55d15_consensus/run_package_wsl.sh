#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
notify() {
  result=$?
  trap - EXIT
  curl -fsS --max-time 15 "https://api.day.app/Eqx2dpLcNQdceTffBdXNuL/Consensus%20exit%20${result}?sound=minuet" >/dev/null || true
  exit "$result"
}
trap notify EXIT
python production/h3m55d15_consensus/build_day_risk_artifacts.py
python production/h3m55d15_consensus/package_production.py
python production/h3m55d15_consensus/verify_bundle.py
