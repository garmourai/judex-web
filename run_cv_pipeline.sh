#!/usr/bin/env bash
# Run the CV triplet pipeline using conda env `myenv` (torch, cv_code deps).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec conda run --no-capture-output -n myenv python run_cv_pipeline.py "$@"
