#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
split="${1:?Usage: bash scripts/prepare_data.sh train|test [additional arguments]}"
shift
exec "${PYTHON:-python}" -m uemo.prepare_data --split "$split" \
  --data-root "${DATA_ROOT:-data/UEmo-Data-v1}" "$@"
