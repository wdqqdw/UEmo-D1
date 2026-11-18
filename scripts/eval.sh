#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false
data_root="${DATA_ROOT:-data/UEmo-Data-v1}"
variant="fixed"
case "${1:-}" in
  fixed|diverse) variant="$1"; shift ;;
  ""|--*) ;;
  *) echo "Usage: bash scripts/eval.sh [fixed|diverse] [evaluation options]" >&2; exit 2 ;;
esac
case "$variant" in
  fixed) metadata="$data_root/test/test_metadata.json" ;;
  diverse) metadata="$data_root/test/test_metadata_diverse.jsonl" ;;
esac
exec "${PYTHON:-python}" -m uemo.eval \
  --model "${MODEL_DIR:-models/UEmo-D1}" \
  --metadata "${METADATA:-$metadata}" \
  --media-root "${MEDIA_ROOT:-$data_root/test}" \
  --output "${OUTPUT_DIR:-outputs/eval-$variant}" \
  --device "${DEVICE:-cuda:0}" "$@"
