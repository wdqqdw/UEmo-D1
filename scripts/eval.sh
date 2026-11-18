#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false
data_root="${DATA_ROOT:-data/UEmo-Data-v1}"
exec "${PYTHON:-python}" -m uemo.eval \
  --model "${MODEL_DIR:-models/UEmo-D1}" \
  --metadata "${METADATA:-$data_root/test/test_metadata.json}" \
  --media-root "${MEDIA_ROOT:-$data_root/test}" \
  --output "${OUTPUT_DIR:-outputs/eval}" \
  --device "${DEVICE:-cuda:0}" "$@"
