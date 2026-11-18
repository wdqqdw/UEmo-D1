#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export FLASH_ATTENTION_DETERMINISTIC=1
data_root="${DATA_ROOT:-data/UEmo-Data-v1}"
exec "${PYTHON:-python}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-4}" --module uemo.train \
  --metadata "${METADATA:-$data_root/train/train_metadata.json}" \
  --media-root "${MEDIA_ROOT:-$data_root/train}" \
  --base-model "${BASE_MODEL:-models/Qwen2.5-Omni-3B}" \
  --output "${OUTPUT_DIR:-outputs/train}" \
  --config "${TRAIN_CONFIG:-configs/train.json}" "$@"
