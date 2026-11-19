# UEmo-D1

## 🧭 Overview

Official implementation of **UEmo-D1: An Efficient and Calibrated Decision Model for Unified Multimodal Emotion Recognition**. This repository provides a shared multimodal data pipeline, LoRA training, temperature calibration, and decision-mode evaluation. Inputs can contain text, images, audio, video, and preceding context; outputs are candidate-option predictions and their probabilities.

**[arXiv](https://arxiv.org/abs/2611.24481)** · **[Model](https://huggingface.co/wudq/UEmo-D1)** · **[Data](https://huggingface.co/datasets/wudq/UEmo-Data-v1)**

```text
configs/train.json        Training configuration
scripts/prepare_data.sh   Download and extract a data split
scripts/train.sh          Run single-node, multi-GPU training
scripts/eval.sh           Run inference and save predictions/metrics
uemo/                    Data, attention, training, evaluation, and calibration code
tests/                   Correctness checks
```

The model release is a complete inference checkpoint. The data release contains 977,270 training records and 19,072 test records, with media organized by source and modality. Training uses diverse instructions. Test metadata is available in aligned fixed-instruction JSON and diverse-instruction JSONL variants that share the same media and samples. See the data repository for the metadata schema.

## 🛠️ Train

### 1. Install the environment

Use Linux, Python 3.12, and NVIDIA CUDA GPUs. The default training configuration uses four GPUs, BF16, and FlashAttention 2. Full-length multimodal examples can require substantial GPU and host memory; keep the original media intact and lower `micro_batch_size` or `padded_token_budget` if necessary. These settings change batching, not sample lengths. The CUDA 12.8 PyTorch wheels require a compatible NVIDIA driver; building FlashAttention also requires a compatible CUDA toolkit.

```bash
git clone https://github.com/wdqqdw/UEmo-D1.git
cd UEmo-D1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install packaging ninja
MAX_JOBS=4 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

Keep the pinned Transformers version: the attention integration checks the implementation it was validated against. CPU correctness checks can be run with `python -m unittest discover -s tests -v`.

### 2. Prepare the training data and original base model

```bash
bash scripts/prepare_data.sh train
python -m uemo.download_model \
  --repo Qwen/Qwen2.5-Omni-3B \
  --output models/Qwen2.5-Omni-3B
```

The data script downloads the selected split, extracts its TAR shards without changing media bytes, and checks every referenced path. It restores `train/<source>/audio/`, `image/`, and `video/` beside `train_metadata.json`. Train and test together need approximately 151 GB for the downloads plus 150 GB for extracted media. Allow additional space for model weights and training checkpoints. To use an existing download:

```bash
DATA_ROOT=/path/to/UEmo-Data-v1 bash scripts/prepare_data.sh train --local-only
```

### 3. Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
DATA_ROOT=data/UEmo-Data-v1 \
BASE_MODEL=models/Qwen2.5-Omni-3B \
OUTPUT_DIR=outputs/train \
bash scripts/train.sh
```

The default recipe starts from the original base model, freezes its encoders and base weights, and trains decoder LoRA parameters with rank 16, alpha 32, and dropout 0. Input-prefix attention is bidirectional. The loss is cross entropy over the sample's valid candidate options. Training uses one epoch, global batch size 64, learning rate `1e-4`, weight decay `0.01`, and 500 warmup steps followed by cosine decay. The published metadata is shuffled; the runner applies a deterministic training permutation with seed `20261004`. It uses every selected record and normalizes the final partial batch by its actual sample count.

`configs/train.json` controls the schedule and batching. Change `NPROC_PER_NODE` for a different GPU count while keeping the global batch size fixed. For a short installation check, use a separate output directory:

```bash
OUTPUT_DIR=outputs/train-smoke bash scripts/train.sh --max-samples 64 --max-steps 1
```

Training saves `train_log.jsonl`, `run.json`, `latest_checkpoint.json`, and `checkpoint-step-*/` directories. Each checkpoint contains an `adapter/` directory plus optimizer, cursor, and RNG state. The original base remains separate. Resume with the same data, configuration, GPU count, and output directory:

```bash
bash scripts/train.sh --resume outputs/train/checkpoint-step-0001000
```

### 4. Calibrate a newly trained adapter

New adapters default to temperature 1. First run the evaluation command in the next section with `--adapter` to save raw option logits. Then fit a temperature without repeating model inference:

```bash
python -m uemo.calibrate \
  --predictions outputs/eval-trained/predictions.jsonl \
  --fraction 0.1 --seed 20261112 \
  --output outputs/calibration
```

This selects a uniform 10% subset without replacement, approximately minimizes equal-setting ECE over 10 bins, and saves `temperature.json`, recalibrated `predictions.jsonl`, and `metrics.json`. The full-set metrics include the calibration subset. Use predictions from a separate calibration split if you need held-out evaluation. The published checkpoint already contains its fixed temperature, `1.0790077648040843`, and needs no refitting.

## 📊 Eval

### 1. Prepare the test data and released model

Use the environment installed above. To evaluate without training, only the test split and the released model are needed:

```bash
bash scripts/prepare_data.sh test
python -m uemo.download_model --repo wudq/UEmo-D1 --output models/UEmo-D1
```

### 2. Fixed-instruction evaluation

Fixed evaluation uses `test/test_metadata.json`: one fixed instruction per setting.

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_DIR=models/UEmo-D1 \
DATA_ROOT=data/UEmo-Data-v1 \
bash scripts/eval.sh fixed
```

Results are saved to `outputs/eval-fixed/` by default. Resume this variant with:

```bash
bash scripts/eval.sh fixed --resume
```

### 3. Diverse-instruction evaluation

Diverse evaluation uses `test/test_metadata_diverse.jsonl`: sample-specific instructions from the frozen diverse-instruction evaluation set. It contains the same 19,072 samples and 41 settings as fixed evaluation. Sample IDs, media, context, option order, and answers are identical; only the instruction differs.

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_DIR=models/UEmo-D1 \
DATA_ROOT=data/UEmo-Data-v1 \
bash scripts/eval.sh diverse
```

Results are saved separately to `outputs/eval-diverse/` by default. Resume this variant with:

```bash
bash scripts/eval.sh diverse --resume
```

The data preparation step downloads both metadata variants. If you already downloaded the earlier test release, fetch just the diverse metadata without downloading media again:

```bash
hf download wudq/UEmo-Data-v1 test/test_metadata_diverse.jsonl \
  --repo-type dataset --revision 2d0c348eee48d44e14c0c0872b1740f5e8e7b619 \
  --local-dir data/UEmo-Data-v1
```

### 4. Shared evaluation options

Both variants use the same inference and scoring code. The runner reads JSON or JSONL, preserves each sample's option order and context, retains full audio, and samples video at 2 FPS. DFEW's ordered image frames remain images. It scores the valid option letters and applies the model's temperature exactly once. There is no API call or fixed 16-frame cap.

Omitting the variant selects `fixed`. `OUTPUT_DIR` overrides the output directory, and `METADATA` overrides the metadata path for custom data. Use separate output directories when changing variants, inputs, model weights, or temperature. For older runs saved in `outputs/eval/`, explicitly set `OUTPUT_DIR=outputs/eval` when resuming.

For a quick installation check:

```bash
OUTPUT_DIR=outputs/eval-fixed-smoke bash scripts/eval.sh fixed --max-samples 8
OUTPUT_DIR=outputs/eval-diverse-smoke bash scripts/eval.sh diverse --max-samples 8
```

To evaluate a newly trained adapter, set `MODEL_DIR` to the original base and pass the adapter directory printed in `latest_checkpoint.json`. Choose `fixed` or `diverse` explicitly:

```bash
MODEL_DIR=models/Qwen2.5-Omni-3B \
OUTPUT_DIR=outputs/eval-trained \
bash scripts/eval.sh fixed --adapter outputs/train/checkpoint-step-0015270/adapter
```

For subsequent inference with a temperature fitted for that adapter, add `--temperature-file outputs/calibration/temperature.json`.

### 5. Read the saved results

```text
outputs/
  eval-fixed/
    predictions.jsonl
    metrics.json
    run.json
    completed.json
  eval-diverse/
    predictions.jsonl
    metrics.json
    run.json
    completed.json
```

Each directory contains incremental predictions (`predictions.jsonl`), aggregated metrics (`metrics.json`), input/model fingerprints and configuration (`run.json`), and completion counts and hashes (`completed.json`).

Each prediction includes `id`, `source`, `source_id`, `subtask` when present, `modality`, `choices`, `gt_answer`, `prediction_letter`, `prediction`, `correct`, `option_logits`, `probabilities`, `temperature`, and `seconds`. Logits and probabilities align with `choices`. An error is saved to `errors.jsonl` and stops the run instead of silently dropping a sample.

Metrics include accuracy, macro/weighted F1, ECE10, and multiclass Brier score. Values use the 0–1 scale. `mean_over_settings` gives each setting equal weight; `overall_sample_weighted` pools all samples. A setting is identified by source, subtask, and target modality, so CREMA-D audio-only and audiovisual results remain separate.
