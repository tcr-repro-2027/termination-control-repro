#!/usr/bin/env bash
# =============================================================================
# One full-parameter SFT run (ms-swift 4.4.2).  Every model in the paper uses
# exactly this configuration; only <size>, <dataset> and <seed> change.
#
#   bash train_single.sh <run_name> <size> <dataset> <seed> [save_mode]
#     size       in {1.7B, 4B, 8B}
#     save_mode  in {final_only (default), trajectory}
#
#   SMOKE=1 bash train_single.sh ...   # 64 samples, written to smoke-<run_name>
#
# Steps.  Global batch = 8 GPUs x micro-batch 1 x grad-accum 8 = 64, 3 epochs.
# HF rounds up twice:  batches/epoch = ceil(N/8),  steps/epoch = ceil(batches/8).
# All paper datasets have N in [8854, 8883] -> 139 steps/epoch -> 417 steps.
#
# final_only uses the epoch save strategy with save_total_limit 1, so the last
# checkpoint is always kept even if the realised step count differs from 417
# (e.g. if `truncation_strategy delete` dropped an over-length sample).
# `trajectory` (save every 42 steps) is kept for completeness; the paper only
# evaluates final checkpoints.  save_only_model=true: ZeRO-3 optimizer shards
# are very large, so an interrupted run is restarted rather than resumed.
# =============================================================================
set -euo pipefail

usage() {
    echo "usage: bash train_single.sh <run_name> <size> <dataset> <seed> [final_only|trajectory]" >&2
}

RUN_NAME=${1:?$(usage)}
SIZE=${2:?$(usage)}
DATASET=${3:?$(usage)}
SEED=${4:?$(usage)}
SAVE_MODE=${5:-final_only}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m tcr.data.layout "$DATASET"

MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models/Qwen3}
TRAIN_ROOT=${TRAIN_ROOT:-$REPO_ROOT/outputs/checkpoints}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

case "$SIZE" in 1.7B|4B|8B) ;; *) echo "[error] size must be 1.7B / 4B / 8B" >&2; exit 1 ;; esac
case "$SAVE_MODE" in trajectory|final_only) ;; *) echo "[error] save_mode must be trajectory / final_only" >&2; exit 1 ;; esac

# DeepSpeed by size: 8B offloads only the optimizer to CPU; 4B and 1.7B use the
# built-in ZeRO-3 config.  Every run of a given size therefore shares one config.
case "$SIZE" in
    8B) DS_CONFIG=${DS_CONFIG:-$SCRIPT_DIR/ds_zero3_offload_optimizer.json} ;;
    *)  DS_CONFIG=${DS_CONFIG:-zero3} ;;
esac
if [[ "$DS_CONFIG" == *.json && ! -f "$DS_CONFIG" ]]; then
    echo "[error] DeepSpeed config not found: $DS_CONFIG" >&2; exit 1
fi

MODEL_PATH=${MODEL_ROOT}/Qwen3-${SIZE}
[[ -d "$MODEL_PATH" ]] || { echo "[error] model not found: $MODEL_PATH" >&2; exit 1; }
[[ -f "$DATASET"    ]] || { echo "[error] dataset not found: $DATASET" >&2; exit 1; }

if [[ "${SMOKE:-0}" == "1" ]]; then
    RUN_NAME=smoke-${RUN_NAME}
    DATASET_ARG="${DATASET}#64"
    echo "[SMOKE] 64 samples; the result is not an experiment"
else
    DATASET_ARG=$DATASET
fi
OUTPUT_DIR=${TRAIN_ROOT}/${RUN_NAME}

if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]]; then
    echo "[error] output directory exists and is not empty, refusing to overwrite: $OUTPUT_DIR" >&2
    echo "        to retrain: rm -rf $OUTPUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUTPUT_DIR"

if [[ "$SAVE_MODE" == "trajectory" ]]; then
    SAVE_ARGS=(--save_strategy steps --save_steps 42)
else
    SAVE_ARGS=(--save_strategy epoch --save_total_limit 1)
fi

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROWS=$(wc -l < "$DATASET")
EXPECTED_STEPS=$(python -c "import math,sys; n=int(sys.argv[1]); print(math.ceil(math.ceil(n/8)/8)*3)" "$ROWS")
SCRIPT_FP=$(sha256sum "$0" | cut -d' ' -f1)
DATA_FP=$(sha256sum "$DATASET" | cut -d' ' -f1)
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
if [[ "$NPROC" -ne 8 ]]; then
    echo "[WARN] $NPROC GPUs: the paper's global batch of 64 assumes 8 GPUs x 1 x 8;" >&2
    echo "       scale --gradient_accumulation_steps so that GPUs x accumulation = 64." >&2
fi

{
    echo "run_name:         $RUN_NAME"
    echo "date_start:       $(date '+%F %T')"
    echo "model:            Qwen3-$SIZE"
    echo "dataset:          ${DATASET_ARG##*/}"
    echo "dataset_rows:     $ROWS"
    echo "expected_steps:   $EXPECTED_STEPS"
    echo "dataset_sha256:   $DATA_FP"
    echo "script_sha256:    $SCRIPT_FP"
    echo "deepspeed:        ${DS_CONFIG##*/}"
    echo "seed:             $SEED"
    echo "save_mode:        $SAVE_MODE (${SAVE_ARGS[*]})"
    echo "---- nvidia-smi ----"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
    echo "---- key packages ----"
    pip list 2>/dev/null | grep -Ei "torch|deepspeed|flash|transformers|swift|accelerate|peft|trl|datasets|tensorboard" || true
} > "$OUTPUT_DIR/env_snapshot.txt"

echo "== [$RUN_NAME] size=$SIZE seed=$SEED save=$SAVE_MODE deepspeed=${DS_CONFIG##*/}"
echo "== $ROWS rows, sha256 $DATA_FP"
echo "== expected optimizer steps: $EXPECTED_STEPS (check 'Total optimization steps' in the log)"
echo "== output: $OUTPUT_DIR"

NPROC_PER_NODE=$NPROC \
CUDA_VISIBLE_DEVICES=$GPUS \
swift sft \
    --model "$MODEL_PATH" \
    --model_type qwen3 \
    --template qwen3 \
    --tuner_type full \
    --dataset "$DATASET_ARG" \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --max_grad_norm 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 32768 \
    --truncation_strategy delete \
    --packing false \
    --deepspeed "$DS_CONFIG" \
    --gradient_checkpointing true \
    --loss_scale default \
    --seed "$SEED" \
    --data_seed "$SEED" \
    --logging_steps 1 \
    "${SAVE_ARGS[@]}" \
    --save_only_model true \
    --output_dir "$OUTPUT_DIR" \
    --add_version false \
    --check_model false \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --report_to tensorboard \
    2>&1 | tee "$OUTPUT_DIR/train.log"

touch "$OUTPUT_DIR/TRAIN_SUCCESS"
echo "== [$RUN_NAME] finished $(date '+%F %T'); checkpoint under $OUTPUT_DIR"
