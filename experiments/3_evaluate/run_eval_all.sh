#!/usr/bin/env bash
# =============================================================================
# Step 3 -- natural generation and scoring of every model, on one frozen protocol.
#
#   bash run_eval_all.sh --check     # preflight only (strongly recommended first)
#   bash run_eval_all.sh --list      # print the task queue
#   nohup bash run_eval_all.sh > eval_$(date +%F_%H%M).log 2>&1 &
#
#   ONLY=qwen3-4b-notrain bash run_eval_all.sh      # only these tags
#   LIMIT=8 ONLY=qwen3-4b-notrain EVAL_ROOT=outputs/eval_smoke bash run_eval_all.sh
#                                                   # ten-minute smoke test
#
# What runs
#   * the three untuned references (Qwen3-1.7B/4B/8B) and the final checkpoint of
#     every run in ../2_train/run_train_all.sh that has a TRAIN_SUCCESS marker;
#   * one vLLM process per GPU (TP=1).  As soon as a model finishes generating,
#     its scoring is handed to a CPU pool and the GPU loads the next checkpoint;
#   * every task appends one row to $EVAL_ROOT/e1_metrics.csv.
#
# Protocol (tcr/evaluation/protocol.py IS the protocol)
#   non-thinking mode, T=0.7, top_p=0.8, top_k=20, presence penalty 1.5,
#   repetition penalty 1.0, 8 samples per document with seeds 0..7, generation
#   to EOS or the 32,768-token total context, 1,106 evaluation documents.
#
# Resume
#   All state is on disk.  After Ctrl-C / a crash, launch the same command again:
#   finished tasks are skipped, half-written response files are continued by key,
#   generated-but-unscored tasks go straight to the CPU pool.  Reuse is identity
#   checked (protocol, eval-set sha256, prompt sha256, model path, LIMIT).
# =============================================================================
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
cd "$REPO_ROOT"

MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models/Qwen3}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/datasets}
TRAIN_ROOT=${TRAIN_ROOT:-$REPO_ROOT/outputs/checkpoints}
EVAL_ROOT=${EVAL_ROOT:-$REPO_ROOT/outputs/eval}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$REPO_ROOT/experiments/2_train/run_train_all.sh}
PROMPT_MODULE=$REPO_ROOT/tcr/prompt_template.py

# Every model is evaluated on this one file (the cleaned evaluation split).
EVAL_DATA=${EVAL_DATA:-$DATA_ROOT/cleanv2/eval_supportclean_keep8.jsonl}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
ANALYSIS_WORKERS=${ANALYSIS_WORKERS:-4}   # CPU scoring workers; 6-8 if cores allow
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
BATCH_RECORDS=${BATCH_RECORDS:-64}        # documents per generate() call (x8 sequences)
BOOTSTRAP=${BOOTSTRAP:-2000}
AUDIT_SAMPLES=${AUDIT_SAMPLES:-40}
LAUNCH_STAGGER=${LAUNCH_STAGGER:-20}      # seconds between two vLLM launches
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
PRIORITY_MAX=${PRIORITY_MAX:-99}
NO_REFERENCE=${NO_REFERENCE:-0}           # 1 = skip the three untuned models
FOLLOW=${FOLLOW:-0}                       # 1 = keep polling for newly finished runs
ONLY=${ONLY:-}
LIMIT=${LIMIT:-}                          # smoke test: first N evaluation documents
CHECK_TOKENIZERS=${CHECK_TOKENIZERS:-1}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -m tcr.data.layout "$EVAL_DATA" || exit 1

extra=(--finals_only)
[[ "$NO_REFERENCE" == "1" ]] && extra+=(--no_reference)
[[ "$FOLLOW"       == "1" ]] && extra+=(--follow)
[[ -n "$ONLY"  ]] && extra+=(--only "$ONLY")
[[ -n "$LIMIT" ]] && extra+=(--limit "$LIMIT")

echo "==== evaluation ===="
echo "  base models:   $MODEL_ROOT"
echo "  checkpoints:   $TRAIN_ROOT"
echo "  eval set:      $EVAL_DATA"
echo "  results:       $EVAL_ROOT"
echo "  GPUs:          $GPUS   scoring workers: $ANALYSIS_WORKERS"
echo ""

mkdir -p "$EVAL_ROOT"

preflight_args=(
    --eval_data "$EVAL_DATA"
    --result_root "$EVAL_ROOT"
    --train_script "$TRAIN_SCRIPT"
    --train_output_root "$TRAIN_ROOT"
    --model_root "$MODEL_ROOT"
    --gpus "$GPUS"
    --priority_max "$PRIORITY_MAX"
    --training_prompt_module "$PROMPT_MODULE"
    --finals_only
)
[[ "$NO_REFERENCE"     == "1" ]] && preflight_args+=(--no_reference)
[[ "$CHECK_TOKENIZERS" == "0" ]] && preflight_args+=(--no_check_tokenizers)
[[ -n "$ONLY"  ]] && preflight_args+=(--only "$ONLY")
[[ -n "$LIMIT" ]] && preflight_args+=(--limit "$LIMIT")

if [[ "${1:-}" == "--check" ]]; then
    python "$SCRIPT_DIR/e1_preflight.py" "${preflight_args[@]}"
    exit $?
fi

if [[ "${1:-}" == "--list" ]]; then
    exec python -m tcr.evaluation.orchestrator --list \
        --eval_data "$EVAL_DATA" \
        --result_root "$EVAL_ROOT" \
        --train_script "$TRAIN_SCRIPT" \
        --train_output_root "$TRAIN_ROOT" \
        --model_root "$MODEL_ROOT" \
        --priority_max "$PRIORITY_MAX" \
        "${extra[@]}"
fi

python "$SCRIPT_DIR/e1_preflight.py" "${preflight_args[@]}" \
    || { echo "==== preflight FAILED: fix the [FAIL] lines above ===="; exit 1; }

echo ""
echo "==== start $(date '+%F %T') ===="
python -m tcr.evaluation.orchestrator \
    --eval_data "$EVAL_DATA" \
    --result_root "$EVAL_ROOT" \
    --train_script "$TRAIN_SCRIPT" \
    --train_output_root "$TRAIN_ROOT" \
    --model_root "$MODEL_ROOT" \
    --gpus "$GPUS" \
    --analysis_workers "$ANALYSIS_WORKERS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --batch_records "$BATCH_RECORDS" \
    --bootstrap "$BOOTSTRAP" \
    --audit_samples "$AUDIT_SAMPLES" \
    --launch_stagger "$LAUNCH_STAGGER" \
    --max_attempts "$MAX_ATTEMPTS" \
    --priority_max "$PRIORITY_MAX" \
    "${extra[@]}"
status=$?

echo ""
echo "==== collect $(date '+%F %T') ===="
# Rebuild the CSV from the per-task summaries (ordered, all columns, de-duplicated)
# and verify that every row used the same eval file, prompt, protocol and denominator.
python "$SCRIPT_DIR/e1_collect.py" --result_root "$EVAL_ROOT"

echo ""
echo "==== done $(date '+%F %T') (orchestrator exit=$status) ===="
echo "  metrics: $EVAL_ROOT/e1_metrics.csv"
echo "  ledger:  $EVAL_ROOT/e1_eval_ledger.csv"
exit $status
