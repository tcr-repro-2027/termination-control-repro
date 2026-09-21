#!/usr/bin/env bash
# =============================================================================
# Step 5 -- input-support probe (appendix "Input-support probe").
#
#   bash run_support_probe.sh --build    # CPU: build the 384 anchors only
#   bash run_support_probe.sh --check    # preflight only
#   bash run_support_probe.sh --list     # print the model queue
#   nohup bash run_support_probe.sh > probe_$(date +%F_%H%M).log 2>&1 &
#
# Stages
#   1. anchors    tcr.support_probe.build   eval set -> anchors.jsonl
#                 128 reference-list anchors x {candidate axis, text axis} with
#                 support removal vs. a length-matched neutral edit at doses 0.5
#                 and 1, the instruction arm, and 128 empty-answer anchors (CPU)
#   2. preflight  e2_preflight.py
#   3. score      tcr.support_probe.score   teacher-forced first-token stop margin
#                 under presence penalty 1.5 and temperature 0.7 (one GPU per model)
#      analyze    tcr.support_probe.analyze slopes, 2,000 anchor-bootstrap resamples
#   4. collect    e2_collect.py --strict    -> $PROBE_ROOT/e2_metrics.csv
#
# The paper reports the 18 models listed in PAPER_TAGS below; they are the
# default queue.  Set ONLY= to a comma-separated subset to score fewer.
# Afterwards:  E2_CSV=$PROBE_ROOT/e2_metrics.csv bash ../4_analysis/run_analysis.sh assets
# and, for the per-stratum table:  python e2_diagnose.py --result_root $PROBE_ROOT --sections gates
# =============================================================================
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
cd "$REPO_ROOT"

MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models/Qwen3}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/datasets}
TRAIN_ROOT=${TRAIN_ROOT:-$REPO_ROOT/outputs/checkpoints}
PROBE_ROOT=${PROBE_ROOT:-$REPO_ROOT/outputs/support_probe}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-$REPO_ROOT/experiments/2_train/run_train_all.sh}
PROMPT_MODULE=$REPO_ROOT/tcr/prompt_template.py

EVAL_DATA=${EVAL_DATA:-$DATA_ROOT/cleanv2/eval_supportclean_keep8.jsonl}
ANCHORS=${ANCHORS:-$PROBE_ROOT/anchors/anchors.jsonl}
ANCHOR_REPORT=${ANCHOR_REPORT:-$ANCHORS.report.json}
# All Qwen3 sizes share one tokenizer; it is needed to balance edits by token length.
BUILD_TOKENIZER=${BUILD_TOKENIZER:-$MODEL_ROOT/Qwen3-4B}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
ANALYSIS_WORKERS=${ANALYSIS_WORKERS:-4}
BATCH_SIZE=${BATCH_SIZE:-2}              # anchor contexts are long (full assistant prefix)
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
N_PER_TYPE=${N_PER_TYPE:-128}
# 0 = a donor entity must have EXACTLY the same token length; an anchor with no
# such donor is dropped rather than approximated, because a length change would
# move the decision position between the removal and the neutral edit.
MAX_LENGTH_DELTA=${MAX_LENGTH_DELTA:-0}
PRIORITY_MAX=${PRIORITY_MAX:-99}
NO_REFERENCE=${NO_REFERENCE:-0}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
LAUNCH_STAGGER=${LAUNCH_STAGGER:-20}

PAPER_TAGS="qwen3-1.7b-notrain,qwen3-4b-notrain,qwen3-8b-notrain"
PAPER_TAGS+=",qwen3-1.7b-cleanv2,qwen3-4b-cleanv2-s42,qwen3-4b-cleanv2-s123,qwen3-8b-cleanv2"
PAPER_TAGS+=",qwen3-4b-keep4-s42,qwen3-4b-keep4a-s42,qwen3-4b-keep4ae-s42"
PAPER_TAGS+=",qwen3-4b-obr-p10-s42,qwen3-4b-obr-p15-s42,qwen3-4b-obr-s42"
PAPER_TAGS+=",qwen3-4b-isc-a-s42,qwen3-4b-isc-e-s42,qwen3-4b-isc-ae-s42"
PAPER_TAGS+=",qwen3-4b-benign-input-s42,qwen3-4b-generic-noise-s42"
ONLY=${ONLY:-$PAPER_TAGS}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

python -m tcr.data.layout "$EVAL_DATA" || exit 1

mkdir -p "$PROBE_ROOT"

# The paper's probe is margin-only (no sampled continuations): --no_hazard always.
extra=(--no_hazard --only "$ONLY")
[[ "$NO_REFERENCE" == "1" ]] && extra+=(--no_reference)

echo "==== input-support probe ===="
echo "  eval set (anchor source): $EVAL_DATA"
echo "  anchors:                  $ANCHORS"
echo "  checkpoints:              $TRAIN_ROOT"
echo "  results:                  $PROBE_ROOT"
echo "  GPUs:                     $GPUS   analysis workers: $ANALYSIS_WORKERS"
echo ""

build_anchors() {
    # Existence is not completeness: the report must be there too, its hash
    # must match, it must describe THIS eval file and THIS protocol, the build
    # must have had a real tokenizer, and every anchor type must be full.
    if [[ -f "$ANCHORS" && "${REBUILD:-0}" != "1" ]] && \
       python "$SCRIPT_DIR/e2_check_anchors.py" "$ANCHORS" "$ANCHOR_REPORT" "$N_PER_TYPE" \
            --eval_data "$EVAL_DATA" --max_length_delta "$MAX_LENGTH_DELTA"; then
        echo "anchors already built (REBUILD=1 forces a rebuild): $ANCHORS"
        return 0
    fi
    echo "---- building anchors ----"
    python -m tcr.support_probe.build \
        --eval_data "$EVAL_DATA" \
        --out "$ANCHORS" \
        --tokenizer "$BUILD_TOKENIZER" \
        --n_per_type "$N_PER_TYPE" \
        --max_length_delta "$MAX_LENGTH_DELTA"
}

preflight() {
    python "$SCRIPT_DIR/e2_preflight.py" \
        --eval_data "$EVAL_DATA" --anchors "$ANCHORS" \
        --anchor_report "$ANCHOR_REPORT" --n_per_type "$N_PER_TYPE" \
        --max_length_delta "$MAX_LENGTH_DELTA" \
        --result_root "$PROBE_ROOT" --train_script "$TRAIN_SCRIPT" \
        --train_output_root "$TRAIN_ROOT" --model_root "$MODEL_ROOT" \
        --gpus "$GPUS" --priority_max "$PRIORITY_MAX" \
        --training_prompt_module "$PROMPT_MODULE" --only "$ONLY" --no_hazard \
        $([[ "$NO_REFERENCE" == "1" ]] && echo --no_reference)
}

if [[ "${1:-}" == "--build" ]]; then build_anchors; exit $?; fi
if [[ "${1:-}" == "--check" ]]; then preflight; exit $?; fi
if [[ "${1:-}" == "--list" ]]; then
    exec python -m tcr.support_probe.orchestrator --list \
        --anchors "$ANCHORS" --result_root "$PROBE_ROOT" \
        --train_script "$TRAIN_SCRIPT" --train_output_root "$TRAIN_ROOT" \
        --model_root "$MODEL_ROOT" --priority_max "$PRIORITY_MAX" "${extra[@]}"
fi

build_anchors || { echo "==== anchor construction FAILED ===="; exit 1; }

echo ""
echo "---- preflight ----"
preflight || { echo "==== preflight FAILED: fix the [FAIL] lines above ===="; exit 1; }

echo ""
echo "==== start $(date '+%F %T') ===="
python -m tcr.support_probe.orchestrator \
    --anchors "$ANCHORS" --anchor_report "$ANCHOR_REPORT" \
    --result_root "$PROBE_ROOT" \
    --train_script "$TRAIN_SCRIPT" --train_output_root "$TRAIN_ROOT" \
    --model_root "$MODEL_ROOT" --gpus "$GPUS" \
    --analysis_workers "$ANALYSIS_WORKERS" --batch_size "$BATCH_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --priority_max "$PRIORITY_MAX" --max_attempts "$MAX_ATTEMPTS" \
    --launch_stagger "$LAUNCH_STAGGER" \
    "${extra[@]}"
status=$?

echo ""
echo "==== collect $(date '+%F %T') ===="
# --strict: a comparability PROBLEM (different anchors / prompt / protocol /
# analysis version / bootstrap across models) fails the collection.
python "$SCRIPT_DIR/e2_collect.py" --result_root "$PROBE_ROOT" --strict
collect_status=$?
[[ $collect_status -ne 0 ]] && echo "==== collection reported comparability problems (see above) ===="
[[ $status -eq 0 ]] && status=$collect_status

echo ""
echo "==== done $(date '+%F %T') (exit=$status) ===="
echo "  metrics: $PROBE_ROOT/e2_metrics.csv"
echo "  report:  $PROBE_ROOT/e2_instrument_report.md"
echo "  ledger:  $PROBE_ROOT/e2_run_ledger.csv"
exit $status
