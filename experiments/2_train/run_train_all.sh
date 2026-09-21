#!/usr/bin/env bash
# =============================================================================
# Train every model of the paper: 20 full-parameter SFT runs, one after another.
#
#   bash run_train_all.sh --check     # preflight only (models, data, deps, disk, GPUs)
#   bash run_train_all.sh --list      # print the queue
#   nohup bash run_train_all.sh > train_$(date +%F_%H%M).log 2>&1 &
#   ONLY=qwen3-4b-cleanv2-s42,qwen3-4b-keep4-s42 bash run_train_all.sh
#
# Behaviour
#   - nothing trains unless the preflight passes;
#   - a failed run is written to the ledger and the queue moves on;
#   - a run with TRAIN_SUCCESS is skipped, so re-launching the script is safe;
#   - a non-empty run directory without TRAIN_SUCCESS is marked needs_attention
#     and skipped -- remove it by hand to retrain;
#   - ledger: $TRAIN_ROOT/train_run_ledger.csv
#
# The RUNS array below is ALSO the model registry of the evaluation stage:
# tcr/evaluation/registry.py parses it (one quoted row per line, eight fields,
# closing parenthesis in column 0), so the models that get evaluated are exactly
# the models this file trains.  Do not put trailing comments on a row line.
#
# Naming used throughout the code (paper term -> identifier):
#   cleaned supervision  -> cleanv2        raw supervision      -> keep4
#   OBR 24.3/15/10/5 %   -> obr, obr_p15, obr_p10, obr_p5
#   candidate / text / both input edits -> isc_a, isc_e, isc_ae
#   benign input edit    -> benign_input   generic label noise  -> generic_noise
#   raw, candidate-only / out-of-text subtype -> keep4_a, keep4_ae
# The run name is the evaluation tag.  Two tags carry no seed suffix
# (qwen3-1.7b-cleanv2, qwen3-8b-cleanv2); their training seed is 42.
# =============================================================================
set -uo pipefail   # no -e: one failed run must not stop the queue

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models/Qwen3}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/datasets}
TRAIN_ROOT=${TRAIN_ROOT:-$REPO_ROOT/outputs/checkpoints}
LEDGER=${TRAIN_ROOT}/train_run_ledger.csv
PRIORITY_MAX=${PRIORITY_MAX:-99}
ONLY=${ONLY:-}

S=$DATA_ROOT
C=$DATA_ROOT/controlled
P=$DATA_ROOT/raw

# priority | run_name | size | dataset | seed | save_mode | terminal_mode | terminal_rho
# (the last two fields are unused by the paper and stay empty)
RUNS=(
  # ---- natural contrast: cleaned vs raw supervision, three scales, two 4B seeds
  "1|qwen3-4b-cleanv2-s42|4B|$S/cleanv2/swift_train_supportclean_keep8.jsonl|42|final_only||"
  "1|qwen3-4b-keep4-s42|4B|$P/swift_train_keep4.jsonl|42|final_only||"
  "1|qwen3-4b-cleanv2-s123|4B|$S/cleanv2/swift_train_supportclean_keep8.jsonl|123|final_only||"
  "1|qwen3-4b-keep4-s123|4B|$P/swift_train_keep4.jsonl|123|final_only||"
  "1|qwen3-1.7b-cleanv2|1.7B|$S/cleanv2/swift_train_supportclean_keep8.jsonl|42|final_only||"
  "1|qwen3-1.7b-keep4-s42|1.7B|$P/swift_train_keep4.jsonl|42|final_only||"
  "1|qwen3-8b-cleanv2|8B|$S/cleanv2/swift_train_supportclean_keep8.jsonl|42|final_only||"
  "1|qwen3-8b-keep4-s42|8B|$P/swift_train_keep4.jsonl|42|final_only||"

  # ---- out-of-candidate block replacement (OBR), nested doses 5 < 10 < 15 < 24.3 %
  "2|qwen3-4b-obr-p5-s42|4B|$C/swift_train_obr_p5.jsonl|42|final_only||"
  "2|qwen3-4b-obr-p10-s42|4B|$C/swift_train_obr_p10.jsonl|42|final_only||"
  "2|qwen3-4b-obr-p15-s42|4B|$C/swift_train_obr_p15.jsonl|42|final_only||"
  "2|qwen3-4b-obr-p15-s123|4B|$C/swift_train_obr_p15.jsonl|123|final_only||"
  "2|qwen3-4b-obr-s42|4B|$C/swift_train_obr.jsonl|42|final_only||"

  # ---- appendix controls: input edits, generic label noise, violation subtypes
  "3|qwen3-4b-isc-a-s42|4B|$C/swift_train_isc_a.jsonl|42|final_only||"
  "3|qwen3-4b-isc-e-s42|4B|$C/swift_train_isc_e.jsonl|42|final_only||"
  "3|qwen3-4b-isc-ae-s42|4B|$C/swift_train_isc_ae.jsonl|42|final_only||"
  "3|qwen3-4b-benign-input-s42|4B|$C/swift_train_benign_input.jsonl|42|final_only||"
  "3|qwen3-4b-generic-noise-s42|4B|$C/swift_train_generic_noise.jsonl|42|final_only||"
  "3|qwen3-4b-keep4a-s42|4B|$P/swift_train_keep4_a.jsonl|42|final_only||"
  "3|qwen3-4b-keep4ae-s42|4B|$P/swift_train_keep4_ae.jsonl|42|final_only||"
)

selected() {   # selected <priority> <run_name>
    [[ "$1" -le "$PRIORITY_MAX" ]] || return 1
    [[ -z "$ONLY" ]] && return 0
    case ",$ONLY," in *",$2,"*) return 0 ;; *) return 1 ;; esac
}

list_runs() {
    printf "%-3s %-30s %-6s %-5s %s\n" "P" "run_name" "size" "seed" "dataset"
    for run in "${RUNS[@]}"; do
        IFS='|' read -r pri name size data seed mode tmode trho <<< "$run"
        selected "$pri" "$name" || continue
        printf "%-3s %-30s %-6s %-5s %s\n" "$pri" "$name" "$size" "$seed" "${data#"$DATA_ROOT"/}"
    done
}

preflight() {
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m tcr.data.layout "$DATA_ROOT" || return 1
    local ok=1
    echo "==== preflight $(date '+%F %T') ===="
    command -v swift >/dev/null 2>&1 \
        || { echo "[FAIL] the 'swift' CLI is missing (pip install ms-swift==4.4.2)"; ok=0; }
    python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() >= 8" >/dev/null 2>&1 \
        || { echo "[FAIL] torch / CUDA / 8 visible GPUs"; ok=0; }
    python -c "import flash_attn" 2>/dev/null || { echo "[FAIL] flash_attn is not installed"; ok=0; }
    python -c "import deepspeed" 2>/dev/null || { echo "[FAIL] deepspeed is not installed"; ok=0; }

    local sizes=()
    for run in "${RUNS[@]}"; do
        IFS='|' read -r pri name size data seed mode tmode trho <<< "$run"
        selected "$pri" "$name" || continue
        sizes+=("$size")
    done
    for s in $(printf '%s\n' "${sizes[@]}" | sort -u); do
        [[ -d "$MODEL_ROOT/Qwen3-$s" ]] || { echo "[FAIL] missing model: $MODEL_ROOT/Qwen3-$s"; ok=0; }
    done

    # Every dataset is checked up front, with the step count HF will compute
    # (two ceilings).  All paper datasets land on 417 optimizer steps.
    echo "---- datasets and expected optimizer steps ----"
    local seen=""
    for run in "${RUNS[@]}"; do
        IFS='|' read -r pri name size data seed mode tmode trho <<< "$run"
        selected "$pri" "$name" || continue
        case "$seen" in *"|$data|"*) continue ;; esac
        seen="$seen|$data|"
        if [[ -f "$data" ]]; then
            local n steps
            n=$(wc -l < "$data")
            steps=$(python -c "import math,sys; n=int(sys.argv[1]); print(math.ceil(math.ceil(n/8)/8)*3)" "$n")
            printf "  %-52s rows=%-6s steps=%s\n" "${data##*/}" "$n" "$steps"
            [[ "$steps" -eq 417 ]] || echo "  [WARN] expected steps $steps != 417 (the paper's runs all have 417)"
        else
            echo "  [FAIL] missing dataset: $data"; ok=0
        fi
    done

    [[ -f "$SCRIPT_DIR/train_single.sh" ]] || { echo "[FAIL] train_single.sh is missing"; ok=0; }
    [[ -f "$SCRIPT_DIR/ds_zero3_offload_optimizer.json" ]] \
        || echo "[WARN] ds_zero3_offload_optimizer.json is missing (needed at 8B)"

    local avail_gb
    avail_gb=$(df -BG --output=avail "$TRAIN_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9') || true
    if [[ -n "${avail_gb:-}" && "$avail_gb" -lt 250 ]]; then
        echo "[WARN] $TRAIN_ROOT has ${avail_gb}G free; 20 final checkpoints need about 200G"
    fi
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1) || true
    if [[ -n "${used:-}" && "$used" -gt 2048 ]]; then
        echo "[WARN] a GPU already holds ${used}MiB; make sure nothing else is running"
    fi

    [[ $ok -eq 1 ]] || { echo "==== preflight FAILED: fix the [FAIL] lines above ===="; return 1; }
    echo "==== preflight passed ===="
}

if [[ "${1:-}" == "--list" ]]; then list_runs; exit 0; fi

mkdir -p "$TRAIN_ROOT"
echo "== training queue (PRIORITY_MAX=$PRIORITY_MAX${ONLY:+, ONLY=$ONLY})"
list_runs
echo ""
preflight || exit 1
[[ "${1:-}" == "--check" ]] && exit 0

[[ -f "$LEDGER" ]] || echo "run_name,priority,size,dataset,seed,save_mode,start,end,duration_min,exit_code,status,output_dir,dataset_sha256,script_sha256" > "$LEDGER"

total_t0=$SECONDS
done_n=0; fail_n=0; skip_n=0
for run in "${RUNS[@]}"; do
    IFS='|' read -r pri name size data seed mode tmode trho <<< "$run"
    selected "$pri" "$name" || continue
    out_dir=$TRAIN_ROOT/$name

    if [[ -f "$out_dir/TRAIN_SUCCESS" ]]; then
        echo "== [$name] already finished, skipping"; skip_n=$((skip_n + 1)); continue
    fi
    if [[ -d "$out_dir" && -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
        echo "== [$name] directory is non-empty but has no TRAIN_SUCCESS (failed earlier?), skipping"
        echo "==   to retrain: rm -rf $out_dir && re-run this script"
        echo "$name,$pri,$size,$data,$seed,$mode,,,,,needs_attention,$out_dir,," >> "$LEDGER"
        skip_n=$((skip_n + 1)); continue
    fi

    start=$(date '+%F %T'); t0=$SECONDS
    echo ""
    echo "############################################################"
    echo "== [$name] start $start  ($size, seed $seed)"
    echo "############################################################"
    MODEL_ROOT=$MODEL_ROOT TRAIN_ROOT=$TRAIN_ROOT \
        bash "$SCRIPT_DIR/train_single.sh" "$name" "$size" "$data" "$seed" "$mode"
    code=$?
    end=$(date '+%F %T'); dur=$(( (SECONDS - t0) / 60 ))

    data_fp=$(sha256sum "$data" 2>/dev/null | cut -d' ' -f1)
    script_fp=$(sha256sum "$SCRIPT_DIR/train_single.sh" | cut -d' ' -f1)
    if [[ $code -eq 0 ]]; then status=success; done_n=$((done_n + 1));
    else status=failed; fail_n=$((fail_n + 1)); fi
    echo "$name,$pri,$size,$data,$seed,$mode,$start,$end,$dur,$code,$status,$out_dir,$data_fp,$script_fp" >> "$LEDGER"
    echo "== [$name] end $end (${dur} min, exit=$code, $status)"

    sleep 30   # let GPU memory and worker processes fully release
done

echo ""
echo "############################################################"
echo "== all done $(date '+%F %T'), $(( (SECONDS - total_t0) / 60 )) min in total"
echo "== success $done_n / failed $fail_n / skipped $skip_n"
echo "== ledger: $LEDGER"
column -s, -t "$LEDGER" 2>/dev/null || cat "$LEDGER"
