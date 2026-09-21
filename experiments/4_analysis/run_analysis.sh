#!/usr/bin/env bash
# =============================================================================
# Step 4 -- every mechanism analysis of the paper, in dependency order.
#
#   bash run_analysis.sh check     # list missing inputs; computes nothing
#   bash run_analysis.sh r0        # CPU   reuse dynamics (Sec. 4.4, Fig. 2, App. B and M)
#   bash run_analysis.sh pool      # CPU   shared natural-prefix pools, 4B and 8B (App. C)
#   bash run_analysis.sh r1        # GPU   termination on shared prefixes (Sec. 4.5, Fig. 3, App. O)
#   bash run_analysis.sh r2        # GPU   contrast direction, single pulse, full
#                                  #       continuations, 8B SAE reading (Sec. 4.6, Fig. 4, Tab. 2, App. G and P)
#   bash run_analysis.sh x1        # GPU   artificial complete-motif gain (App. F and N)
#   bash run_analysis.sh assets    # CPU   all tables and appendix CSVs (+ diagnostic plots)
#   bash run_analysis.sh figures   # CPU   the six figures of the paper (F1-F4, A1-A2)
#   bash run_analysis.sh all       # everything above, stopping at the first failure
#
# The only dependency chain is r0 -> pool -> r1 -> r2.  x1 and assets do not
# block it; `assets` can be re-run at any time and marks an experiment that has
# not run as "not measured" instead of drawing zeros.
#
# Environment (see ../../env.sh): DATA_ROOT TRAIN_ROOT EVAL_ROOT PAPER_ROOT
#   MODEL_ROOT SAE_ROOT, GPUS (default 0..7), BOOTSTRAP (default 5000, process
#   analysis), SCALES (default 4B,8B; `SCALES=4B` runs only that branch).
# Optional inputs of `assets`:
#   E2_CSV      e2_metrics.csv from step 5      -> appendix/C_e2_limits.csv
#   DOSE_TABLE  obr_dose_data_table.csv (step 1) -> Table 2 data description
# =============================================================================
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}
PY=${PY:-python}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

case "${1:-all}" in
    assets|figures) ;;
    *) $PY -m tcr.data.layout "${DATA_ROOT:-$REPO_ROOT/datasets}" || exit 1 ;;
esac

export PAPER_ROOT=${PAPER_ROOT:-$REPO_ROOT/outputs/paper}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
BOOTSTRAP=${BOOTSTRAP:-5000}
SCALES=${SCALES:-4B,8B}
IFS=',' read -r -a SCALE_LIST <<< "$SCALES"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
N_GPU=${#GPU_LIST[@]}
LOG_DIR=$PAPER_ROOT/logs
mkdir -p "$LOG_DIR"

say() { echo "== [analysis $(date '+%F %T')] $*"; }

# fan_out <job-name> -- one job per line on stdin, round-robin over the GPU list.
# Each job's output goes to its own log; the function waits for all of them and
# returns non-zero if any failed.  A shard that died leaves a smaller matrix
# that still aggregates without complaint, so a swallowed failure would be
# reported as a result rather than as a gap.
fan_out() {
    local name=$1; shift
    local -a jobs=()
    while IFS= read -r line; do [[ -n "$line" ]] && jobs+=("$line"); done
    local index=0 failures=0
    local -a pids=() names=()
    for job in "${jobs[@]}"; do
        local gpu=${GPU_LIST[$((index % N_GPU))]}
        local slot="$name-$index"
        local log="$LOG_DIR/${slot}.log"
        say "$slot on GPU $gpu -> $log"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$gpu $PY $job --device cuda:0 > "$log" 2>&1 &
        pids+=($!); names+=("$slot")
        index=$((index + 1))
        if (( index % N_GPU == 0 )); then
            for position in "${!pids[@]}"; do
                wait "${pids[$position]}" || { failures=$((failures + 1));
                    say "[!] ${names[$position]} failed; see $LOG_DIR/${names[$position]}.log"; }
            done
            pids=(); names=()
        fi
    done
    for position in "${!pids[@]}"; do
        wait "${pids[$position]}" || { failures=$((failures + 1));
            say "[!] ${names[$position]} failed; see $LOG_DIR/${names[$position]}.log"; }
    done
    say "$name: ${#jobs[@]} jobs, $failures failed"
    return $(( failures > 0 ? 1 : 0 ))
}

# run_step <log-name> <command...> -- run one step, log it, keep ITS exit code
# (`cmd | tee` would report tee's status).
run_step() {
    local name=$1; shift
    "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local status=${PIPESTATUS[0]}
    if (( status != 0 )); then
        say "[!] $name failed (exit $status); see $LOG_DIR/${name}.log"
    fi
    return "$status"
}

registry() {   # registry <python expression over tcr.paper.registry>
    $PY -c "import sys; sys.path.insert(0,'$REPO_ROOT'); from tcr.paper import registry; print($1)"
}

stage_check() {
    say "input check"
    run_step preflight $PY "$HERE/preflight.py"
}

stage_r0() {
    say "R0: nine model pairs, CPU only"
    run_step r0 $PY "$HERE/run_r0.py" --bootstrap "$BOOTSTRAP"
}

stage_pool() {
    say "prefix pools for $SCALES"
    run_step pool $PY "$HERE/build_prefix_pool.py" --scales "$SCALES"
}

stage_r1() {
    for scale in "${SCALE_LIST[@]}"; do
        say "R1 $scale"
        models=$(registry "' '.join(registry.R1_SCORED['$scale'])")
        [[ -n "$models" ]] || { say "[!] no R1 models for $scale"; return 1; }
        {
            for tag in $models; do
                for shard in $(seq 0 $((N_GPU - 1))); do
                    echo "$HERE/run_r1.py --scale $scale --models $tag --shard $shard --num-shards $N_GPU"
                done
            done
        } | fan_out "r1-$scale" || return 1
        run_step "r1-$scale-aggregate" \
            $PY "$HERE/run_r1.py" --scale "$scale" --aggregate-only || return 1
    done
}

stage_r2() {
    for scale in "${SCALE_LIST[@]}"; do
        say "R2 $scale: collect residual activations"
        models=$(registry "(lambda c: ' '.join([c['clean'], c['raw'], *c.get('transfer', ())]))(registry.R2_MODELS['$scale'])")
        [[ -n "$models" ]] || { say "[!] no R2 models for $scale"; return 1; }
        {
            for tag in $models; do
                for shard in $(seq 0 $((N_GPU - 1))); do
                    echo "$HERE/run_r2.py collect --scale $scale --models $tag --shard $shard --num-shards $N_GPU"
                done
            done
        } | fan_out "r2-collect-$scale" || return 1

        say "R2 $scale: contrast direction and layer selection (CPU)"
        run_step "r2-direction-$scale" \
            $PY "$HERE/run_r2.py" direction --scale "$scale" || return 1

        # The close-token bias is fixed once per model, on development anchors,
        # before any test anchor runs; sharding it would give the shards
        # different controls.
        say "R2 $scale: close-bias calibration"
        { for tag in $models; do
            echo "$HERE/run_r2.py bias --scale $scale --models $tag"
          done; } | fan_out "r2-bias-$scale" || return 1

        say "R2 $scale: single-pulse short readouts"
        {
            for tag in $models; do
                for shard in $(seq 0 $((N_GPU - 1))); do
                    echo "$HERE/run_r2.py short --scale $scale --models $tag --shard $shard --num-shards $N_GPU"
                done
            done
        } | fan_out "r2-short-$scale" || return 1

        if [[ "$scale" == "4B" ]]; then
            say "R2 $scale: full continuations"
            {
                for tag in $models; do
                    for shard in $(seq 0 $((N_GPU - 1))); do
                        echo "$HERE/run_r2.py long --scale $scale --models $tag --shard $shard --num-shards $N_GPU"
                    done
                done
            } | fan_out "r2-long-$scale" || return 1
        fi
        if [[ "$scale" == "8B" ]]; then
            local sae_dir=${SAE_ROOT:-$REPO_ROOT/models/SAE-Res-Qwen3-8B-Base-W64K-L0_100}
            if [[ -d "$sae_dir" ]]; then
                say "R2 $scale: SAE feature reading (CPU, after collection)"
                run_step "r2-sae-$scale" \
                    $PY "$HERE/run_r2.py" sae --scale "$scale" || return 1
            else
                # Only the sparse-feature appendix needs the SAE; every other
                # 8B result is independent of it.
                say "[skip] no SAE at $sae_dir: the 8B feature reading is left unmeasured"
            fi
        fi
        run_step "r2-report-$scale" \
            $PY "$HERE/run_r2.py" report --scale "$scale" || return 1
    done
}

stage_x1() {
    say "X1: probe pool (CPU + tokenizer)"
    # A failed pool build must stop here, or the scoring silently reuses the
    # probe file from a previous run.
    run_step x1-pool $PY "$HERE/run_x1.py" --stage pool || return 1
    models=$(registry "' '.join(registry.X1_MODELS)")
    [[ -n "$models" ]] || { say "[!] no X1 models"; return 1; }
    { for tag in $models; do
        echo "$HERE/run_x1.py --stage score --models $tag"
      done; } | fan_out "x1" || return 1
    run_step x1-aggregate $PY "$HERE/run_x1.py" --stage aggregate
}

stage_assets() {
    say "tables, figures, appendices"
    local extra=()
    [[ -n "${E2_CSV:-}" ]] && extra+=(--e2-csv "$E2_CSV")
    [[ -n "${DOSE_TABLE:-}" ]] && extra+=(--dose-table "$DOSE_TABLE")
    run_step assets $PY "$HERE/make_paper_assets.py" --bootstrap 2000 \
        ${extra[@]+"${extra[@]}"}
}

stage_figures() {
    say "paper figures F1-F4, A1-A2"
    run_step figures $PY "$HERE/make_figures.py" --results "$PAPER_ROOT" \
        --metrics "${EVAL_ROOT:-$REPO_ROOT/outputs/eval}/e1_metrics.csv" \
        --output "$PAPER_ROOT/figures_paper" --previews "$PAPER_ROOT/figures_paper"
}

case "${1:-all}" in
    check)  stage_check ;;
    r0)     stage_r0 ;;
    pool)   stage_pool ;;
    r1)     stage_r1 ;;
    r2)     stage_r2 ;;
    x1)     stage_x1 ;;
    assets) stage_assets ;;
    figures) stage_figures ;;
    all)
        # Stop at the first stage that fails.  Carrying on would aggregate a
        # partial matrix and print it as a finished result.
        for stage in check r0 pool r1 r2 x1 assets figures; do
            "stage_$stage" || {
                say "[!] stage '$stage' failed; stopping here."
                say "    fix it, then re-run: bash run_analysis.sh $stage"
                say "    finished stages are cached and will be skipped."
                exit 1
            }
        done
        ;;
    *) echo "unknown stage: $1" >&2; exit 1 ;;
esac
say "stage '${1:-all}' finished"
