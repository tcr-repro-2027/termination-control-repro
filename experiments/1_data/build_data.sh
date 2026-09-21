#!/usr/bin/env bash
# =============================================================================
# Step 1 -- (re)build the supervision conditions.  CPU only; needs the Qwen3-4B
# tokenizer for the token-length matching.  The released data archive already
# contains every file this script writes, so this step is OPTIONAL: run it to
# verify that the training sets are a deterministic function of the corpus.
#
#   bash build_data.sh raw        # raw supervision + the two violation-subtype arms
#   bash build_data.sh obr        # OBR 24.3 % (donor allocation + matching), validate
#   bash build_data.sh doses      # nested 15 / 10 / 5 % subsets + dose table
#   bash build_data.sh noise      # generic label noise
#   bash build_data.sh formats    # record form -> SWIFT chat form (what `swift sft` reads)
#   bash build_data.sh stats      # corpus statistics table (App. training-results)
#   bash build_data.sh all        # raw obr doses noise formats stats
#   bash build_data.sh stages     # UPSTREAM, optional: original corpus -> cleaning stages
#
# What each stage reads and writes is listed in datasets/README.md.
# The four input-edit controls (isc_a / isc_e / isc_ae / benign_input) are
# released as frozen files only: their edits were drafted with an LLM under
# validation rules, so they are not a deterministic function of the corpus.
# =============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "$HERE/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/datasets}
MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models/Qwen3}
TOKENIZER=${TOKENIZER:-$MODEL_ROOT/Qwen3-4B}
PY=${PY:-python}
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

$PY -m tcr.data.layout "$DATA_ROOT"

# The builders resolve <repo>/datasets from --repo-root; point it at DATA_ROOT's parent.
DATA_PARENT=$(cd "$DATA_ROOT/.." && pwd)
[[ "$(basename "$DATA_ROOT")" == "datasets" ]] || {
    echo "[error] DATA_ROOT must be a directory named 'datasets' (got $DATA_ROOT)" >&2; exit 1; }

C=$DATA_ROOT/controlled
CLEAN=$DATA_ROOT/cleanv2/train_supportclean_keep8.jsonl
PAIRS=$C/obr_pair_manifest.jsonl
# the three arms the paper trains on (the builder also knows a size-matched
# random-removal arm that the paper does not use)
RAW_ARMS=keep4,keep4_a,keep4_ae

say() { echo "== [data $(date '+%T')] $*"; }

stage_stages() {
    say "upstream: near-duplicate scan on the original corpus"
    $PY "$HERE/near_duplicate_scan.py" --repo-root "$DATA_PARENT" --source raw
    say "upstream: cleaning stages (base -> dedup -> filter -> clean -> cleanv2)"
    $PY "$HERE/build_stages.py" --repo-root "$DATA_PARENT" \
        --drop-near-duplicates "$DATA_ROOT/build_reports/near_duplicate_removals.json"
}

stage_raw() {
    say "raw supervision (keep4) and subtype arms (keep4_a, keep4_ae)"
    $PY "$HERE/build_p4_arms.py" --datasets-root "$DATA_ROOT" --only "$RAW_ARMS"
    $PY "$HERE/verify_p4_arms.py" --datasets-root "$DATA_ROOT"
}

stage_obr() {
    say "OBR 24.3 %: donor allocation and within-record matching"
    $PY "$HERE/rebuild_obr.py" --repo-root "$DATA_PARENT" --tokenizer "$TOKENIZER" \
        --data-out "$C" --report-out "$DATA_ROOT/build_reports"
    # rebuild_obr writes the pairing next to its report; the rest of the
    # pipeline reads it from controlled/.
    if [[ -f "$DATA_ROOT/build_reports/obr_pair_manifest.jsonl" ]]; then
        mv -f "$DATA_ROOT/build_reports/obr_pair_manifest.jsonl" "$PAIRS"
    fi
    say "OBR 24.3 %: invariants (inputs, block counts, one replacement per position)"
    $PY "$HERE/validate_obr.py" --repo-root "$DATA_PARENT"
}

stage_doses() {
    say "OBR 15 % and 10 %: prefixes of the stratified nested ordering"
    $PY "$HERE/build_obr_dose.py" --repo-root "$DATA_PARENT" --manifest "$PAIRS" \
        --out-dir "$C" --report "$DATA_ROOT/build_reports/obr_dose_report.json"
    say "OBR 5 %: same ordering, checked against the 10 % and 15 % files"
    $PY "$HERE/build_obr5.py" \
        --reference-data "$CLEAN" \
        --obr-data "$C/train_obr.jsonl" \
        --pairs "$PAIRS" \
        --out "$C/train_obr_p5.jsonl" \
        --swift-out "$C/swift_train_obr_p5.jsonl" \
        --trained-arm "p15=$C/train_obr_p15.jsonl" \
        --trained-arm "p10=$C/train_obr_p10.jsonl"
}

stage_noise() {
    say "generic label noise (same anchors as the input-edit controls)"
    $PY "$HERE/rebuild_generic_noise.py" --repo-root "$DATA_PARENT" --tokenizer "$TOKENIZER" \
        --manifest "$C/edit_anchor_manifest.jsonl" \
        --results "$DATA_ROOT/build_reports" --out-dir "$C"
}

stage_formats() {
    say "SWIFT chat format, written next to each record-form file"
    $PY "$HERE/build_training_formats.py" --datasets-root "$DATA_ROOT" --out-root "$DATA_ROOT"
    $PY "$HERE/build_p4_formats.py" --datasets-root "$DATA_ROOT" --out-root "$DATA_ROOT" --only "$RAW_ARMS"
}

stage_stats() {
    say "corpus statistics and the replacement-dose table"
    $PY "$HERE/dataset_statistics.py" --datasets-root "$DATA_ROOT" --tokenizer "$TOKENIZER" \
        --allow-missing
    $PY "$HERE/obr_dose_table.py" --pairs "$PAIRS" \
        --stats "$DATA_ROOT/dataset_statistics/dataset_statistics.csv" --out-dir "$C"
}

case "${1:-}" in
    stages)  stage_stages ;;
    raw)     stage_raw ;;
    obr)     stage_obr ;;
    doses)   stage_doses ;;
    noise)   stage_noise ;;
    formats) stage_formats ;;
    stats)   stage_stats ;;
    all)     stage_raw; stage_obr; stage_doses; stage_noise; stage_formats; stage_stats ;;
    *) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
say "stage '${1}' finished"
