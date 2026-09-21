#!/usr/bin/env bash
# Source this file once per shell:   source env.sh
#
# Every script reads its paths from these variables (and falls back to the same
# repo-relative defaults when they are unset), so moving data, checkpoints or
# outputs to another disk means editing this file only.

_this=${BASH_SOURCE[0]:-$0}
export REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$_this")" && pwd)}

# ---- inputs -----------------------------------------------------------------
# datasets/ contains format previews; point DATA_ROOT at the full download.
export DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/datasets}
# models/Qwen3/    must contain Qwen3-1.7B, Qwen3-4B, Qwen3-8B (Hugging Face snapshots)
export MODEL_ROOT=${MODEL_ROOT:-$REPO_ROOT/models/Qwen3}
# released sparse autoencoder used for the 8B feature reading (appendix)
export SAE_ROOT=${SAE_ROOT:-$REPO_ROOT/models/SAE-Res-Qwen3-8B-Base-W64K-L0_100}

# ---- outputs ----------------------------------------------------------------
export TRAIN_ROOT=${TRAIN_ROOT:-$REPO_ROOT/outputs/checkpoints}   # step 2
export EVAL_ROOT=${EVAL_ROOT:-$REPO_ROOT/outputs/eval}            # step 3
export PAPER_ROOT=${PAPER_ROOT:-$REPO_ROOT/outputs/paper}         # step 4
export PROBE_ROOT=${PROBE_ROOT:-$REPO_ROOT/outputs/support_probe} # step 5

# ---- hardware ---------------------------------------------------------------
export GPUS=${GPUS:-0,1,2,3,4,5,6,7}

# ---- runtime ----------------------------------------------------------------
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# All models are read from local directories.  Unset these two if you want
# transformers / vLLM to be able to reach the Hugging Face Hub.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

unset _this
