# coding=utf-8
"""The fixed E1 evaluation protocol (e1-v1.0).

All models use one evaluation set, one prompt, one sampling profile and the
same scoring definitions. The protocol is recorded in each output summary.

1. Training and evaluation import the full template in ``tcr.prompt_template``.
2. Every model uses ``cleanv2/eval_supportclean_keep8.jsonl``.
3. Structured block events provide the primary repetition measurements.
   The token-period detector is retained as a separate ``legacy_*`` readout.

Sampling follows the ``nothink`` profile in ``tcr.extraction.protocol``.
Changes to protocol values require a new ``PROTOCOL_VERSION``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

PROTOCOL_VERSION: str = "e1-v1.0"

# ---------------------------------------------------------------- generation
MODE: str = "nothink"
"""E-Natural main mode (§8.1).  `think` is not part of E1."""

K: int = 8
SEEDS: Tuple[int, ...] = tuple(range(K))
"""K=8 samples per record, each its own sequence carrying its own seed.
Never one request with n=8 -- per-choice seeds cannot be pinned that way."""

MAX_MODEL_LEN: int = 32768
MAX_TOKENS: None = None
"""Uncapped: vLLM's ``SamplingParams(max_tokens=None)`` means "until EOS or
until the context window is exhausted".  The vLLM default is 16, so None has
to be passed explicitly."""

MIN_FREE_TOKENS: int = 512
"""A record whose prompt leaves less than this much room is not generated at
all; it is recorded as skipped so the denominator stays explicit.  The
tokenizer is shared by every Qwen3 size, so the skip set is identical for
every model and the comparison stays balanced."""

SAMPLING: Dict[str, float] = {
    # Qwen3 official nothink recommendation; identical to protocol v1.0.
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

CHAT_PREFIX_SUFFIX: str = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
"""What ``apply_chat_template(..., add_generation_prompt=True,
enable_thinking=False)`` must end with for a Qwen3 checkpoint.  Training fed
ms-swift the assistant turn WITHOUT a think block and the qwen3 template
prepended exactly this (`Template._add_non_thinking_prefix`), so a checkpoint
whose template drifted would silently evaluate a different conditioning than
it was trained under.  Checked once per model, recorded in the run manifest."""

# ------------------------------------------------------ legacy token detector
# Secondary token-orbit readout, shared with the event detector.
# Structured block events provide the primary repetition definition.
LEGACY_TARGET: str = "answer"
"""nothink has no reasoning segment, so `answer` == v1.0's `full`."""
LEGACY_THRESHOLD: int = 1000
LEGACY_PERIOD_MIN: int = 1
LEGACY_PERIOD_MAX: int = 200
LEGACY_MIN_REPEATS: int = 50
LEGACY_BEFORE_TOKENS: int = 100

# -------------------------------------------------- structured block detector
# Frozen 01b defaults (`tcr.events`).  `MAX_MOTIF_PERIOD_BLOCKS = 0` means
# exhaustive: no scientific cap on the motif period, at a cost that is
# quadratic in the block count of a single response.
CAPTURE_MIN_REPEATS: int = 3
MAX_MOTIF_PERIOD_BLOCKS: int = 0
REJECT_EXTRA_FIELDS: bool = True
EPISODE_VARIANT: str = "lineage"
EPISODE_CAPTURE_KIND: str = "primary"

# ------------------------------------------------------------------ inference
BOOTSTRAP: int = 2000
"""Prompt-clustered bootstrap for the headline rate CIs.  Responses of one
record are correlated, so records (not responses) are the resampling unit."""
BOOTSTRAP_SEED: int = 20260831


def sampling_params() -> Dict[str, float]:
    """A fresh copy of the frozen sampling profile."""
    return dict(SAMPLING)


def legacy_detector_config() -> Dict[str, Any]:
    return {
        "target": LEGACY_TARGET,
        "threshold": LEGACY_THRESHOLD,
        "period_min": LEGACY_PERIOD_MIN,
        "period_max": LEGACY_PERIOD_MAX,
        "min_repeats": LEGACY_MIN_REPEATS,
        "before_tokens": LEGACY_BEFORE_TOKENS,
    }


def structured_detector_config() -> Dict[str, Any]:
    return {
        "capture_min_repeats": CAPTURE_MIN_REPEATS,
        "max_motif_period_blocks": MAX_MOTIF_PERIOD_BLOCKS,
        "reject_extra_fields": REJECT_EXTRA_FIELDS,
        "episode_variant": EPISODE_VARIANT,
        "episode_capture_kind": EPISODE_CAPTURE_KIND,
    }


def describe() -> Dict[str, Any]:
    """The whole frozen protocol, for stamping into every artefact."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": MODE,
        "k": K,
        "seeds": list(SEEDS),
        "max_model_len": MAX_MODEL_LEN,
        "max_tokens": MAX_TOKENS,
        "min_free_tokens": MIN_FREE_TOKENS,
        "sampling": sampling_params(),
        "legacy_detector": legacy_detector_config(),
        "structured_detector": structured_detector_config(),
        "bootstrap": BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
