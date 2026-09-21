# coding=utf-8
"""The FROZEN evaluation protocol (v1.0).

Every knob that defines the sampling side of the protocol lives here, in code,
so a run is reproducible from the repository state alone.  ANY change to any
value below requires bumping ``PROTOCOL_VERSION`` -- the version is stamped
into every eval/detection summary and into the runs ledger.

Sampling follows the official Qwen3 recommendations:

* nothink : temperature=0.7, top_p=0.8,  top_k=20, min_p=0, presence_penalty=1.5,
            repetition_penalty=1.0
* think   : temperature=0.6, top_p=0.95, top_k=20, min_p=0, presence_penalty=1.5,
            repetition_penalty=1.0

K = 8 samples per record with per-sample FIXED seeds 0..7: each sample is its
own sequence carrying its own seed (never one request with n=8, whose per-choice
seeds could not be pinned).

max_tokens is NOT capped: generation may use the full remaining context
(``max_model_len - prompt_len``).  With the vLLM ``LLM`` class this is expressed
as ``SamplingParams(max_tokens=None)`` -- None means "generate until EOS or the
model length limit" (the vLLM default is 16, so None must be passed explicitly).
"""

from typing import Dict, Tuple

PROTOCOL_VERSION: str = "v1.0"

K: int = 8                                  # samples per record
SEEDS: Tuple[int, ...] = tuple(range(K))    # per-sample fixed seeds 0..7

MODES: Tuple[str, ...] = ("nothink", "think")

_SAMPLING: Dict[str, Dict[str, float]] = {
    "nothink": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
    "think": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    },
}


def sampling_for(mode: str) -> Dict[str, float]:
    """Return a copy of the frozen sampling params for ``mode``."""
    if mode not in _SAMPLING:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return dict(_SAMPLING[mode])
