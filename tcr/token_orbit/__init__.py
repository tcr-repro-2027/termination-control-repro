# coding=utf-8
"""tcr.token_orbit -- unified repetition (loop) detection interface.

Public API:

* :func:`score_response`  -- THE unified per-sample record:
  loop / hit_max_tokens / rep4 / gzip_ratio (+ onset / period / loopContent
  / numRepeats / contextBefore when a loop is present).
* :func:`detect_loop`     -- loop anatomy only (``None`` when no loop).
* :func:`build_text`      -- assemble the analysed text per detection target
  (``full`` / ``answer`` / ``reasoning``).
* :func:`find_periodic_run` -- the raw periodicity kernel on token ids.
* :func:`build_hf_encode` -- default HF fast-tokenizer ``encode`` factory.
"""

from tcr.token_orbit.detector import (
    DetectorConfig,
    EncodeFn,
    PeriodConfig,
    TARGETS,
    build_text,
    detect_loop,
    score_response,
)
from tcr.token_orbit.metrics import gzip_ratio, hit_max_tokens, rep_n
from tcr.token_orbit.periodicity import PeriodicRun, find_periodic_run
from tcr.token_orbit.tokenization import build_hf_encode

__all__ = [
    "DetectorConfig",
    "EncodeFn",
    "PeriodConfig",
    "PeriodicRun",
    "TARGETS",
    "build_hf_encode",
    "build_text",
    "detect_loop",
    "find_periodic_run",
    "gzip_ratio",
    "hit_max_tokens",
    "rep_n",
    "score_response",
]
