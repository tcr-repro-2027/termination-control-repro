"""Frozen S1 constants.

Sampling parameters are copied VERBATIM from the frozen protocol v1.0
(`tcr/extraction/protocol.py`, nothink profile)
and from the frozen 08e chat-prefix constants (`tcr/motif/p2b/data.py`).  Any
change requires bumping ``S1_VERSION`` and re-registering the gates.
"""

from __future__ import annotations

S1_VERSION = "s1-v1.2"

# --- deployment sampling profile (nothink, protocol v1.0) -------------------
NOTHINK_SAMPLING: dict[str, float] = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
}

# --- frozen Qwen3 nothink chat prefix (identical to 08e tcr.motif.p2b.data) -----
CHAT_USER_PREFIX = "<|im_start|>user\n"
CHAT_ASSISTANT_NOTHINK = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# --- anchor construction defaults ------------------------------------------
N_A_ANCHORS = 48          # M0 natural normal-stop boundaries
N_B_ANCHORS = 48          # M1 overrun effective-completion boundaries
N_C_PROMPTS = 24          # progress-scan prompts (subset of B donors)
C_PROGRESS_FRACTIONS = (0.5, 0.75, 1.0, 1.25)
MIN_BLOCKS_A = 6          # A donors need enough blocks for an internal opener
MIN_BLOCK_B = 4           # B boundary block must be at least this (1-based)
CLOSE_TAIL_MAX_TOKENS = 8     # close path tokens after the divergence point
OPENER_MAX_TOKENS = 12        # continue path tokens after the divergence point
CLOSE_TAIL_MAX_CHARS = 16     # raw close text length guard (']' plus whitespace)
OPENER_COVER_SUBSTRING = '"source'
OPENER_COVER_MAX_CHARS = 24   # '"source' must appear this early in the next block
SELECTION_SEED = 20260811

# --- measurement defaults ---------------------------------------------------
K_RESAMPLE = 16
MAX_NEW_TOKENS = 32
SAMPLE_BATCH = 8

# --- gate thresholds (pre-registered) ---------------------------------------
CHECKPOINT_SPEARMAN_MIN = 0.8
