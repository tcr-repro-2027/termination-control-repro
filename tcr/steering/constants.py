"""Frozen S2 constants.

S2 identifies the termination-damage direction d_stop FRESH from the frozen
S1 anchors (01b-lineage complete-block boundaries).  No feature sets, onsets
or periods from 05b/07b/08e are loaded — only their SAE weight-format and
hook conventions are referenced (Qwen-Scope official layout, resid_post,
zero-based decoder-block indexing).
"""

from __future__ import annotations

S2_VERSION = "s2-v1.0"

# --- model geometry (Qwen3-8B) ---------------------------------------------
N_LAYERS = 36
D_MODEL = 4096

# --- direction identification ----------------------------------------------
SPLIT_SEED = 20260813          # dev/holdout prompt split within each anchor type
DEV_FRACTION = 0.5
PRIMARY_ANCHOR_TYPE = "A"      # the live-decision states drive d_stop
HOLDOUT_AUC_MIN = 0.85         # Gate S2a: paired holdout separability at l*

# --- SAE decomposition (fresh identification) ------------------------------
SAE_TOPK_PROJECTION = 16       # top |cos(d_stop, W_dec_f)| features
SAE_TOPK_DIFFERENTIAL = 16     # top |mean act M1 - M0| boundary features
SAE_MIN_ACTIVE_SHARE = 0.05    # differential view: active in >=5% anchors in either model
FEATURE_CARD_SNIPPETS = 8
FEATURE_CARD_MAX_FEATURES = 32
CARD_PREFIX_SUBSET = 24        # per anchor type, prefixes scanned for max-activating spans

# --- steering (R-b random-matched controls) --------------------------------
ALPHA_GRID = (1.0, 2.0, 4.0)   # units of ||mean dev-gap|| at the frozen layer
N_RANDOM_DIRECTIONS = 8
RANDOM_SEED = 20260814
K_RESAMPLE = 16
MAX_NEW_TOKENS = 32
SAMPLE_BATCH = 8
B_ALPHA_FULL = 4.0             # B anchors: support-floor test at the largest alpha
QUALITY_ALPHA = 2.0            # pre-registered strength for the quality check
QUALITY_N_ANCHORS = 16
QUALITY_SAMPLES = 4
QUALITY_MAX_NEW_TOKENS = 768
