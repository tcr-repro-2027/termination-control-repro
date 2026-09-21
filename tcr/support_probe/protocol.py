# coding=utf-8
"""The FROZEN E2 protocol (e2-v1.0): support-conditioning instrumentation.

E2 measures one quantity at one place: the **StopMargin** at a block boundary
(§9.4),

    SM(C) = S(close | C) - S(continue | C)

and asks how it responds when the INPUT's support for the remaining blocks is
manipulated (§9.5).  Everything else here exists to make that difference
attributable to the manipulation and to nothing else.

Two design invariants carry most of the weight
----------------------------------------------
1. **Within one anchor, every variant shares a byte-identical assistant
   prefix.**  Only the user prompt changes.  So the close/continue tokens, the
   presence-penalty state and the decision position's meaning are constant, and
   `delta = SM(manipulated) - SM(neutral)` isolates the input edit.
   A consequence worth knowing: the presence penalty is then the same constant
   in both terms, so it cancels out of every delta and slope -- raw and
   primary readouts differ only by the temperature factor.  Both are reported
   anyway, because the *absolute* level is what the baseline strata (§9.5) and
   the rank/sampler readouts are defined on.
2. **The sampler pipeline is the E-Natural one.**  A margin measured under a
   different temperature or penalty is not the margin that governs the
   generation E1 scored, so the numbers here would not connect to LoopRate.

Sign convention (§9.5), frozen so a reversed slope is a real finding and not a
bookkeeping error:

* removing remaining support should RAISE StopMargin, so
  `delta_minus = SM(support-removed) - SM(neutral)` is expected positive;
* adding new evidence should LOWER it, so
  `delta_plus = SM(neutral) - SM(evidence-added)` is expected positive.

Any change to a value below requires bumping ``PROTOCOL_VERSION``.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

PROTOCOL_VERSION: str = "e2-v1.2"
"""Bumped from e2-v1.0: the MEASUREMENT changed.

v1.0 -> v1.1 covers the hazard classifier (first-character -> canonical
continuation text), the top-k/top-p order, the anchor selection and length
rules, and the hazard's sampling state being recorded and matched.  Raw
readouts and hazard files carry this version in their identity, so a file
produced under v1.0 can no longer satisfy a v1.1 run -- which is the point:
several of those changes alter the numbers in a v1.0 file without altering
anything else about how it looks.

v1.1 -> v1.2: the positive-control texts no longer contain the close bracket
they control (see `anchors.PC_STOP`).  That changes the prompt of three cells
per anchor, so it changes the measurement and needs a rebuild and a re-score."""

SUMMARY_SCHEMA_VERSION: str = "e2-summary-v4"
"""What the ANALYSIS produced, as distinct from what the instrument measured.

`PROTOCOL_VERSION` pins the measurement; this pins the code that turned it into
numbers.  They move independently: the anchor-clustered hazard interval, the
threaded resample count and the occurrence-matched sensitivity all changed what
a summary MEANS without changing a single readout, and a summary from before
them is not comparable with one from after -- while its measurement identity
matches perfectly.  So an existing summary whose schema is not this one is
recomputed rather than trusted.  Bump it whenever a summary's fields change
shape or meaning; analysis is CPU-only, so the recompute is nearly free.

v2 -> v3: a third margin family (`sm_unpenalised`), the hazard's
`presence_state` and `reachable_share`, and a gate 7 that compares against the
state the hazard was sampled in rather than against `sm_primary`.

v3 -> v4: floor/ceiling is judged on the PRE-FILTER probability instead of the
post-top-k/top-p one, so `n_informative` and every interval computed on it
change; `sampler_reachable_share` reports separately what the old rule was
measuring.  Gate 6 now only counts readouts whose CI excludes zero."""

MODE: str = "nothink"
"""E2 conditions exactly as E-Natural does: the nothink chat prefix, so the
margin measured here is the margin that governed the E1 generations."""

# ------------------------------------------------------------ sampler pipeline
# Identical to `tcr.evaluation.protocol.SAMPLING` -- deliberately duplicated as
# constants rather than imported, so E2 keeps working when uploaded alone and a
# drift between the two is a visible diff rather than a silent inheritance.
TEMPERATURE: float = 0.7
TOP_P: float = 0.8
TOP_K: int = 20
MIN_P: float = 0.0
PRESENCE_PENALTY: float = 1.5
REPETITION_PENALTY: float = 1.0
MAX_MODEL_LEN: int = 32768

#: Order the readouts are computed in, matching vLLM's sampler:
#: presence penalty -> temperature -> top_k -> top_p -> softmax.  §9.4 asks for
#: the margin "after presence penalty and temperature, before top-k/top-p"
#: (`primary`) and for the unmodified logit difference (`raw`).
SAMPLER_ORDER: Tuple[str, ...] = ("presence", "temperature", "top_k", "top_p")

# -------------------------------------------------------------------- anchors
N_ANCHORS_PER_TYPE: int = 128
"""§E2: 128 each of preterminal evidence, preterminal admissibility, terminal
evidence-add and zero-answer anchors."""

ANCHOR_TYPES: Tuple[str, ...] = ("evidence", "admissibility", "evidence_add",
                                 "zero_answer")
"""The four types §E2 specifies."""

BUILT_ANCHOR_TYPES: Tuple[str, ...] = ("evidence", "admissibility",
                                       "zero_answer")
"""The three this harness actually constructs and measures.

`evidence_add` (§8.3 C, terminal evidence-add: a document edited so that ONE
MORE relation becomes supported at the end) is specified but NOT built here.
Adding evidence means writing new document text that a human must audit for
exactly one new supported pair; there is no way to synthesise it from the
existing record the way removal is synthesised, so it needs an LLM-proposal
plus human-audit round that has not been run.

This is carried as data, not as a comment: it is written into every manifest,
report and summary row, so a run cannot be described as "all four E2 anchor
types" by anyone reading the outputs.  ECI-add is UNMEASURED in this run.
"""

REMAINDER_SIZES: Tuple[int, ...] = (2, 3, 4)
"""How many gold blocks are left unemitted at a preterminal anchor.

"Preterminal" is taken literally: the cut sits a few blocks before the end, so
"remove ALL remaining support" is a one- or two-endpoint edit rather than a
rewrite of the document.  Measured on the frozen eval set, a remainder of 2
leaves 335 fully-removable records (mean 1.9 endpoint edits, 3.9 mentions) --
comfortably more than the 128 needed, and the edit stays small enough for a
token-matched neutral counterpart to exist."""

LEVELS: Tuple[float, ...] = (0.0, 0.5, 1.0)
"""Normalised manipulation strength r (§9.5).  For a remainder of n blocks,
r removes support from `round(r * n)` of them, so the slope b is "predicted
StopMargin change from no manipulation to full manipulation", in nat."""

MIN_TEXT_MENTIONS: int = 1
MAX_CONTEXT_TOKENS: int = 24576
"""An anchor whose prompt + assistant prefix exceeds this is dropped rather
than truncated: truncation would silently change the very prefix the margin is
conditioned on."""

# ------------------------------------------------------------ positive control
PC_TAU: float = 0.5
"""§9.5: normalised analysis keeps only anchors with |PC| >= tau nat."""
PC_TAU_ROBUST: Tuple[float, ...] = (0.25, 1.0)

# --------------------------------------------------------------- stop hazard
HAZARD_SAMPLES: int = 16
"""§E2: 16 short continuations per anchor."""
HAZARD_MAX_TOKENS: int = 96
HAZARD_SEEDS: Tuple[int, ...] = tuple(range(HAZARD_SAMPLES))

HAZARD_PRESENCE_STATE: str = "prefill_unpenalised"
"""Which presence-penalty state the short continuations are sampled in.

This is a DEVIATION from the generation E1 scored, and it is recorded rather
than hidden because it changes the numbers.

In E1 the model generates the whole answer, so by the time it reaches a block
boundary every previously emitted block is an OUTPUT token and carries the
presence penalty.  `score` reproduces that: its `presence_ids` are the assistant
prefix.  The hazard cannot: it hands vLLM the prefix inside
`TokensPrompt(prompt_token_ids=...)`, and vLLM applies the presence penalty to
output tokens only (prompt tokens reach the repetition penalty alone).  So the
hazard's first step samples from a distribution in which the prefix is NOT
penalised.

Measured on the frozen anchor set, that is not a rounding difference: on
254 of 384 anchors the continue token `"},` occurs in the prefix and the close
token `"}` does not, so `sm_primary` carries a +1.5 nat push toward closing
(+2.14 logits after temperature) that the hazard's state does not have.  The
close rate the hazard reports is therefore a LOWER BOUND on the rate the same
model would show in E1-style generation.

Two consequences worth being precise about:

* every delta and slope is UNAFFECTED.  The prefix is byte-identical across an
  anchor's variants, so the penalty is the same constant in the manipulated and
  neutral arms and cancels out.  ACI, ECI and their CIs do not depend on this;
* the absolute level does depend on it.  So the scorer reports the margin in
  BOTH states (`sm_primary` and `sm_unpenalised`), gate 7 compares the hazard
  against the state it was actually measured in, and the floor/ceiling
  classification for the hazard uses that state's probabilities.

Fixing the measurement itself would need vLLM to treat the prefix as generated
-- reachable with a per-request logits processor, whose API differs between
vLLM V0 and V1 and cannot be verified from here.  Recording and matching the
state is version-independent and makes the comparison honest; it does not make
the hazard's absolute rate an E1-generation rate."""

# -------------------------------------------------------------------- stats
BOOTSTRAP: int = 2000
BOOTSTRAP_SEED: int = 20260901
BUILD_SEED: int = 20260901

#: Baseline StopMargin quintiles (§9.5 基线分层与共同支持).
BASELINE_QUANTILES: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


def sampler_params() -> Dict[str, float]:
    return {"temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
            "min_p": MIN_P, "presence_penalty": PRESENCE_PENALTY,
            "repetition_penalty": REPETITION_PENALTY}


def describe() -> Dict[str, Any]:
    """The whole frozen protocol, stamped into every artefact."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": MODE,
        "sampler": sampler_params(),
        "sampler_order": list(SAMPLER_ORDER),
        "max_model_len": MAX_MODEL_LEN,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "anchor_types": list(ANCHOR_TYPES),
        "built_anchor_types": list(BUILT_ANCHOR_TYPES),
        "unbuilt_anchor_types": [name for name in ANCHOR_TYPES
                                 if name not in BUILT_ANCHOR_TYPES],
        "n_anchors_per_type": N_ANCHORS_PER_TYPE,
        "remainder_sizes": list(REMAINDER_SIZES),
        "levels": list(LEVELS),
        "pc_tau": PC_TAU,
        "pc_tau_robust": list(PC_TAU_ROBUST),
        "hazard_samples": HAZARD_SAMPLES,
        "hazard_max_tokens": HAZARD_MAX_TOKENS,
        "bootstrap": BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "build_seed": BUILD_SEED,
    }
