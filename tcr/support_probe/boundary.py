# coding=utf-8
"""Where the decision is, and what the sampler does to it.

Finding the decision point
--------------------------
A block boundary is a place in TEXT, not a place in tokens, and Qwen3's BPE does
not respect it.  Measured on `datasets/Qwen3-4B`, not assumed:

    '}]' -> [25439]   single token
    '},' -> [2137]    single token
    '[]' -> [1294]    single token

and inside a real serialised relation list the merge reaches back one character
further still, so the same boundary tokenises three different ways:

    context alone      ... '"}'          (len 118)
    context + ']'      ... '"}'   ']'    (len 119)
    context + ', {'    ... '"},'  ' {'   (len 119)

Cutting after `}` would leave the model on `"}`, while the sequence it trained
on carries `"},` there -- a different token, so the next-token distribution
would be read from a state training never produced.  Across all 384 built
anchors this happens **384 times out of 384**; there is no clean case.

So the boundary is not guessed.  Both futures are tokenised and compared:

    ids_close = tok(context + "]")
    ids_cont  = tok(context + ", {")
    L         = length of their common prefix

`ids_close[:L]` is then a real tokenisation of a real string, `ids_close[L]` and
`ids_cont[L]` are the two competing first tokens whatever the merge behaviour
was, and one forward pass over `ids_close[:L]` yields the distribution both are
scored under.

What that first token does and does not commit to: on the close branch it is
`"}` (leaving `]` for the following step), on the continue branch `"},`.  The
comparison is therefore between the two canonical continuations, which is what
§9.4's "close 首 token 的 rank / top-k/p 存活 / 采样概率" is defined on.  It is
not a claim that `"}` can only be followed by `]`.

:class:`DecisionPointFinder` computes the same answer without the redundant
work -- see its docstring.

Reproducing the sampler
-----------------------
§9.4 wants the margin "after presence penalty and temperature, before
top-k/top-p", plus rank and survival readouts.  Those are properties of vLLM's
pipeline, so the pipeline is reproduced in the order vLLM applies it:

    presence penalty -> temperature -> top_k -> top_p -> softmax

Getting the order wrong would not fail loudly; it would just report a different
number, so :func:`sampler_readouts` is written to be unit-testable without a
model and is tested against hand-computed cases.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from . import protocol

LOG = logging.getLogger("tcr.support_probe.boundary")


@dataclasses.dataclass(frozen=True)
class DecisionPoint:
    """The context to run, and the two tokens whose scores are compared."""
    context_ids: List[int]
    close_token: int
    continue_token: int
    position: int
    generated_start: int
    """Index where the assistant's own tokens begin; the presence penalty is
    defined over the GENERATED text, so the prompt must not count."""

    @property
    def presence_ids(self) -> List[int]:
        """Generated tokens so far -- legitimately EMPTY at a zero-answer
        anchor, where `[]` is a single token and the decision therefore falls
        on the very first generated token.  Nothing has been produced yet, so
        no token can carry a presence penalty."""
        return self.context_ids[self.generated_start:]

    @property
    def n_context_tokens(self) -> int:
        return len(self.context_ids)


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def find_decision_point(encode, context_text: str, close_continuation: str,
                        continue_continuation: str,
                        generation_prompt: Optional[str] = None) -> DecisionPoint:
    """Locate the first token at which closing and continuing diverge.

    ``encode`` maps text to token ids (no special tokens added).  When
    ``generation_prompt`` is given -- the chat-templated user turn up to and
    including the assistant header -- its tokenisation must be a prefix of the
    full context, which is what makes "generated tokens" well defined for the
    presence penalty; if it is not, the caller is told rather than silently
    getting a wrong penalty."""
    ids_close = list(encode(context_text + close_continuation))
    ids_cont = list(encode(context_text + continue_continuation))
    position = common_prefix_length(ids_close, ids_cont)
    if position >= len(ids_close) or position >= len(ids_cont):
        raise ValueError("close and continue continuations do not diverge; "
                         "one tokenises as a prefix of the other")

    generated_start = 0
    if generation_prompt is not None:
        prompt_ids = list(encode(generation_prompt))
        if ids_close[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "the chat generation prompt is not a token prefix of the full "
                "context; presence penalty over 'generated' tokens would be "
                "wrong.  Check the chat template.")
        generated_start = len(prompt_ids)

    return DecisionPoint(context_ids=ids_close[:position],
                         close_token=ids_close[position],
                         continue_token=ids_cont[position],
                         position=position,
                         generated_start=generated_start)


class DecisionPointFinder:
    """The same decision point, without re-tokenising what has not changed.

    :func:`find_decision_point` is correct but does three full tokenisations of
    a context that runs to ~10k tokens: once per future, plus once for the
    generation prompt.  Two measured properties of Qwen3's tokenizer remove
    almost all of that:

    * ``tok(prompt + prefix) == tok(prompt) + tok(prefix)`` -- verified on every
      anchor, so the assistant prefix can be tokenised ONCE per anchor and
      reused by all of its variants (they differ only in the prompt);
    * re-tokenising only the last few tokens plus a continuation reproduces the
      full re-tokenisation exactly -- verified for windows of 2, 4, 8 and 16
      tokens on both futures, so the two branches cost two tiny calls instead
      of two full passes.

    Measured on a 5193-token prefix with 8 variants: 243 ms -> 55 ms per anchor,
    93.5 s -> 21.0 s over the anchor set.

    Neither property is assumed.  The first ``verify_first`` DISTINCT PREFIXES
    are checked against :func:`find_decision_point` token for token; on any
    disagreement the finder logs, permanently switches to the reference
    implementation, and sets :attr:`fell_back` so the run manifest records that
    it happened.  A different tokenizer therefore costs speed, never correctness.

    The budget counts prefixes, not cells, deliberately: an anchor's eight
    variants share one assistant prefix, so a per-cell budget of eight would
    verify a single prefix and then trust every later one -- and it is prefix
    length and the identity of the last tokens, not the prompt, that decide
    whether the tail window is wide enough.
    """

    def __init__(self, encode, decode, *, window: int = 8,
                 verify_first: int = 32) -> None:
        self.encode = encode
        self.decode = decode
        self.window = window
        self.verify_first = verify_first
        self.verified = 0
        self.fell_back = False
        self._prefix_cache: Dict[str, List[int]] = {}
        self._verified_prefixes: Set[str] = set()

    def prefix_ids(self, prefix_text: str) -> List[int]:
        """Tokenise an assistant prefix once and keep it for its variants."""
        cached = self._prefix_cache.get(prefix_text)
        if cached is None:
            if len(self._prefix_cache) > 2:      # only the current anchor matters
                self._prefix_cache.clear()
            cached = list(self.encode(prefix_text))
            self._prefix_cache[prefix_text] = cached
        return cached

    def find(self, *, prompt_text: str, prefix_text: str,
             close_continuation: str, continue_continuation: str) -> DecisionPoint:
        context_text = prompt_text + prefix_text
        if self.fell_back:
            return find_decision_point(self.encode, context_text,
                                       close_continuation, continue_continuation,
                                       generation_prompt=prompt_text)

        prompt_ids = list(self.encode(prompt_text))
        base = prompt_ids + self.prefix_ids(prefix_text)
        window = min(self.window, len(base))
        head, tail_text = base[:-window], self.decode(base[-window:])
        ids_close = head + list(self.encode(tail_text + close_continuation))
        ids_cont = head + list(self.encode(tail_text + continue_continuation))
        position = common_prefix_length(ids_close, ids_cont)
        if position >= len(ids_close) or position >= len(ids_cont):
            raise ValueError("close and continue continuations do not diverge")
        point = DecisionPoint(context_ids=ids_close[:position],
                              close_token=ids_close[position],
                              continue_token=ids_cont[position],
                              position=position,
                              generated_start=len(prompt_ids))

        if (len(self._verified_prefixes) < self.verify_first
                and prefix_text not in self._verified_prefixes):
            reference = find_decision_point(
                self.encode, context_text, close_continuation,
                continue_continuation, generation_prompt=prompt_text)
            if (reference.context_ids != point.context_ids
                    or reference.close_token != point.close_token
                    or reference.continue_token != point.continue_token
                    or reference.generated_start != point.generated_start):
                LOG.error(
                    "the fast decision-point path disagrees with the reference "
                    "on this tokenizer (prompt|prefix join or tail window is "
                    "not stable); falling back to full re-tokenisation for the "
                    "rest of the run")
                self.fell_back = True
                return reference
            self._verified_prefixes.add(prefix_text)
            self.verified += 1
        return point

    def manifest(self) -> Dict[str, Any]:
        return {"decision_point_window": self.window,
                "decision_point_verified_prefixes": self.verified,
                "decision_point_verify_budget": self.verify_first,
                "decision_point_fell_back": self.fell_back}


# ------------------------------------------------------------------- sampler

def _apply_presence(logits: np.ndarray, presence_ids: Sequence[int],
                    penalty: float) -> np.ndarray:
    """vLLM subtracts the presence penalty from every token already generated.

    At a block boundary this is not a detail: `,` has occurred many times and
    `]` has not, so the penalty shifts the margin by a full `penalty` nat in
    the closing direction.  It is identical across an anchor's variants (they
    share the assistant prefix), so it cancels from every delta -- but it is
    exactly what decides the ABSOLUTE margin the sampler sees."""
    out = logits.astype(np.float64, copy=True)
    if penalty and presence_ids:
        out[np.unique(np.asarray(presence_ids, dtype=np.int64))] -= penalty
    return out


def _top_k_top_p_mask(logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    """The set vLLM would leave alive, computed the way vLLM computes it.

    The ORDER is the point.  vLLM sorts ascending, sets everything below the
    k-th largest logit to -inf, and only then takes the softmax -- so top-p runs
    on a distribution RENORMALISED over the top-k survivors, not on the full
    vocabulary.  Doing top-p on the full vocabulary and intersecting with top-k
    keeps strictly more tokens (the top-k mass is <= 1, so renormalising makes
    every survivor's probability larger and the cumulative threshold arrives
    sooner), which would inflate `n_kept_by_sampler` and could report a close
    token as surviving top-p when the real sampler had already dropped it.

    Two further details are copied rather than approximated: top-k is a value
    threshold, so a tie at the k-th largest logit keeps every tied token; and
    the argmax is force-kept, so top-p can never return an empty set.
    """
    n = logits.shape[0]
    order = np.argsort(logits, kind="stable")            # ascending, as vLLM
    values = logits[order]
    alive = np.ones(n, dtype=bool)
    if top_k and 0 < top_k < n:
        alive &= values >= values[n - top_k]
    if top_p is not None and 0.0 < top_p < 1.0:
        masked = np.where(alive, values, -np.inf)
        probs = np.exp(masked - masked.max())
        probs /= probs.sum()                             # over survivors only
        cumulative = np.cumsum(probs)
        drop = cumulative <= (1.0 - top_p)
        drop[-1] = False                                 # argmax always lives
        alive &= ~drop
    mask = np.zeros(n, dtype=bool)
    mask[order[alive]] = True
    return mask


def sampler_readouts(logits: np.ndarray, *, close_token: int,
                     continue_token: int, presence_ids: Sequence[int],
                     temperature: float = protocol.TEMPERATURE,
                     top_p: float = protocol.TOP_P, top_k: int = protocol.TOP_K,
                     presence_penalty: float = protocol.PRESENCE_PENALTY
                     ) -> Dict[str, Any]:
    """Every §9.4/§9.5 readout at one decision point.

    Returns the two margins (`sm_raw`, `sm_primary`), the close token's rank and
    survival through top-k/top-p, and the probabilities the sampler would
    actually draw from.  §9.5 only allows the phrase "support conditioning fell"
    when the raw logit, the calibrated result and at least one rank/sampler
    readout move together, which is why all three families are returned side by
    side rather than one summary number."""
    raw = logits.astype(np.float64, copy=False)
    sm_raw = float(raw[close_token] - raw[continue_token])

    penalised = _apply_presence(raw, presence_ids, presence_penalty)

    def state(values: np.ndarray) -> Dict[str, Any]:
        """Every sampler-facing readout under one penalty state."""
        scaled = values / max(temperature, 1e-6)
        # Rank is taken on the (penalised) logits: that is the ordering the
        # sampler sees.  Temperature is a positive monotone map and cannot
        # change it.
        close_rank = int((values > values[close_token]).sum()) + 1
        continue_rank = int((values > values[continue_token]).sum()) + 1
        mask = _top_k_top_p_mask(scaled, top_k, top_p)
        kept = scaled.copy()
        kept[~mask] = -np.inf
        probs = np.exp(kept - np.max(kept))
        probs /= probs.sum()
        full_probs = np.exp(scaled - scaled.max())
        full_probs /= full_probs.sum()
        return {
            "sm": float(scaled[close_token] - scaled[continue_token]),
            "close_rank": close_rank,
            "continue_rank": continue_rank,
            "close_in_top_k": bool(close_rank <= top_k),
            "close_survives_top_p": bool(mask[close_token]),
            "close_sampler_prob": float(probs[close_token]),
            "continue_sampler_prob": float(probs[continue_token]),
            "close_prob_pre_filter": float(full_probs[close_token]),
            "n_kept_by_sampler": int(mask.sum()),
        }

    primary = state(penalised)
    # The state the SHORT CONTINUATIONS are actually sampled in: vLLM is handed
    # the assistant prefix as prompt, and prompt tokens do not reach the
    # presence penalty (see protocol.HAZARD_PRESENCE_STATE).  Measured on the
    # frozen anchor set, that is a +1.5 nat difference on 254 of 384 anchors, so
    # the hazard has to be read against THIS family, not against `sm_primary`.
    # It costs nothing: the same logits, no second forward.
    unpenalised = state(raw)

    log_probs = raw - _logsumexp(raw)
    out: Dict[str, Any] = {
        "sm_raw": sm_raw,
        "sm_primary": primary["sm"],
        "sm_unpenalised": unpenalised["sm"],
        "close_logprob": float(log_probs[close_token]),
        "continue_logprob": float(log_probs[continue_token]),
        "n_presence_ids": len(set(presence_ids)),
        "close_presence_penalised": bool(close_token in set(presence_ids)),
        "continue_presence_penalised": bool(continue_token in set(presence_ids)),
    }
    for name, value in primary.items():
        if name != "sm":
            out[name] = value
    for name, value in unpenalised.items():
        if name != "sm":
            out[f"{name}_unpenalised"] = value
    return out


def _logsumexp(values: np.ndarray) -> float:
    peak = float(values.max())
    return peak + float(np.log(np.exp(values - peak).sum()))
