"""Deployment policy-space math (numpy reference implementation).

The deployed nothink sampler (vLLM, protocol v1.0) applies, in order:

1. additive presence penalty (-1.5) on every token id already present in the
   OUTPUT so far (prompt tokens are never penalized);
2. repetition_penalty = 1.0 (no-op, asserted);
3. division by temperature (0.7);
4. top-k (20) then top-p (0.8) truncation (min_p = 0.0 no-op, asserted);
5. sampling.

"Policy space" for StopMargin = log-softmax AFTER step 3 and BEFORE step 4,
matching the ``apply_presence_temperature`` convention. This
module is pure numpy so the math is unit-testable without torch; the GPU
measurement uses the same transformations.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .constants import NOTHINK_SAMPLING


def assert_frozen_sampling(config: Mapping[str, float]) -> None:
    """Hard-fail unless the sampling profile equals the frozen nothink protocol."""
    for key, expected in NOTHINK_SAMPLING.items():
        actual = config.get(key)
        if actual is None or float(actual) != float(expected):
            raise ValueError(
                f"sampling parameter {key}={actual!r} differs from frozen nothink {expected!r}"
            )
    if float(config["repetition_penalty"]) != 1.0:
        raise ValueError("S1 freezes repetition_penalty=1.0")
    if float(config["min_p"]) != 0.0:
        raise ValueError("S1 freezes min_p=0.0")


def policy_logits_np(
    raw_logits: np.ndarray,
    *,
    presence_ids: Iterable[int],
    presence_penalty: float = NOTHINK_SAMPLING["presence_penalty"],
    temperature: float = NOTHINK_SAMPLING["temperature"],
) -> np.ndarray:
    """Presence-penalized, temperature-scaled logits (pre top-k/p)."""
    scores = np.asarray(raw_logits, dtype=np.float64).copy()
    ids = sorted({int(value) for value in presence_ids})
    if presence_penalty != 0.0 and ids:
        scores[np.asarray(ids, dtype=np.int64)] -= float(presence_penalty)
    if temperature != 1.0:
        scores = scores / float(temperature)
    return scores


def log_softmax_np(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    shifted = scores - np.max(scores)
    return shifted - np.log(np.sum(np.exp(shifted)))


def policy_logprobs_np(raw_logits: np.ndarray, *, presence_ids: Iterable[int]) -> np.ndarray:
    return log_softmax_np(policy_logits_np(raw_logits, presence_ids=presence_ids))


def raw_logprobs_np(raw_logits: np.ndarray) -> np.ndarray:
    return log_softmax_np(np.asarray(raw_logits, dtype=np.float64))


def topk_topp_keep_mask_np(
    policy_logits: np.ndarray,
    *,
    top_k: int = int(NOTHINK_SAMPLING["top_k"]),
    top_p: float = NOTHINK_SAMPLING["top_p"],
) -> np.ndarray:
    """Boolean survival mask after top-k then top-p (HF warper semantics).

    top-k keeps the k highest logits (ties broken by index order, mirroring
    ``torch.topk``); top-p then keeps the smallest descending-order set whose
    probability mass reaches ``top_p``, INCLUDING the crossing token.
    """
    scores = np.asarray(policy_logits, dtype=np.float64)
    keep = np.zeros(scores.shape[-1], dtype=bool)
    order = np.argsort(-scores, kind="stable")
    if top_k > 0:
        order = order[: min(top_k, order.size)]
    kept_scores = scores[order]
    probs = np.exp(kept_scores - np.max(kept_scores))
    probs = probs / probs.sum()
    cumulative_before = np.concatenate([[0.0], np.cumsum(probs)[:-1]])
    survivors = order[cumulative_before < float(top_p)]
    keep[survivors] = True
    if not keep.any():
        keep[order[0]] = True
    return keep


def logsumexp_np(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("logsumexp over an empty set")
    peak = float(np.max(array))
    return peak + float(np.log(np.sum(np.exp(array - peak))))


# Characters a continuation may legitimately emit while still finishing the
# current block (the boundary token can straddle the closing `"}`): quotes,
# closing brace, sentence-final punctuation, whitespace.
_TAIL_ALPHABET = set(' \t\r\n"}.。”’\'』」！？!?…')


def classify_continuation(text: str, *, ended_with_stop_token: bool, scan_limit: int = 24) -> str:
    """Classify one resampled continuation at a block boundary.

    The boundary sits on the longest common token prefix of the close and
    continue variants, so a sampled continuation first re-emits the block-tail
    remainder (e.g. ``"}``) before the decision character.  The FIRST decisive
    character within ``scan_limit`` wins:

    ``]``          -> stop (close the JSON list);
    ``,`` or ``{`` -> continue (start another relation block);
    none found     -> stop if the model emitted a stop token and produced only
    block-tail characters, else other.
    """
    for ch in text[:scan_limit]:
        if ch == "]":
            return "stop"
        if ch in {",", "{"}:
            return "continue"
    stripped = text.strip()
    if ended_with_stop_token and all(ch in _TAIL_ALPHABET for ch in stripped):
        return "stop"
    return "other"
