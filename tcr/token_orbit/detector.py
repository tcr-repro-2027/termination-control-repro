# coding=utf-8
"""Unified, benchmark-agnostic repetition detector.

For one generated text the loop decision is::

    token_length > threshold      (it ran (close) to the generation limit)
        AND a token unit of period p repeats >= k times consecutively

The periodicity pass scans the *whole* token sequence, so a loop is detected
even when the response repeats for a long stretch and then ends with a
different, non-repeating sentence.

Compared to the previous detector interface, every sample now gets a FULL
unified record via :func:`score_response` -- the binary ``loop`` flag plus the
loop anatomy required by the downstream feature-identification
experiments (``onset`` / ``period`` / ``loopContent``) and the auxiliary
severity metrics (``hit_max_tokens`` / ``rep4`` / ``gzip_ratio``), all
computed from a single tokenisation of the text.

The detector is tokenizer-agnostic: it receives an ``encode`` callable that
maps a text string to ``(token_ids, offset_mapping)`` where
``offset_mapping[i]`` is the ``(start_char, end_char)`` span of token ``i`` in
the original string (see :func:`tcr.token_orbit.tokenization.build_hf_encode`).
"""

import dataclasses
from typing import Callable, Dict, Optional, Sequence, Tuple

from tcr.token_orbit.metrics import gzip_ratio, hit_max_tokens, rep_n
from tcr.token_orbit.periodicity import find_periodic_run

EncodeFn = Callable[[str], Tuple[Sequence[int], Sequence[Tuple[int, int]]]]

TARGETS = ("full", "answer", "reasoning")


@dataclasses.dataclass
class PeriodConfig:
    """Parameters for the periodicity-based repetition detection.

    Defaults require ``k`` = 10 consecutive repeats of a repeating unit whose
    period (length in tokens) is in ``[p_min, p_max]``.  ``p_min`` = 1 catches
    single-token loops; raise ``p_max`` to catch loops with longer repeating
    units (cost is linear in ``p_max``).
    """
    p_min: int = 1
    p_max: int = 100
    k: int = 10


@dataclasses.dataclass
class DetectorConfig:
    """All knobs for the detector."""
    threshold: int                       # token-length gate (~ max gen length)
    period: PeriodConfig = dataclasses.field(default_factory=PeriodConfig)
    before_tokens: int = 100             # tokens of context captured before the loop


def build_text(answer: str, reasoning: str, target: str) -> str:
    """Assemble the text to analyse for ONE sample, per detection ``target``.

    * ``answer``    -> the final answer only
    * ``reasoning`` -> the thinking only
    * ``full``      -> ``reasoning + "\\n" + answer`` (the whole generation;
                       equals the answer when there is no reasoning)
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    answer = answer or ""
    reasoning = reasoning or ""
    if target == "answer":
        return answer
    if target == "reasoning":
        return reasoning
    if reasoning and answer:
        return f"{reasoning}\n{answer}"
    return reasoning or answer


def _loop_details(
    text: str,
    token_ids: Sequence[int],
    offsets: Sequence[Tuple[int, int]],
    config: DetectorConfig,
) -> Optional[Dict]:
    """Loop anatomy for an already-tokenised text; ``None`` if no periodic run."""
    run = find_periodic_run(
        token_ids, config.period.p_min, config.period.p_max, config.period.k
    )
    if run is None:
        return None
    onset_char = offsets[run.onset][0]
    unit_end_char = offsets[run.onset + run.period][0]
    before_char = offsets[max(0, run.onset - config.before_tokens)][0]
    return {
        # onset == loop start token index in the GENERATED text's token id
        # sequence (the field the feature-identification experiments key on;
        # equivalent to the old interface's "firstTokenPosition").
        "onset": run.onset,
        "onsetChar": onset_char,          # char position of the onset in `text`
        "period": run.period,             # repeating-unit length, in tokens
        "numRepeats": run.num_repeats,    # consecutive copies in the run
        "loopContent": text[onset_char:unit_end_char],       # one period, as text
        "contextBefore": text[before_char:onset_char],       # <= before_tokens tokens
    }


def detect_loop(text: str, encode: EncodeFn, config: DetectorConfig) -> Optional[Dict]:
    """Return the loop-anatomy record for one text, or ``None`` if no loop.

    Applies the two-layer decision: length gate first (cheap, prunes the vast
    majority), then the consecutive-periodicity scan.
    """
    if not text:
        return None
    token_ids, offsets = encode(text)
    if len(token_ids) <= config.threshold:
        return None
    return _loop_details(text, token_ids, offsets, config)


def score_response(
    text: str, finish_reason: str, encode: EncodeFn, config: DetectorConfig
) -> Dict:
    """The unified per-sample record: loop / hit_max_tokens / rep4 / gzip_ratio.

    Tokenises ``text`` exactly once and returns::

        {"loop": 0|1, "hit_max_tokens": 0|1, "rep4": float,
         "gzip_ratio": float, "gen_len": int,
         # only when loop == 1:
         "onset", "onsetChar", "period", "numRepeats",
         "loopContent", "contextBefore"}
    """
    token_ids, offsets = encode(text) if text else ([], [])
    record: Dict = {
        "loop": 0,
        "hit_max_tokens": hit_max_tokens(finish_reason),
        "rep4": round(rep_n(token_ids, 4), 6),
        "gzip_ratio": round(gzip_ratio(text), 6),
        "gen_len": len(token_ids),
    }
    if len(token_ids) > config.threshold:
        details = _loop_details(text, token_ids, offsets, config)
        if details is not None:
            record["loop"] = 1
            record.update(details)
    return record
