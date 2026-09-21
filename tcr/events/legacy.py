"""Adapter for the immutable 01 token-period detector coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class LegacyOrbit:
    exists: bool
    raw_onset_token: int | None
    raw_period_tokens: int | None
    observed_num_repeats: int | None
    first_actual_repeat_token: int | None
    orbit_confirmed_at_token_exclusive: int | None
    raw_span_end_token_exclusive: int | None
    hit_max_tokens: bool
    loop_content: str | None
    context_before: str | None
    periodicity_reverified: bool | None = None


def adapt_legacy_sample(
    sample: dict[str, Any],
    *,
    min_repeats: int,
    n_tokens: int,
    token_ids: Sequence[int] | None = None,
) -> LegacyOrbit:
    loop = bool(sample.get("loop", sample.get("loop_token_legacy", False)))
    hit_max = bool(sample.get("hit_max_tokens", False))
    if not loop:
        return LegacyOrbit(False, None, None, None, None, None, None, hit_max, None, None, None)
    onset = sample.get("onset", sample.get("token_onset"))
    period = sample.get("period", sample.get("token_period"))
    repeats = sample.get(
        "numRepeats", sample.get("num_repeats", sample.get("token_num_repeats"))
    )
    if onset is None or period is None or repeats is None:
        raise ValueError("positive legacy sample lacks onset/period/numRepeats")
    onset, period, repeats = int(onset), int(period), int(repeats)
    if onset < 0 or period <= 0 or repeats < min_repeats:
        raise ValueError(
            f"invalid legacy coordinates onset={onset} period={period} "
            f"repeats={repeats} min_repeats={min_repeats}"
        )
    confirmation = onset + min_repeats * period
    span_end = onset + repeats * period
    if onset >= n_tokens or confirmation > n_tokens or span_end > n_tokens:
        raise ValueError(
            "legacy coordinates exceed retokenized response: "
            f"onset={onset} confirm={confirmation} span_end={span_end} n={n_tokens}"
        )
    reverified: bool | None = None
    if token_ids is not None:
        if len(token_ids) != n_tokens:
            raise ValueError("token_ids length disagrees with n_tokens")
        mismatch = next(
            (index for index in range(onset + period, span_end) if token_ids[index] != token_ids[index - period]),
            None,
        )
        if mismatch is not None:
            raise ValueError(
                "legacy periodicity failed exact replay at token "
                f"{mismatch}: onset={onset} period={period}"
            )
        reverified = True
    return LegacyOrbit(
        exists=True,
        raw_onset_token=onset,
        raw_period_tokens=period,
        observed_num_repeats=repeats,
        first_actual_repeat_token=onset + period,
        orbit_confirmed_at_token_exclusive=confirmation,
        raw_span_end_token_exclusive=span_end,
        hit_max_tokens=hit_max,
        loop_content=sample.get("loopContent", sample.get("loop_content")),
        context_before=sample.get("contextBefore", sample.get("context_before")),
        periodicity_reverified=reverified,
    )


def token_boundary_char(
    offsets: Sequence[tuple[int, int]], token_boundary: int, text_length: int
) -> int:
    if token_boundary < 0 or token_boundary > len(offsets):
        raise ValueError(f"token boundary {token_boundary} outside [0,{len(offsets)}]")
    if token_boundary == len(offsets):
        return text_length
    return int(offsets[token_boundary][0])
