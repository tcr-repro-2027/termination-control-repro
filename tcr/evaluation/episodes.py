# coding=utf-8
"""Reuse-episode summary (§9.3): exposure vs per-episode propensity.

`01b`'s P0d asks the fork this project's whole dynamics story turns on:

    does a model capture more because it experiences MORE reuse episodes
    (exposure, i.e. it fails to terminate), or because EACH episode is more
    likely to capture (per-episode propensity)?

The paired Shapley decomposition that answers it lives in
`tcr.events.p0d_episode_hazard` and stays there -- it compares two models
and needs both in one file.  What E1 needs is the *marginal* half of it for
every single model, so `e1_metrics.csv` carries "episodes per response",
"per-episode capture hazard" and "gap stop hazard" as columns and the exposure
story can be read straight off the stage curve.

Every definition is imported, not restated: the block classes, the lineage
episode state machine and the competing-outcome assignment are P0d's frozen
ones.  Only the pooling is local, and it is deliberately pooled (events over
slots at risk) rather than the interleaved-hazard array P0d builds -- one
number per model, comparable across the whole matrix.

A row that P0d's validators reject does not kill the task: it is counted in
``n_errors`` and its message kept.  These columns are diagnostic; the primary
E1 endpoints do not depend on them.
"""

from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping

from . import protocol
from tcr.events.p0d_episode_hazard import (        # noqa: E402
    assign_outcome, classify_blocks, extract_block_sequence, segment_episodes,
)

__all__ = ["episode_summary", "response_outcomes"]

_ANALYSER_TAG = "M0"
"""P0d validates `model_tag in ("M0", "M1")` because it is a paired analysis.
E1 runs it one model at a time, so the tag is swapped on a shallow copy; the
real tag stays in the event row on disk."""


def response_outcomes(rows: Iterable[Mapping[str, Any]], *,
                      variant: str = protocol.EPISODE_VARIANT,
                      capture_kind: str = protocol.EPISODE_CAPTURE_KIND
                      ) -> tuple[List[Any], List[str]]:
    """P0d outcomes for every row, plus the messages of the ones that failed."""
    outcomes: List[Any] = []
    errors: List[str] = []
    for row in rows:
        # A response with no complete block is NOT skipped.  It has a perfectly
        # well-defined episode-axis outcome -- zero episodes, terminating in the
        # first gap slot -- and dropping it would shrink the denominator of
        # every per-response rate below, systematically flattering exactly the
        # models that emit the most unparseable output.
        try:
            sequence = extract_block_sequence({**row, "model_tag": _ANALYSER_TAG})
            kinds, previous = classify_blocks(sequence)
            episodes = segment_episodes(sequence, kinds, previous, variant=variant)
            outcomes.append(assign_outcome(sequence, kinds, episodes,
                                           capture_kind=capture_kind,
                                           episode_variant=variant))
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{row.get('sample_id')}: {type(exc).__name__}: {exc}")
    return outcomes, errors


def episode_summary(rows: Iterable[Mapping[str, Any]], *,
                    variant: str = protocol.EPISODE_VARIANT,
                    capture_kind: str = protocol.EPISODE_CAPTURE_KIND
                    ) -> Dict[str, Any]:
    """Pooled episode-axis rates for ONE model."""
    outcomes, errors = response_outcomes(rows, variant=variant,
                                         capture_kind=capture_kind)
    summary: Dict[str, Any] = {
        "episode_variant": variant,
        "episode_capture_kind": capture_kind,
        "n_scored_responses": len(outcomes),
        "n_errors": len(errors),
        "errors_sample": errors[:5],
    }
    # `len(outcomes)` is the denominator of every per-response rate below, so it
    # has to be every response.  Rows are only ever missing here when P0d's
    # validators rejected them, and that is reported rather than absorbed.
    if errors:
        summary["denominator_is_complete"] = False
    else:
        summary["denominator_is_complete"] = True
    if not outcomes:
        return summary

    captures = [item for item in outcomes if item.terminal_event == "capture"]
    # Each cycle k is a gap half-slot (novel content, where a normal stop ends
    # the response) followed by an episode half-slot (where a capture can
    # happen).  A response that experienced n episodes therefore SURVIVED n gap
    # slots, plus one more if it terminated between episodes.
    episodes_started = sum(item.n_episodes_experienced for item in outcomes)
    gap_slots = sum(item.n_episodes_experienced + int(item.terminal_stage == "gap")
                    for item in outcomes)
    gap_stops = sum(1 for item in outcomes
                    if item.terminal_stage == "gap" and item.terminal_event == "stop")
    in_episode_stops = sum(1 for item in outcomes
                           if item.terminal_stage == "episode"
                           and item.terminal_event == "stop")
    censors = sum(1 for item in outcomes if item.terminal_event == "censor")

    def rate(numerator: int, denominator: int) -> float | None:
        return (numerator / denominator) if denominator else None

    distances = [item.distance_last_new_triple_blocks for item in captures
                 if item.distance_last_new_triple_blocks is not None]
    first_after = [item.capture_is_first_episode_after_last_new for item in captures
                   if item.capture_is_first_episode_after_last_new is not None]

    summary.update({
        "episodes_per_response": episodes_started / len(outcomes),
        "reuse_blocks_per_response": mean(item.n_reuse_blocks for item in outcomes),
        "responses_with_any_episode_rate": sum(
            1 for item in outcomes if item.n_episodes_experienced > 0) / len(outcomes),
        "n_episodes_started": episodes_started,
        "per_episode_capture_hazard": rate(len(captures), episodes_started),
        "gap_stop_hazard": rate(gap_stops, gap_slots),
        "in_episode_stop_hazard": rate(in_episode_stops, episodes_started),
        "censor_rate": censors / len(outcomes),
        "capture_terminal_rate": len(captures) / len(outcomes),
        # Named apart from the event-level `p_orbit_given_capture`: the
        # denominator here is responses whose TERMINAL event is a capture,
        # which is the episode-axis reading, not "any capture happened".
        "p_orbit_given_terminal_capture": rate(
            sum(1 for item in captures if item.stable_orbit), len(captures)),
        "mean_recovered_episodes": mean(item.recovered_episodes for item in outcomes),
        "captured_episode_ordinal_mean": (
            mean(item.captured_episode_ordinal for item in captures) if captures else None),
        "distance_last_new_triple_blocks_mean": mean(distances) if distances else None,
        "capture_is_first_episode_after_last_new_rate": (
            sum(first_after) / len(first_after) if first_after else None),
    })
    return summary
