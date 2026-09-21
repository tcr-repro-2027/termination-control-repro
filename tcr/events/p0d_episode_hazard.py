"""P0d reuse-episode capture-hazard decomposition.

This module consumes the frozen all-mode ``event_rows.jsonl`` produced by the
01b relation-event detector.  It does not reparse gold data, move the legacy
raw onset, load an LLM/SAE, or change any 01b event definition.  Its single
question is the last mechanistic fork left open by P0b/P0c:

    Does M1 capture more because it experiences MORE reuse episodes
    (exposure via missing termination), or because EACH episode is more
    likely to be captured (per-episode propensity)?

Frozen definitions
------------------

* Block classes (from per-block ``triple_hash`` / ``identity_complete``):
  - reuse block:  identity-complete and its raw-exact triple hash already
    occurred at an earlier identity-complete block (same definition as the
    frozen ``first_nonempty_triple_reuse`` seed);
  - novel block:  identity-complete first occurrence;
  - empty block:  identity-incomplete (never reuse-eligible).
* ``prev(i)``: most recent earlier identity-complete block with the same
  triple hash; ``lag(i) = i - prev(i)``.
* Episode (variant ``lineage``, primary): maximal run of *consecutive* reuse
  blocks further split at motif changes.  A consecutive reuse block ``i``
  extends the open episode iff it stays in the same parser sequence segment
  and is lineage-linked:  ``lag(i) == lag(i-1)`` (same copying process), or
  its triple hash is already in the episode, or ``prev(i)`` points into the
  episode's own blocks/sources.  Novel blocks, empty blocks, segment breaks
  and motif switches all close the episode.
* Episode (variant ``contiguous``, sensitivity): same, but any consecutive
  same-segment reuse block extends the episode (no lineage split).
* Capture point (kinds analysed separately):
  - primary:  ``motif_capture_triple.second_copy_start_0based`` for rows that
    pass the P0b ``is_semantic_capture`` validation;
  - legacy:   ``alignment_evidence.matching_quad_run.block_onset +
    block_period`` for structured-aligned stable orbits.
* Competing outcomes on an interleaved two-stage axis.  Each episode cycle
  ``k`` occupies two half-slots:  the gap slot ``G_k`` (novel content after
  recovery ``k-1``; a normal stop here is a stop event, hit-max is censoring,
  surviving means episode ``k`` starts) and the episode slot ``E_k`` (episode
  ``k`` exists; capture is the capture event, a normal stop that ends the
  response inside the episode is a stop event, hit-max censors, surviving
  means the episode recovered).  Episodes after the capture episode belong to
  the orbit regime and are excluded (capture is absorbing).
* The per-episode capture hazard is therefore CONDITIONAL on the episode
  existing: ``h(k) = captures_k / episodes_started_k``.  Without the split, a
  model that merely stops less would mechanically inflate the per-slot
  capture hazard by its gap-survival factor and fake a propensity effect —
  the exact confound P0d exists to remove.
* Shapley decomposition: identical machinery as P0b
  (``competing_decomposition``), applied on the interleaved axis.  Swapping
  the capture-cause hazards (episode slots) gives the per-episode propensity
  component; swapping the stop-cause hazards (gap + in-episode stops) gives
  the stop/exposure component.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .io_utils import canonical_key, iter_jsonl, sha256_text
from .p0b_competing_risk import (
    MODEL_INDEX,
    MODEL_TAGS,
    cif_metrics,
    competing_decomposition,
    is_semantic_capture,
    paired_result,
    quantile_ci,
    safe_ratio,
    scalar_result,
    validate_prevalence_rows,
)

# Frozen in tcr.events.alignment; imported from P0c to stay single-source.
from .p0c_set_completion import STRUCTURED_ALIGNMENT_TYPES

EPISODE_VARIANTS = ("lineage", "contiguous")
CAPTURE_KINDS = ("primary", "legacy")
TERMINAL_EVENTS = ("capture", "stop", "censor")
STOP_LOCATIONS = ("in_episode", "between_episodes")
CLOSURE_REASONS = (
    "novel_block",
    "empty_block",
    "segment_break",
    "motif_switch",
    "response_end",
    "capture_truncation",
)

BLOCK_NOVEL = 0
BLOCK_REUSE = 1
BLOCK_EMPTY = 2


@dataclass(frozen=True)
class BlockSequence:
    """Validated per-response block-event view of one event row."""

    sample_id: str
    prompt_id: str
    model_tag: str
    seed: int
    n_blocks: int
    hit_max: bool
    stable_orbit: bool
    triple_hashes: tuple[str, ...]
    quad_hashes: tuple[str, ...]
    identity_complete: tuple[bool, ...]
    segments: tuple[int, ...]
    response_sha256: str
    key_id: str
    first_seed_index: int | None
    primary_capture_start: int | None
    legacy_capture_start: int | None
    legacy_orbit_unmapped: bool


@dataclass(frozen=True)
class Episode:
    ordinal: int
    start: int
    end: int
    n_distinct_hashes: int
    closure_reason: str

    @property
    def n_blocks(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class ResponseOutcome:
    """Episode-axis competing outcome of one response for one capture kind."""

    sample_id: str
    prompt_id: str
    model_tag: str
    seed: int
    capture_kind: str
    episode_variant: str
    n_reuse_blocks: int
    n_episodes_experienced: int
    terminal_stage: str  # gap | episode
    terminal_ordinal: int  # cycle index k of the terminal half-slot
    terminal_event: str  # capture | stop | censor
    stop_location: str | None  # in_episode | between_episodes (stop/censor)
    captured_episode_ordinal: int | None
    captured_episode_start: int | None
    captured_episode_n_blocks: int | None
    recovered_episodes: int
    stable_orbit: bool
    hit_max: bool
    legacy_orbit_unmapped: bool
    post_capture_novel_blocks: int | None
    last_new_triple_before_capture: int | None
    distance_last_new_triple_blocks: int | None
    episodes_between_last_new_and_capture: int | None
    capture_is_first_episode_after_last_new: bool | None
    last_new_pair_before_capture: int | None
    distance_last_new_pair_blocks: int | None
    episode_block_lengths: tuple[int, ...] = field(repr=False)
    closure_reasons: tuple[str, ...] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        value = self.__dict__.copy()
        value["episode_block_lengths"] = list(self.episode_block_lengths)
        value["closure_reasons"] = list(self.closure_reasons)
        return value


def load_event_rows(
    result_dir: str | Path, *, allow_sample: bool = False
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Load all-mode prevalence rows with the standard pairing validation."""
    result = Path(result_dir)
    run_manifest_path = result / "run_manifest.json"
    event_rows_path = result / "event_rows.jsonl"
    for path in (run_manifest_path, event_rows_path):
        if not path.exists():
            raise FileNotFoundError(path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    mode = run_manifest.get("selection", {}).get("mode")
    if mode != "all" and not allow_sample:
        raise ValueError(
            f"P0d requires the all-mode 01b result directory (found mode={mode!r}); "
            "set ALLOW_SAMPLE=1 only for smoke tests"
        )
    if bool(run_manifest.get("protocol", {}).get("raw_onset_moved")):
        raise ValueError("P0d requires raw_onset_moved=false")
    # P0d never reads parser diagnostics or audit snippets; dropping them at
    # load time keeps the all-mode file's memory footprint small.
    rows = []
    for row in iter_jsonl(event_rows_path):
        row.pop("parse_diagnostics", None)
        row.pop("audit_snippets", None)
        rows.append(row)
    prevalence, prompt_ids = validate_prevalence_rows(rows)
    return prevalence, prompt_ids, run_manifest


def _capture_start_from_row(row: Mapping[str, Any], n_blocks: int, triple_hashes: Sequence[str]) -> int | None:
    """Validated primary semantic-capture second-copy start."""
    if not is_semantic_capture(row):
        return None
    capture = row["motif_capture_triple"]
    onset = int(capture["block_onset_0based"])
    period = int(capture["block_period"])
    second = int(capture["second_copy_start_0based"])
    confirmed = int(capture["confirmed_at_block_exclusive_0based"])
    sample_id = row.get("sample_id")
    if period <= 0 or onset < 0 or second != onset + period:
        raise ValueError(f"invalid frozen capture coordinates: {sample_id}")
    if confirmed != onset + 3 * period or confirmed > n_blocks:
        raise ValueError(f"invalid frozen capture confirmation: {sample_id}")
    for repeat in (1, 2):
        for offset in range(period):
            if triple_hashes[onset + repeat * period + offset] != triple_hashes[onset + offset]:
                raise ValueError(
                    f"frozen capture does not contain three exact motif copies: {sample_id}"
                )
    return second


def _legacy_capture_start_from_row(
    row: Mapping[str, Any], n_blocks: int, quad_hashes: Sequence[str]
) -> tuple[int | None, bool]:
    """Validated legacy-aligned capture start and the orbit-unmapped flag."""
    legacy = row.get("legacy_orbit", {})
    orbit_exists = bool(legacy.get("exists"))
    alignment_type = str(row.get("alignment_type", ""))
    matching = row.get("alignment_evidence", {}).get("matching_quad_run")
    if not orbit_exists:
        return None, False
    if alignment_type not in STRUCTURED_ALIGNMENT_TYPES or not isinstance(matching, Mapping):
        return None, True
    onset = int(matching["block_onset"])
    period = int(matching["block_period"])
    sample_id = row.get("sample_id")
    if period <= 0 or onset < 0 or onset + 2 * period > n_blocks:
        raise ValueError(f"invalid legacy matching quad run coordinates: {sample_id}")
    for offset in range(period):
        if quad_hashes[onset + period + offset] != quad_hashes[onset + offset]:
            raise ValueError(f"legacy quad run second copy mismatch: {sample_id}")
    return onset + period, False


def extract_block_sequence(row: Mapping[str, Any]) -> BlockSequence:
    sample_id = str(row.get("sample_id"))
    model = str(row.get("model_tag"))
    if model not in MODEL_TAGS:
        raise ValueError(f"unexpected model_tag={model!r}")
    blocks = row.get("block_index", [])
    if not isinstance(blocks, list):
        raise TypeError(f"block_index must be a list: {sample_id}")
    n_blocks = int(row.get("n_blocks", 0))
    if n_blocks != len(blocks):
        raise ValueError(f"n_blocks/block_index length mismatch: {sample_id}")
    triple_hashes = tuple(str(item.get("triple_hash")) for item in blocks)
    quad_hashes = tuple(str(item.get("quad_hash")) for item in blocks)
    identity = tuple(bool(item.get("identity_complete")) for item in blocks)
    segments = tuple(int(item.get("sequence_segment", -1)) for item in blocks)
    declared_identity = int(row.get("n_identity_complete_blocks", 0))
    if sum(identity) != declared_identity:
        raise ValueError(f"identity-complete count mismatch: {sample_id}")

    first_seed = row.get("first_nonempty_triple_reuse", {})
    first_seed_index = (
        int(first_seed["block_index_0based"]) if first_seed.get("exists") else None
    )
    primary_capture = _capture_start_from_row(row, n_blocks, triple_hashes)
    legacy_capture, legacy_unmapped = _legacy_capture_start_from_row(row, n_blocks, quad_hashes)
    return BlockSequence(
        sample_id=sample_id,
        prompt_id=str(row.get("stable_prompt_id")),
        model_tag=model,
        seed=int(row.get("seed")),
        n_blocks=n_blocks,
        hit_max=bool(row.get("legacy_orbit", {}).get("hit_max_tokens")),
        stable_orbit=bool(row.get("legacy_orbit", {}).get("exists")),
        triple_hashes=triple_hashes,
        quad_hashes=quad_hashes,
        identity_complete=identity,
        segments=segments,
        response_sha256=str(row.get("response_sha256")),
        key_id=canonical_key(row.get("key")),
        first_seed_index=first_seed_index,
        primary_capture_start=primary_capture,
        legacy_capture_start=legacy_capture,
        legacy_orbit_unmapped=legacy_unmapped,
    )


def classify_blocks(seq: BlockSequence) -> tuple[list[int], list[int | None]]:
    """Per-block class and most-recent previous occurrence (reuse lineage)."""
    kinds: list[int] = []
    prev: list[int | None] = []
    seen: dict[str, int] = {}
    for index in range(seq.n_blocks):
        if not seq.identity_complete[index]:
            kinds.append(BLOCK_EMPTY)
            prev.append(None)
            continue
        signature = seq.triple_hashes[index]
        earlier = seen.get(signature)
        if earlier is not None:
            kinds.append(BLOCK_REUSE)
            prev.append(earlier)
        else:
            kinds.append(BLOCK_NOVEL)
            prev.append(None)
        seen[signature] = index
    reuse_indices = [index for index, kind in enumerate(kinds) if kind == BLOCK_REUSE]
    recomputed_first = reuse_indices[0] if reuse_indices else None
    if recomputed_first != seq.first_seed_index:
        raise ValueError(
            "recomputed first nonempty reuse disagrees with frozen seed: "
            f"{seq.sample_id} ({recomputed_first} != {seq.first_seed_index})"
        )
    return kinds, prev


def segment_episodes(
    seq: BlockSequence,
    kinds: Sequence[int],
    prev: Sequence[int | None],
    *,
    variant: str,
) -> list[Episode]:
    """Deterministic episode state machine over the block sequence."""
    if variant not in EPISODE_VARIANTS:
        raise ValueError(f"unknown episode variant {variant!r}")
    episodes: list[Episode] = []
    start: int | None = None
    hashes: set[str] = set()
    index_set: set[int] = set()
    last_lag: int | None = None

    def close(end: int, reason: str) -> None:
        nonlocal start, hashes, index_set, last_lag
        if start is None:
            return
        episodes.append(
            Episode(
                ordinal=len(episodes) + 1,
                start=start,
                end=end,
                n_distinct_hashes=len(hashes),
                closure_reason=reason,
            )
        )
        start = None
        hashes = set()
        index_set = set()
        last_lag = None

    for index in range(seq.n_blocks):
        kind = kinds[index]
        if kind != BLOCK_REUSE:
            close(index - 1, "novel_block" if kind == BLOCK_NOVEL else "empty_block")
            continue
        previous = prev[index]
        assert previous is not None
        lag = index - previous
        if start is None:
            start = index
            hashes = {seq.triple_hashes[index]}
            index_set = {index, previous}
            last_lag = lag
            continue
        same_segment = seq.segments[index] == seq.segments[index - 1]
        if not same_segment:
            close(index - 1, "segment_break")
            start = index
            hashes = {seq.triple_hashes[index]}
            index_set = {index, previous}
            last_lag = lag
            continue
        if variant == "lineage":
            linked = (
                lag == last_lag
                or seq.triple_hashes[index] in hashes
                or previous in index_set
            )
            if not linked:
                close(index - 1, "motif_switch")
                start = index
                hashes = {seq.triple_hashes[index]}
                index_set = {index, previous}
                last_lag = lag
                continue
        hashes.add(seq.triple_hashes[index])
        index_set.add(index)
        index_set.add(previous)
        last_lag = lag
    close(seq.n_blocks - 1, "response_end")
    return episodes


def _last_index_before(flags: Sequence[bool], limit: int) -> int | None:
    for index in range(limit - 1, -1, -1):
        if flags[index]:
            return index
    return None


def assign_outcome(
    seq: BlockSequence,
    kinds: Sequence[int],
    episodes: Sequence[Episode],
    *,
    capture_kind: str,
    episode_variant: str,
    pair_first_flags: Sequence[bool] | None = None,
) -> ResponseOutcome:
    if capture_kind not in CAPTURE_KINDS:
        raise ValueError(f"unknown capture kind {capture_kind!r}")
    capture_start = (
        seq.primary_capture_start if capture_kind == "primary" else seq.legacy_capture_start
    )
    n_reuse = sum(1 for kind in kinds if kind == BLOCK_REUSE)
    novel_flags = [kind == BLOCK_NOVEL for kind in kinds]
    legacy_unmapped = seq.legacy_orbit_unmapped if capture_kind == "legacy" else False

    captured_ordinal: int | None = None
    captured_start: int | None = None
    captured_len: int | None = None
    post_capture_novel: int | None = None
    last_new_triple: int | None = None
    d_triple: int | None = None
    episodes_between: int | None = None
    first_after_last_new: bool | None = None
    last_new_pair: int | None = None
    d_pair: int | None = None

    if capture_start is not None:
        if kinds[capture_start] != BLOCK_REUSE:
            raise ValueError(
                f"{capture_kind} capture start is not a nonempty reuse block: "
                f"{seq.sample_id}, block={capture_start}"
            )
        containing = [ep for ep in episodes if ep.start <= capture_start <= ep.end]
        if len(containing) != 1:
            raise AssertionError(
                f"capture block must lie in exactly one episode: {seq.sample_id}"
            )
        captured = containing[0]
        captured_ordinal = captured.ordinal
        captured_start = captured.start
        captured_len = captured.n_blocks
        kept = list(episodes[: captured.ordinal])
        kept[-1] = Episode(
            ordinal=captured.ordinal,
            start=captured.start,
            end=captured.end,
            n_distinct_hashes=captured.n_distinct_hashes,
            closure_reason="capture_truncation",
        )
        post_capture_novel = sum(novel_flags[captured.end + 1 :])
        last_new_triple = _last_index_before(novel_flags, captured.start)
        if last_new_triple is None:
            raise AssertionError(
                f"a captured response must contain a novel block before capture: {seq.sample_id}"
            )
        d_triple = captured.start - last_new_triple
        episodes_between = sum(
            1 for ep in kept[:-1] if ep.start > last_new_triple
        )
        first_after_last_new = episodes_between == 0
        if pair_first_flags is not None:
            last_new_pair = _last_index_before(pair_first_flags, captured.start)
            if last_new_pair is not None:
                d_pair = captured.start - last_new_pair
        terminal_event = "capture"
        terminal_stage = "episode"
        terminal_ordinal = captured.ordinal
        stop_location = None
        episodes_experienced = captured.ordinal
        recovered = captured.ordinal - 1
        episode_lengths = tuple(ep.n_blocks for ep in kept)
        closure_reasons = tuple(ep.closure_reason for ep in kept)
    else:
        episodes_experienced = len(episodes)
        recovered = len(episodes)
        terminal_event = "censor" if (seq.hit_max or legacy_unmapped) else "stop"
        if episodes and episodes[-1].end == seq.n_blocks - 1:
            terminal_stage = "episode"
            terminal_ordinal = len(episodes)
            stop_location = "in_episode"
            recovered = len(episodes) - 1
        else:
            terminal_stage = "gap"
            terminal_ordinal = len(episodes) + 1
            stop_location = "between_episodes"
        episode_lengths = tuple(ep.n_blocks for ep in episodes)
        closure_reasons = tuple(ep.closure_reason for ep in episodes)

    return ResponseOutcome(
        sample_id=seq.sample_id,
        prompt_id=seq.prompt_id,
        model_tag=seq.model_tag,
        seed=seq.seed,
        capture_kind=capture_kind,
        episode_variant=episode_variant,
        n_reuse_blocks=n_reuse,
        n_episodes_experienced=episodes_experienced,
        terminal_stage=terminal_stage,
        terminal_ordinal=terminal_ordinal,
        terminal_event=terminal_event,
        stop_location=stop_location,
        captured_episode_ordinal=captured_ordinal,
        captured_episode_start=captured_start,
        captured_episode_n_blocks=captured_len,
        recovered_episodes=recovered,
        stable_orbit=seq.stable_orbit,
        hit_max=seq.hit_max,
        legacy_orbit_unmapped=legacy_unmapped,
        post_capture_novel_blocks=post_capture_novel,
        last_new_triple_before_capture=last_new_triple,
        distance_last_new_triple_blocks=d_triple,
        episodes_between_last_new_and_capture=episodes_between,
        capture_is_first_episode_after_last_new=first_after_last_new,
        last_new_pair_before_capture=last_new_pair,
        distance_last_new_pair_blocks=d_pair,
        episode_block_lengths=episode_lengths,
        closure_reasons=closure_reasons,
    )


def analyze_sequences(
    sequences: Sequence[BlockSequence],
    *,
    variant: str,
    capture_kind: str,
    pair_flags: Mapping[str, Sequence[bool]] | None = None,
) -> list[ResponseOutcome]:
    outcomes: list[ResponseOutcome] = []
    for seq in sequences:
        kinds, prev = classify_blocks(seq)
        episodes = segment_episodes(seq, kinds, prev, variant=variant)
        outcomes.append(
            assign_outcome(
                seq,
                kinds,
                episodes,
                capture_kind=capture_kind,
                episode_variant=variant,
                pair_first_flags=pair_flags.get(seq.sample_id) if pair_flags else None,
            )
        )
    return outcomes


def validate_capture_identity(
    sequences: Sequence[BlockSequence], outcomes: Sequence[ResponseOutcome]
) -> None:
    """The episode-level capture indicator must reproduce the row-level flag."""
    if len(sequences) != len(outcomes):
        raise AssertionError("sequence/outcome length mismatch")
    for seq, outcome in zip(sequences, outcomes):
        expected = (
            seq.primary_capture_start if outcome.capture_kind == "primary" else seq.legacy_capture_start
        ) is not None
        actual = outcome.terminal_event == "capture"
        if expected != actual:
            raise AssertionError(f"capture identity violated: {seq.sample_id}")


def _terminal_index(outcome: ResponseOutcome) -> int:
    """0-based interleaved index of the terminal half-slot.

    Cycle ``k`` occupies interleaved indices ``2k-2`` (gap ``G_k``) and
    ``2k-1`` (episode ``E_k``).
    """
    k = outcome.terminal_ordinal
    if outcome.terminal_stage == "gap":
        return 2 * k - 2
    if outcome.terminal_stage == "episode":
        return 2 * k - 1
    raise ValueError(f"unknown terminal stage {outcome.terminal_stage!r}")


def build_slot_arrays(
    outcomes: Sequence[ResponseOutcome], prompt_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-prompt/model at-risk and event arrays on the interleaved axis.

    Even indices are gap half-slots (stop/censor between episodes), odd
    indices are episode half-slots (capture, in-episode stop/censor).
    """
    length = max((_terminal_index(outcome) + 1 for outcome in outcomes), default=1)
    length += length % 2  # keep whole cycles: pad to the episode half-slot
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    risk = np.zeros((len(prompt_ids), len(MODEL_TAGS), length), dtype=np.float64)
    capture = np.zeros_like(risk)
    stop = np.zeros_like(risk)
    for outcome in outcomes:
        pidx = prompt_index[outcome.prompt_id]
        midx = MODEL_INDEX[outcome.model_tag]
        t = _terminal_index(outcome)
        risk[pidx, midx, : t + 1] += 1.0
        if outcome.terminal_event == "capture":
            if t % 2 == 0:
                raise AssertionError("capture must occur on an episode half-slot")
            capture[pidx, midx, t] += 1.0
        elif outcome.terminal_event == "stop":
            stop[pidx, midx, t] += 1.0
    if np.any(capture + stop > risk):
        raise AssertionError("half-slot event count exceeds risk set")
    if np.any(capture[..., 0::2] > 0):
        raise AssertionError("capture events recorded on gap half-slots")
    return risk, capture, stop


def _hazard_pair(risk: np.ndarray, capture: np.ndarray, stop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_capture = np.divide(capture, risk, out=np.zeros_like(capture), where=risk > 0)
    h_stop = np.divide(stop, risk, out=np.zeros_like(stop), where=risk > 0)
    if np.any(h_capture + h_stop > 1.0 + 1e-10):
        raise AssertionError("competing episode hazards sum to more than one")
    return h_capture, h_stop


RESPONSE_METRICS: tuple[tuple[str, Callable[[ResponseOutcome], float | None], Callable[[ResponseOutcome], bool]], ...] = (
    ("capture_rate", lambda o: float(o.terminal_event == "capture"), lambda o: True),
    ("stop_rate", lambda o: float(o.terminal_event == "stop"), lambda o: True),
    ("censor_rate", lambda o: float(o.terminal_event == "censor"), lambda o: True),
    ("stable_orbit_rate", lambda o: float(o.stable_orbit), lambda o: True),
    ("mean_episodes_experienced", lambda o: float(o.n_episodes_experienced), lambda o: True),
    ("zero_episode_rate", lambda o: float(o.n_episodes_experienced == 0), lambda o: True),
    ("mean_reuse_blocks", lambda o: float(o.n_reuse_blocks), lambda o: True),
    (
        "captured_episode_ordinal_mean",
        lambda o: float(o.captured_episode_ordinal) if o.captured_episode_ordinal else None,
        lambda o: o.terminal_event == "capture",
    ),
    (
        "recovered_before_capture_mean",
        lambda o: float(o.recovered_episodes),
        lambda o: o.terminal_event == "capture",
    ),
    (
        "capture_in_first_episode_rate",
        lambda o: float(o.captured_episode_ordinal == 1),
        lambda o: o.terminal_event == "capture",
    ),
    (
        "orbit_given_captured",
        lambda o: float(o.stable_orbit),
        lambda o: o.terminal_event == "capture",
    ),
    (
        "orbit_given_uncaptured",
        lambda o: float(o.stable_orbit),
        lambda o: o.terminal_event != "capture",
    ),
    (
        "stop_in_episode_share",
        lambda o: float(o.stop_location == "in_episode"),
        lambda o: o.terminal_event in {"stop", "censor"},
    ),
    (
        "capture_first_episode_after_last_new_rate",
        lambda o: float(bool(o.capture_is_first_episode_after_last_new)),
        lambda o: o.terminal_event == "capture" and o.capture_is_first_episode_after_last_new is not None,
    ),
)


def response_metric_arrays(
    outcomes: Sequence[ResponseOutcome], prompt_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prompt_index = {prompt_id: index for index, prompt_id in enumerate(prompt_ids)}
    names = [name for name, _fn, _elig in RESPONSE_METRICS]
    sums = np.zeros((len(prompt_ids), len(MODEL_TAGS), len(names)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for outcome in outcomes:
        pidx = prompt_index[outcome.prompt_id]
        midx = MODEL_INDEX[outcome.model_tag]
        for kidx, (_name, value_fn, eligible_fn) in enumerate(RESPONSE_METRICS):
            if not eligible_fn(outcome):
                continue
            value = value_fn(outcome)
            if value is None:
                continue
            number = float(value)
            if math.isnan(number):
                continue
            sums[pidx, midx, kidx] += number
            counts[pidx, midx, kidx] += 1.0
    return sums, counts, names


HAZARD_ORDINALS = (1, 2, 3, 4, 5, 6, 7, 8)
CIF_HORIZONS = (1, 2, 3, 5, 8)


def common_support_boundary(risk: np.ndarray, *, n_min: int) -> int:
    """Largest ordinal k with at least ``n_min`` at-risk episodes for BOTH models.

    ``risk`` is the aggregated (model, half-slot) array.  Episode at-risk
    counts are non-increasing in k, so the boundary is well defined; 0 means
    no per-slot region (everything pooled).
    """
    n_cycles = risk.shape[-1] // 2
    boundary = 0
    for k in range(1, n_cycles + 1):
        episode_risk = risk[..., 2 * k - 1]
        if float(np.min(episode_risk)) >= n_min:
            boundary = k
        else:
            break
    return boundary


def _tail_pooled_hazards(
    risk: np.ndarray, capture: np.ndarray, stop: np.ndarray, *, k_pool: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hazard curves with per-model, per-cause pooled rates beyond ``k_pool``.

    Beyond the common-support boundary one model's per-slot hazards sit on
    (near-)empty risk sets; imputing 0 there systematically biases the
    counterfactual swap in the Shapley decomposition (an ordinal a model
    never reaches would be scored as zero propensity).  Instead, ordinals
    > ``k_pool`` share one pooled hazard per model and cause, estimated from
    that model's own tail events — an explicit, pre-registered
    ordinal-constant tail model.
    """
    h_capture, h_stop = _hazard_pair(risk, capture, stop)
    length = risk.shape[-1]
    boundary = 2 * k_pool
    if boundary >= length:
        return h_capture, h_stop
    h_capture = h_capture.copy()
    h_stop = h_stop.copy()
    t = np.arange(length)
    episode_tail = (t >= boundary) & (t % 2 == 1)
    gap_tail = (t >= boundary) & (t % 2 == 0)
    for mask, events, h in (
        (episode_tail, capture, h_capture),
        (episode_tail, stop, h_stop),
        (gap_tail, stop, h_stop),
    ):
        pooled = safe_ratio(np.sum(events[..., mask], axis=-1), np.sum(risk[..., mask], axis=-1))
        pooled = np.nan_to_num(pooled, nan=0.0)
        h[..., mask] = pooled[..., None]
    return h_capture, h_stop


def _slot_summaries(
    risk: np.ndarray, capture: np.ndarray, stop: np.ndarray, *, k_pool: int
) -> dict[str, np.ndarray]:
    """Interleaved-axis metrics on aggregated (…, model, half-slot) counts.

    Per-episode capture hazards are read off episode half-slots (odd
    indices), so they are conditional on the episode existing; gap stop
    hazards are read off even indices.  Observed hazards/CIFs use the raw
    curves; the Shapley decomposition runs on tail-pooled curves so both
    counterfactuals stay identified beyond the common-support boundary.
    """
    h_capture, h_stop = _hazard_pair(risk, capture, stop)
    metrics = cif_metrics(h_capture, h_stop)
    hp_capture, hp_stop = _tail_pooled_hazards(risk, capture, stop, k_pool=k_pool)
    decomposition = competing_decomposition(
        hp_capture[..., 0, :], hp_stop[..., 0, :], hp_capture[..., 1, :], hp_stop[..., 1, :]
    )
    n_cycles = risk.shape[-1] // 2
    episode_slots = slice(1, None, 2)
    gap_slots = slice(0, None, 2)
    out: dict[str, np.ndarray] = {
        "pooled_capture_hazard": safe_ratio(
            np.sum(capture[..., episode_slots], axis=-1),
            np.sum(risk[..., episode_slots], axis=-1),
        ),
        "pooled_gap_stop_hazard": safe_ratio(
            np.sum(stop[..., gap_slots], axis=-1), np.sum(risk[..., gap_slots], axis=-1)
        ),
        "pooled_inepisode_stop_hazard": safe_ratio(
            np.sum(stop[..., episode_slots], axis=-1),
            np.sum(risk[..., episode_slots], axis=-1),
        ),
        "cif_capture_final": metrics["cif_seed"],
        "cif_stop_final": metrics["cif_stop"],
        "propensity_component": decomposition["seed_hazard_component"],
        "exposure_component": decomposition["stop_exposure_component"],
        "decomposition_total": decomposition["total_diff"],
        "identity_error": decomposition["identity_error"],
        "raw_cif_diff": metrics["cif_seed"][..., 1] - metrics["cif_seed"][..., 0],
    }
    for ordinal in HAZARD_ORDINALS:
        if ordinal <= n_cycles:
            out[f"hazard_k{ordinal}"] = h_capture[..., 2 * ordinal - 1]
    tail_start = len(HAZARD_ORDINALS)
    if n_cycles > tail_start:
        tail = slice(2 * tail_start + 1, None, 2)
        out["hazard_k9plus_pooled"] = safe_ratio(
            np.sum(capture[..., tail], axis=-1), np.sum(risk[..., tail], axis=-1)
        )
    for horizon in CIF_HORIZONS:
        if horizon <= n_cycles:
            out[f"cif_capture_at_k{horizon}"] = metrics["cif_seed_curve"][..., 2 * horizon - 1]
    return out


def bootstrap_episode_analysis(
    *,
    risk: np.ndarray,
    capture: np.ndarray,
    stop: np.ndarray,
    metric_sums: np.ndarray,
    metric_counts: np.ndarray,
    metric_names: Sequence[str],
    n_bootstrap: int,
    random_seed: int,
    batch_size: int = 200,
    tail_pool_min: int = 50,
) -> dict[str, Any]:
    """Prompt-cluster paired bootstrap over slot metrics and response metrics.

    The common-support boundary ``k_pool`` is fixed from the observed data
    (not re-chosen per draw) so the decomposition estimand stays stable.
    """
    n_prompts = risk.shape[0]
    risk_agg = np.sum(risk, axis=0)
    k_pool = common_support_boundary(risk_agg, n_min=tail_pool_min)
    observed_slot = _slot_summaries(
        risk_agg, np.sum(capture, axis=0), np.sum(stop, axis=0), k_pool=k_pool
    )
    observed_rates = safe_ratio(np.sum(metric_sums, axis=0), np.sum(metric_counts, axis=0))
    observed_counts = np.sum(metric_counts, axis=0)

    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    rng = np.random.default_rng(random_seed)
    completed = 0
    while completed < n_bootstrap:
        batch = min(batch_size, n_bootstrap - completed)
        weights = rng.multinomial(n_prompts, np.full(n_prompts, 1.0 / n_prompts), size=batch)
        risk_b = np.einsum("bn,nmk->bmk", weights, risk, optimize=True)
        capture_b = np.einsum("bn,nmk->bmk", weights, capture, optimize=True)
        stop_b = np.einsum("bn,nmk->bmk", weights, stop, optimize=True)
        slot_b = _slot_summaries(risk_b, capture_b, stop_b, k_pool=k_pool)
        for name, value in slot_b.items():
            if value.ndim == 2:  # (batch, model) paired metric
                collected[f"slot_{name}"].append(value[:, 1] - value[:, 0])
            else:  # (batch,) decomposition scalar
                collected[f"slot_{name}"].append(value)
        sums_b = np.einsum("bn,nmk->bmk", weights, metric_sums, optimize=True)
        counts_b = np.einsum("bn,nmk->bmk", weights, metric_counts, optimize=True)
        rates_b = safe_ratio(sums_b, counts_b)
        for index, name in enumerate(metric_names):
            collected[f"resp_{name}"].append(rates_b[:, 1, index] - rates_b[:, 0, index])
        completed += batch

    draws = {name: np.concatenate(values) for name, values in collected.items()}
    slot_results: dict[str, dict[str, float]] = {}
    for name, value in observed_slot.items():
        if np.ndim(value) == 1:  # per-model
            slot_results[name] = paired_result(value[0], value[1], draws[f"slot_{name}"])
        else:
            slot_results[name] = scalar_result(float(value), draws[f"slot_{name}"])
    response_results: dict[str, dict[str, float]] = {}
    for index, name in enumerate(metric_names):
        diff_draw = draws[f"resp_{name}"]
        response_results[name] = {
            "m0": float(observed_rates[0, index]),
            "m1": float(observed_rates[1, index]),
            "diff": float(observed_rates[1, index] - observed_rates[0, index]),
            "ci_low": quantile_ci(diff_draw, 0.025),
            "ci_high": quantile_ci(diff_draw, 0.975),
            "n_m0": int(observed_counts[0, index]),
            "n_m1": int(observed_counts[1, index]),
        }
    return {
        "slot": slot_results,
        "response": response_results,
        "kmax": int(risk.shape[-1] // 2),
        "k_pool": int(k_pool),
        "tail_pool_min": int(tail_pool_min),
    }


def episode_curve_rows(
    risk: np.ndarray,
    capture: np.ndarray,
    stop: np.ndarray,
    *,
    variant: str,
    capture_kind: str,
) -> list[dict[str, Any]]:
    """Observed per-cycle table (aggregated over prompts).

    One row per episode ordinal ``k``, exposing both half-slots: the gap
    (stop opportunity before episode ``k``) and the episode itself (capture
    hazard conditional on the episode existing).
    """
    risk_agg = np.sum(risk, axis=0)
    capture_agg = np.sum(capture, axis=0)
    stop_agg = np.sum(stop, axis=0)
    h_capture, h_stop = _hazard_pair(risk_agg, capture_agg, stop_agg)
    metrics = cif_metrics(h_capture, h_stop)
    n_cycles = risk_agg.shape[-1] // 2
    rows: list[dict[str, Any]] = []
    for model in MODEL_TAGS:
        midx = MODEL_INDEX[model]
        for k in range(1, n_cycles + 1):
            gap_t = 2 * k - 2
            epi_t = 2 * k - 1
            if risk_agg[midx, gap_t] <= 0 and risk_agg[1 - midx, gap_t] <= 0:
                continue
            gap_survivors = risk_agg[midx, epi_t]
            next_gap = risk_agg[midx, epi_t + 1] if epi_t + 1 < risk_agg.shape[-1] else 0.0
            rows.append(
                {
                    "episode_variant": variant,
                    "capture_kind": capture_kind,
                    "model": model,
                    "episode_ordinal": k,
                    "gap_at_risk": int(risk_agg[midx, gap_t]),
                    "gap_stop_events": int(stop_agg[midx, gap_t]),
                    "gap_censored": int(risk_agg[midx, gap_t] - stop_agg[midx, gap_t] - gap_survivors),
                    "gap_stop_hazard": float(h_stop[midx, gap_t]),
                    "episodes_started": int(gap_survivors),
                    "capture_events": int(capture_agg[midx, epi_t]),
                    "inepisode_stop_events": int(stop_agg[midx, epi_t]),
                    "inepisode_censored": int(
                        gap_survivors - capture_agg[midx, epi_t] - stop_agg[midx, epi_t] - next_gap
                    ),
                    "capture_hazard_given_episode": float(h_capture[midx, epi_t]),
                    "capture_cif_after_episode": float(metrics["cif_seed_curve"][midx, epi_t]),
                    "stop_cif_after_episode": float(metrics["cif_stop_curve"][midx, epi_t]),
                    "event_free_survival": float(metrics["survival_curve"][midx, epi_t]),
                }
            )
    return rows


def episode_count_rows(
    outcomes: Sequence[ResponseOutcome], *, variant: str, capture_kind: str
) -> list[dict[str, Any]]:
    def bin_label(count: int) -> str:
        if count <= 5:
            return str(count)
        if count <= 10:
            return "6-10"
        if count <= 20:
            return "11-20"
        return ">20"

    labels = [str(i) for i in range(6)] + ["6-10", "11-20", ">20"]
    rows: list[dict[str, Any]] = []
    for model in MODEL_TAGS:
        model_outcomes = [o for o in outcomes if o.model_tag == model]
        total = len(model_outcomes)
        table: dict[str, Counter] = {label: Counter() for label in labels}
        for outcome in model_outcomes:
            table[bin_label(outcome.n_episodes_experienced)][outcome.terminal_event] += 1
        for label in labels:
            counts = table[label]
            n_bin = sum(counts.values())
            rows.append(
                {
                    "episode_variant": variant,
                    "capture_kind": capture_kind,
                    "model": model,
                    "episodes_experienced_bin": label,
                    "n_responses": n_bin,
                    "share_of_model": n_bin / total if total else math.nan,
                    "captured": counts.get("capture", 0),
                    "normal_stop": counts.get("stop", 0),
                    "hit_max_censored": counts.get("censor", 0),
                }
            )
    return rows


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    if not array.size:
        return {"n": 0, "p25": math.nan, "median": math.nan, "p75": math.nan, "mean": math.nan}
    return {
        "n": int(array.size),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "mean": float(np.mean(array)),
    }


def descriptive_episode_summary(
    outcomes: Sequence[ResponseOutcome]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for model in MODEL_TAGS:
        rows = [o for o in outcomes if o.model_tag == model]
        captured = [o for o in rows if o.terminal_event == "capture"]
        episode_lengths = [
            length for o in rows for length in o.episode_block_lengths
        ]
        closure = Counter(reason for o in rows for reason in o.closure_reasons)
        output[model] = {
            "n_responses": len(rows),
            "episodes_experienced": _quantiles([o.n_episodes_experienced for o in rows]),
            "episodes_experienced_captured": _quantiles(
                [o.n_episodes_experienced for o in captured]
            ),
            "captured_episode_ordinal": _quantiles(
                [o.captured_episode_ordinal for o in captured if o.captured_episode_ordinal]
            ),
            "captured_episode_n_blocks": _quantiles(
                [o.captured_episode_n_blocks for o in captured if o.captured_episode_n_blocks]
            ),
            "episode_block_length": _quantiles(episode_lengths),
            "closure_reasons": {reason: int(closure.get(reason, 0)) for reason in CLOSURE_REASONS},
            "post_capture_novel_responses": sum(
                1 for o in captured if (o.post_capture_novel_blocks or 0) > 0
            ),
            "legacy_orbit_unmapped": sum(1 for o in rows if o.legacy_orbit_unmapped),
        }
    return output


def distance_rows(
    outcomes: Sequence[ResponseOutcome], *, variant: str, capture_kind: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_TAGS:
        captured = [
            o
            for o in outcomes
            if o.model_tag == model and o.terminal_event == "capture"
        ]
        triple = _quantiles(
            [o.distance_last_new_triple_blocks for o in captured if o.distance_last_new_triple_blocks is not None]
        )
        between = _quantiles(
            [o.episodes_between_last_new_and_capture for o in captured if o.episodes_between_last_new_and_capture is not None]
        )
        pair_values = [
            o.distance_last_new_pair_blocks for o in captured if o.distance_last_new_pair_blocks is not None
        ]
        pair = _quantiles(pair_values)
        first_after = [
            o.capture_is_first_episode_after_last_new
            for o in captured
            if o.capture_is_first_episode_after_last_new is not None
        ]
        rows.append(
            {
                "episode_variant": variant,
                "capture_kind": capture_kind,
                "model": model,
                "n_captured": len(captured),
                "dist_last_new_triple_blocks_median": triple["median"],
                "dist_last_new_triple_blocks_p25": triple["p25"],
                "dist_last_new_triple_blocks_p75": triple["p75"],
                "episodes_between_last_new_and_capture_median": between["median"],
                "episodes_between_last_new_and_capture_p75": between["p75"],
                "capture_first_episode_after_last_new_share": (
                    sum(first_after) / len(first_after) if first_after else math.nan
                ),
                "n_with_pair_distance": pair["n"],
                "dist_last_new_pair_blocks_median": pair["median"],
                "dist_last_new_pair_blocks_p25": pair["p25"],
                "dist_last_new_pair_blocks_p75": pair["p75"],
            }
        )
    return rows


def reparse_pair_novelty(
    sequences: Sequence[BlockSequence],
    *,
    responses_m0: str | Path,
    responses_m1: str | Path,
    target: str,
) -> dict[str, list[bool]]:
    """Recompute per-block raw-exact (source,target) first-occurrence flags.

    The reparse revalidates each response against the frozen event row
    (response SHA-256, block count, triple hashes, identity flags) before any
    pair novelty is trusted.  Pairs follow the same raw-exact, identity-
    complete convention as the frozen triple reuse definition.
    """
    from .audit import _signature_hash
    from .block_parser import parse_relation_blocks
    from .p0c_set_completion import response_text

    expected: dict[str, dict[str, BlockSequence]] = {"M0": {}, "M1": {}}
    by_key: dict[str, dict[str, set[int]]] = {"M0": defaultdict(set), "M1": defaultdict(set)}
    for seq in sequences:
        expected[seq.model_tag][f"{seq.key_id}::{seq.seed}"] = seq
        by_key[seq.model_tag][seq.key_id].add(seq.seed)

    flags: dict[str, list[bool]] = {}
    for model, path in (("M0", responses_m0), ("M1", responses_m1)):
        seen_pairs: set[tuple[str, int]] = set()
        for row in iter_jsonl(path):
            if "key" not in row:
                raise ValueError(f"response row lacks key: {path}")
            key_id = canonical_key(row["key"])
            relevant = by_key[model].get(key_id, set())
            if not relevant:
                continue
            samples = row.get("responses")
            if not isinstance(samples, list):
                raise ValueError(f"response key={row['key']!r} lacks responses list")
            for sample in samples:
                if not isinstance(sample, Mapping) or sample.get("seed") is None:
                    raise ValueError(f"malformed response sample for key={row['key']!r}")
                seed = int(sample["seed"])
                if seed not in relevant:
                    continue
                seq = expected[model][f"{key_id}::{seed}"]
                text = response_text(sample, target)
                if sha256_text(text) != seq.response_sha256:
                    raise ValueError(f"response SHA mismatch during reparse: {seq.sample_id}")
                parsed = parse_relation_blocks(text, offsets=None, reject_extra_fields=True)
                blocks = parsed.blocks
                if len(blocks) != seq.n_blocks:
                    raise ValueError(f"reparse n_blocks mismatch: {seq.sample_id}")
                hashes = tuple(_signature_hash(block.triple_signature) for block in blocks)
                if hashes != seq.triple_hashes:
                    raise ValueError(f"reparse triple hashes mismatch: {seq.sample_id}")
                identity = tuple(block.identity_complete for block in blocks)
                if identity != seq.identity_complete:
                    raise ValueError(f"reparse identity flags mismatch: {seq.sample_id}")
                pair_seen: set[tuple[str, str]] = set()
                first_flags: list[bool] = []
                for block in blocks:
                    if not block.identity_complete:
                        first_flags.append(False)
                        continue
                    pair = (block.triple_signature[0], block.triple_signature[1])
                    if pair in pair_seen:
                        first_flags.append(False)
                    else:
                        first_flags.append(True)
                        pair_seen.add(pair)
                flags[seq.sample_id] = first_flags
                seen_pairs.add((key_id, seed))
        missing = {
            (seq.key_id, seq.seed)
            for seq in sequences
            if seq.model_tag == model
        } - seen_pairs
        if missing:
            raise ValueError(
                f"response file misses {len(missing)} selected {model} rows during reparse"
            )
    return flags


def decision_from_results(
    primary_results: Mapping[str, Any],
    sensitivity_results: Mapping[str, Any],
    *,
    exposure_share_threshold: float = 0.5,
) -> dict[str, Any]:
    """Pre-registered P0d gate on the (lineage, primary-capture) analysis.

    Branch B (propensity also elevated):  Shapley per-episode capture-
    propensity component CI entirely above 0.
    Branch A (exposure dominated):        propensity CI crosses 0 or is
    entirely below 0, while the stop/exposure component CI is above 0 and
    carries more than half of the total capture-CIF difference.
    Otherwise UNRESOLVED.  The contiguous-variant sensitivity run must map to
    the same branch, or the label is downgraded to SENSITIVITY_DIVERGENT.
    The orbit-stage flag is computed independently from
    P(stable orbit | captured).
    """

    def branch(results: Mapping[str, Any]) -> str:
        propensity = results["slot"]["propensity_component"]
        exposure = results["slot"]["exposure_component"]
        total = results["slot"]["decomposition_total"]["value"]
        if propensity["ci_low"] > 0:
            return "B_PROPENSITY_ALSO_ELEVATED"
        exposure_share = exposure["value"] / total if total else math.nan
        if (
            exposure["ci_low"] > 0
            and math.isfinite(exposure_share)
            and exposure_share > exposure_share_threshold
        ):
            return "A_EXPOSURE_DOMINATED"
        return "UNRESOLVED"

    primary_branch = branch(primary_results)
    sensitivity_branch = branch(sensitivity_results)
    orbit = primary_results["response"]["orbit_given_captured"]
    orbit_flag = orbit["ci_low"] > 0
    if primary_branch == sensitivity_branch:
        label = f"P0D_{primary_branch}"
    else:
        label = f"P0D_SENSITIVITY_DIVERGENT({primary_branch}|{sensitivity_branch})"
    if primary_branch == "A_EXPOSURE_DOMINATED":
        next_step = (
            "冻结单机制主线：终止失准→暴露→偶发capture；进入 R1/S1 终止决策行为量验证；"
            "T-DRS 只做终止成分，F_core/07b 写入 capture→orbit 支撑证据"
        )
    elif primary_branch == "B_PROPENSITY_ALSO_ELEVATED":
        next_step = (
            "两阶段机制：终止失准+捕获增强；R1/S1 照常，R2 后追加 episode 起点固定前缀 "
            "F_core clamp 干预式验证（须过随机匹配对照）"
        )
    else:
        next_step = "分解不可判定；先核对 lineage/contiguous 差异与 slot 级计数，再定分支"
    if orbit_flag:
        next_step += "；orbit-stage flag 触发：P2b 瓶颈修复机制保留为 capture→orbit 阶段证据"
    return {
        "label": label,
        "primary_branch": primary_branch,
        "sensitivity_branch": sensitivity_branch,
        "orbit_stage_flag": bool(orbit_flag),
        "orbit_given_captured": {key: orbit[key] for key in ("m0", "m1", "diff", "ci_low", "ci_high")},
        "next_step": next_step,
    }
