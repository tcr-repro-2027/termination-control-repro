"""S1 anchor construction (A / B / C) from frozen 01b + P0c artifacts.

Anchors are complete-block boundaries turned into exact token-aligned
prefixes.  No model is loaded here; only the fast tokenizer (for replay) and
the shared relation-extraction prompt builder in ``tcr.prompt_template``
are required.

Frozen anchor definitions
-------------------------

* A (M0 self-selected stop boundary): an M0 response that terminated normally
  (``finish_reason == "stop"``, no hit-max, no stable orbit, no semantic
  capture, full JSON list valid), truncated to the end of its LAST complete
  block — the position where M0 actually chose ``]``.
* B (M1 effective-completion boundary): an M1 overrun response (hit-max),
  truncated to the end of the block holding its own LAST new relaxed gold
  relation (``last_new_relaxed_block_1based`` from the frozen P0c summary) —
  the response's own effective completion point.  When the response has a
  primary semantic capture, the boundary must lie strictly before the capture
  second-copy start.
* C (progress scan): additional boundaries of the B-donor responses at the
  complete blocks nearest to 0.5/0.75/1.0/1.25 x gold G.  Plot-only; never
  part of the main gate.

Both models are later measured on the SAME anchor prefix (same decision
point, same prefix), which is the R-a-compliant comparison.

Token geometry (the Qwen BPE reality)
-------------------------------------

Qwen's byte-level BPE merges the block-closing ``"}`` with what FOLLOWS it:
``"},`` / ``"},\n`` are single tokens on the continue side while the close
side tokenizes as ``"}``/``"}\n`` + ``]``.  A block's char_end is therefore
usually NOT a token boundary, and the real continue-vs-close decision happens
AT the token that closes the block (``"},…`` vs ``"}]``-family) — exactly the
plan's ``"},\n{`` vs ``"}]+EOS`` formulation.  Anchors are therefore built on
the LONGEST COMMON TOKEN PREFIX of the two textual variants::

    close_variant    = text[:char_end] + close_text     (A: the donor's real tail)
    continue_variant = text[:char_end] + opener_text    (donor's modal inter-block opener)

The shared prefix (ending typically at the last description token) is stored
as token ids verbatim; the close/continue paths are the diverging token
tails, whose FIRST tokens are the real decision alternatives.  Openers and
close tails are extracted char-wise (no token-alignment assumption).  The
close path's final EOS step is added at measurement time from the model's
generation config.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .constants import (
    CLOSE_TAIL_MAX_CHARS,
    CLOSE_TAIL_MAX_TOKENS,
    C_PROGRESS_FRACTIONS,
    MIN_BLOCK_B,
    MIN_BLOCKS_A,
    N_A_ANCHORS,
    N_B_ANCHORS,
    N_C_PROMPTS,
    OPENER_COVER_MAX_CHARS,
    OPENER_COVER_SUBSTRING,
    OPENER_MAX_TOKENS,
    SELECTION_SEED,
)
from .io_utils import canonical_key, sha256_text

DECISIVE_CHARS = {"]", ",", "{"}


@dataclass(frozen=True)
class BlockSpan:
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    sequence_segment: int
    identity_complete: bool


@dataclass(frozen=True)
class DonorRow:
    sample_id: str
    prompt_id: str
    model_tag: str
    key: Any
    key_id: str
    seed: int
    n_blocks: int
    blocks: tuple[BlockSpan, ...]
    hit_max: bool
    stable_orbit: bool
    full_json_list_valid: bool
    semantic_capture: bool
    capture_second_copy_start: int | None
    response_sha256: str


@dataclass
class AnchorDraft:
    anchor_type: str
    donor: DonorRow
    boundary_block_1based: int
    gold_blocks: int
    last_new_relaxed_block_1based: int | None


@dataclass
class BuiltAnchor:
    anchor_id: str
    anchor_type: str
    sample_id: str
    prompt_id: str
    key: Any
    seed: int
    source_model: str
    boundary_block_1based: int
    n_blocks: int
    gold_blocks: int
    progress_boundary: float
    boundary_char_end: int
    prefix_token_count: int
    prefix_response_ids: list[int]
    boundary_aligned: bool  # prefix ends exactly at the block's char_end
    gap_text: str  # chars between the prefix end and char_end ('' if aligned)
    prompt_text: str
    prompt_sha256: str
    prefix_response_text: str
    response_sha256: str
    finish_reason: str
    close_tail_ids: list[int]
    close_tail_text: str
    close_tail_source: str  # own | modal
    continue_ids: list[int]
    continue_text: str
    continue_source: str  # own_modal | global_modal
    boundary_before_primary_capture: bool | None
    last_new_relaxed_block_1based: int | None
    same_as_b_boundary: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def is_semantic_capture_row(row: Mapping[str, Any]) -> bool:
    event = row.get("motif_capture_triple", {})
    if not isinstance(event, Mapping) or not event.get("exists"):
        return False
    triples = event.get("motif_triples", [])
    return bool(triples) and all(
        isinstance(triple, list)
        and len(triple) == 3
        and all(isinstance(value, str) and value.strip() for value in triple)
        for triple in triples
    )


def donor_from_event_row(row: Mapping[str, Any]) -> DonorRow:
    blocks_raw = row.get("block_index", [])
    if not isinstance(blocks_raw, list):
        raise TypeError(f"block_index must be a list: {row.get('sample_id')}")
    blocks = tuple(
        BlockSpan(
            char_start=int(item["char_start"]),
            char_end=int(item["char_end"]),
            token_start=int(item["token_start"]),
            token_end=int(item["token_end"]),
            sequence_segment=int(item.get("sequence_segment", -1)),
            identity_complete=bool(item.get("identity_complete")),
        )
        for item in blocks_raw
    )
    n_blocks = int(row.get("n_blocks", 0))
    if n_blocks != len(blocks):
        raise ValueError(f"n_blocks mismatch: {row.get('sample_id')}")
    capture = row.get("motif_capture_triple", {})
    semantic = is_semantic_capture_row(row)
    return DonorRow(
        sample_id=str(row.get("sample_id")),
        prompt_id=str(row.get("stable_prompt_id")),
        model_tag=str(row.get("model_tag")),
        key=row.get("key"),
        key_id=canonical_key(row.get("key")),
        seed=int(row.get("seed")),
        n_blocks=n_blocks,
        blocks=blocks,
        hit_max=bool(row.get("legacy_orbit", {}).get("hit_max_tokens")),
        stable_orbit=bool(row.get("legacy_orbit", {}).get("exists")),
        full_json_list_valid=bool(row.get("full_json_list_valid")),
        semantic_capture=semantic,
        capture_second_copy_start=(
            int(capture["second_copy_start_0based"]) if semantic else None
        ),
        response_sha256=str(row.get("response_sha256")),
    )


def select_candidates(
    donors: Sequence[DonorRow],
    p0c_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[list[AnchorDraft], list[AnchorDraft], dict[str, int]]:
    """Deterministic A/B donor candidate lists (RNG-ordered, oversampled).

    One donor per prompt (lowest eligible seed).  The returned lists are
    longer than N_A/N_B so downstream tokenizer-level validation can skip
    donors and backfill without a new selection round.
    """
    skip = Counter()

    def p0c(sample_id: str) -> Mapping[str, Any]:
        row = p0c_rows.get(sample_id)
        if row is None:
            raise KeyError(f"P0C summary misses sample {sample_id}")
        return row

    a_by_prompt: dict[str, DonorRow] = {}
    b_by_prompt: dict[str, DonorRow] = {}
    for donor in sorted(donors, key=lambda d: (d.prompt_id, d.seed)):
        if donor.model_tag == "M0":
            if donor.hit_max or donor.stable_orbit or donor.semantic_capture:
                skip["a_state"] += 1
                continue
            if not donor.full_json_list_valid:
                skip["a_invalid_json"] += 1
                continue
            if donor.n_blocks < MIN_BLOCKS_A:
                skip["a_too_short"] += 1
                continue
            a_by_prompt.setdefault(donor.prompt_id, donor)
        elif donor.model_tag == "M1":
            if not donor.hit_max:
                skip["b_not_hitmax"] += 1
                continue
            summary = p0c(donor.sample_id)
            last_new = summary.get("last_new_relaxed_block_1based")
            if last_new is None:
                skip["b_no_last_new_relaxed"] += 1
                continue
            last_new = int(last_new)
            if last_new < MIN_BLOCK_B or last_new >= donor.n_blocks:
                skip["b_boundary_out_of_range"] += 1
                continue
            if donor.capture_second_copy_start is not None and (
                last_new - 1 >= donor.capture_second_copy_start
            ):
                skip["b_boundary_not_before_capture"] += 1
                continue
            b_by_prompt.setdefault(donor.prompt_id, donor)

    rng = random.Random(SELECTION_SEED)
    a_prompts = sorted(a_by_prompt)
    b_prompts = sorted(b_by_prompt)
    rng.shuffle(a_prompts)
    rng.shuffle(b_prompts)

    def draft_a(donor: DonorRow) -> AnchorDraft:
        summary = p0c(donor.sample_id)
        return AnchorDraft(
            anchor_type="A",
            donor=donor,
            boundary_block_1based=donor.n_blocks,
            gold_blocks=int(summary["gold_blocks"]),
            last_new_relaxed_block_1based=None,
        )

    def draft_b(donor: DonorRow) -> AnchorDraft:
        summary = p0c(donor.sample_id)
        return AnchorDraft(
            anchor_type="B",
            donor=donor,
            boundary_block_1based=int(summary["last_new_relaxed_block_1based"]),
            gold_blocks=int(summary["gold_blocks"]),
            last_new_relaxed_block_1based=int(summary["last_new_relaxed_block_1based"]),
        )

    a_drafts = [draft_a(a_by_prompt[p]) for p in a_prompts]
    b_drafts = [draft_b(b_by_prompt[p]) for p in b_prompts]
    skip["a_eligible_prompts"] = len(a_prompts)
    skip["b_eligible_prompts"] = len(b_prompts)
    return a_drafts, b_drafts, dict(skip)


def validate_response(donor: DonorRow, text: str) -> None:
    if sha256_text(text) != donor.response_sha256:
        raise ValueError(f"response SHA mismatch: {donor.sample_id}")
    last = donor.blocks[donor.n_blocks - 1]
    if last.char_end > len(text):
        raise ValueError(f"frozen block char span exceeds response length: {donor.sample_id}")


def extract_close_text(text: str, char_end: int) -> str | None:
    """The donor's real post-boundary close text (A donors only)."""
    tail = text[char_end:]
    if not tail or len(tail) > CLOSE_TAIL_MAX_CHARS:
        return None
    if not tail.lstrip().startswith("]"):
        return None
    return tail


def extract_opener_texts(donor: DonorRow, text: str, *, before_block_1based: int) -> list[str]:
    """Char-wise inter-block continue openers strictly before the anchor.

    Opener text = separator between block j and block j+1 plus the head of
    block j+1 up to (and including) its ``"source`` key.  No token-alignment
    assumption: divergence handling happens later on the token level.
    """
    openers: list[str] = []
    for j in range(1, before_block_1based):
        prev_span = donor.blocks[j - 1]
        next_span = donor.blocks[j]
        head = text[next_span.char_start : min(next_span.char_start + OPENER_COVER_MAX_CHARS, next_span.char_end)]
        position = head.find(OPENER_COVER_SUBSTRING)
        if position < 0:
            continue
        cover_end = next_span.char_start + position + len(OPENER_COVER_SUBSTRING)
        openers.append(text[prev_span.char_end : cover_end])
    return openers


def modal_text(items: Sequence[str]) -> str | None:
    if not items:
        return None
    counts = Counter(items)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def tokenize_anchor_paths(
    tokenizer,
    *,
    text: str,
    char_end: int,
    close_text: str,
    opener_text: str,
) -> tuple[dict[str, Any] | None, str]:
    """Common-token-prefix construction of the prefix and both decision paths.

    Returns ``(payload, reason)``; ``payload`` is None when the anchor must be
    skipped, with ``reason`` naming the diagnostic counter.
    """
    close_variant = text[:char_end] + close_text
    cont_variant = text[:char_end] + opener_text
    enc_close = tokenizer(close_variant, add_special_tokens=False, return_offsets_mapping=True)
    enc_cont = tokenizer(cont_variant, add_special_tokens=False, return_offsets_mapping=True)
    ids_close = list(enc_close["input_ids"])
    ids_cont = list(enc_cont["input_ids"])
    off_close = [(int(a), int(b)) for a, b in enc_close["offset_mapping"]]
    off_cont = [(int(a), int(b)) for a, b in enc_cont["offset_mapping"]]

    limit = min(len(ids_close), len(ids_cont))
    shared = 0
    while shared < limit and ids_close[shared] == ids_cont[shared]:
        shared += 1
    if shared == 0:
        return None, "no_common_token_prefix"
    if shared >= len(ids_close) or shared >= len(ids_cont):
        return None, "path_fully_shared"

    prefix_char_end = off_close[shared - 1][1]
    if prefix_char_end > char_end + len(close_text):
        return None, "prefix_beyond_variants"
    gap_text = close_variant[prefix_char_end:char_end] if prefix_char_end < char_end else ""
    if any(ch in DECISIVE_CHARS for ch in gap_text):
        return None, "gap_contains_decisive_char"

    close_path = ids_close[shared:]
    if not close_path or len(close_path) > CLOSE_TAIL_MAX_TOKENS:
        return None, "close_path_length"

    cover_position = cont_variant.find(OPENER_COVER_SUBSTRING, char_end - 2)
    if cover_position < 0:
        return None, "opener_cover_missing"
    cover_end = cover_position + len(OPENER_COVER_SUBSTRING)
    cont_path: list[int] = []
    for index in range(shared, len(ids_cont)):
        cont_path.append(ids_cont[index])
        if off_cont[index][1] >= cover_end:
            break
    if not cont_path or len(cont_path) > OPENER_MAX_TOKENS:
        return None, "continue_path_length"
    if close_path[0] == cont_path[0]:
        return None, "paths_share_first_token"

    return (
        {
            "prefix_ids": ids_close[:shared],
            "prefix_text": close_variant[:prefix_char_end],
            "prefix_char_end": prefix_char_end,
            "boundary_aligned": prefix_char_end == char_end,
            "gap_text": gap_text,
            "close_ids": close_path,
            "close_text": close_variant[prefix_char_end:],
            "continue_ids": cont_path,
            "continue_text": cont_variant[prefix_char_end : off_cont[shared + len(cont_path) - 1][1]],
        },
        "",
    )


def build_anchor(
    draft: AnchorDraft,
    payload: Mapping[str, Any],
    *,
    anchor_type: str,
    boundary_block_1based: int,
    prompt_text: str,
    finish_reason: str,
    close_source: str,
    opener_source: str,
    same_as_b: bool = False,
) -> BuiltAnchor:
    donor = draft.donor
    span = donor.blocks[boundary_block_1based - 1]
    capture_flag = (
        None
        if donor.capture_second_copy_start is None
        else boundary_block_1based - 1 < donor.capture_second_copy_start
    )
    anchor_id = f"{anchor_type}_{sha256_text(donor.sample_id)[:10]}_b{boundary_block_1based}"
    return BuiltAnchor(
        anchor_id=anchor_id,
        anchor_type=anchor_type,
        sample_id=donor.sample_id,
        prompt_id=donor.prompt_id,
        key=donor.key,
        seed=donor.seed,
        source_model=donor.model_tag,
        boundary_block_1based=boundary_block_1based,
        n_blocks=donor.n_blocks,
        gold_blocks=draft.gold_blocks,
        progress_boundary=boundary_block_1based / draft.gold_blocks,
        boundary_char_end=span.char_end,
        prefix_token_count=len(payload["prefix_ids"]),
        prefix_response_ids=list(payload["prefix_ids"]),
        boundary_aligned=bool(payload["boundary_aligned"]),
        gap_text=str(payload["gap_text"]),
        prompt_text=prompt_text,
        prompt_sha256=sha256_text(prompt_text),
        prefix_response_text=str(payload["prefix_text"]),
        response_sha256=donor.response_sha256,
        finish_reason=finish_reason,
        close_tail_ids=list(payload["close_ids"]),
        close_tail_text=str(payload["close_text"]),
        close_tail_source=close_source,
        continue_ids=list(payload["continue_ids"]),
        continue_text=str(payload["continue_text"]),
        continue_source=opener_source,
        boundary_before_primary_capture=capture_flag,
        last_new_relaxed_block_1based=draft.last_new_relaxed_block_1based,
        same_as_b_boundary=same_as_b,
    )


def c_boundaries(donor: DonorRow, gold_blocks: int) -> list[tuple[float, int]]:
    picks: list[tuple[float, int]] = []
    used: set[int] = set()
    for fraction in C_PROGRESS_FRACTIONS:
        block = int(round(fraction * gold_blocks))
        block = max(1, min(block, donor.n_blocks))
        if block not in used:
            used.add(block)
            picks.append((fraction, block))
    return picks


def build_all_anchors(
    *,
    a_drafts: Sequence[AnchorDraft],
    b_drafts: Sequence[AnchorDraft],
    response_lookup: Callable[[DonorRow], Mapping[str, Any]],
    prompt_builder: Callable[[str, str], str],
    tokenizer,
    n_a: int = N_A_ANCHORS,
    n_b: int = N_B_ANCHORS,
    n_c_prompts: int = N_C_PROMPTS,
) -> tuple[list[BuiltAnchor], dict[str, Any]]:
    """Validate drafts char-wise, tokenize decision paths, build final anchors."""
    diagnostics: Counter = Counter()
    a_built: list[BuiltAnchor] = []
    b_built: list[BuiltAnchor] = []
    c_built: list[BuiltAnchor] = []
    a_close_texts: list[str] = []
    global_opener_texts: list[str] = []

    def prepare(draft: AnchorDraft):
        sample = response_lookup(draft.donor)
        text = sample.get("response", "")
        if not isinstance(text, str) or not text:
            diagnostics["missing_response_text"] += 1
            return None
        validate_response(draft.donor, text)
        prompt_text = prompt_builder(str(sample["text"]), str(sample["entities_str"]))
        finish_reason = str(sample.get("finish_reason", ""))
        return text, prompt_text, finish_reason

    def paths_for(
        draft: AnchorDraft,
        text: str,
        *,
        boundary_block_1based: int,
        close_text: str,
        tag: str,
    ):
        char_end = draft.donor.blocks[boundary_block_1based - 1].char_end
        own_openers = extract_opener_texts(
            draft.donor, text, before_block_1based=boundary_block_1based
        )
        opener = modal_text(own_openers)
        opener_source = "own_modal"
        if opener is None:
            opener = modal_text(global_opener_texts)
            opener_source = "global_modal"
            diagnostics[f"{tag}_opener_fallback_global"] += 1
        if opener is None:
            diagnostics[f"{tag}_no_opener"] += 1
            return None
        payload, reason = tokenize_anchor_paths(
            tokenizer,
            text=text,
            char_end=char_end,
            close_text=close_text,
            opener_text=opener,
        )
        if payload is None:
            diagnostics[f"{tag}_{reason}"] += 1
            return None
        return payload, opener_source, own_openers

    # --- pass 1: A anchors (their real close texts feed the modal) ----------
    for draft in a_drafts:
        if len(a_built) >= n_a:
            break
        prepared = prepare(draft)
        if prepared is None:
            continue
        text, prompt_text, finish_reason = prepared
        if finish_reason and finish_reason != "stop":
            diagnostics["a_finish_reason_not_stop"] += 1
            continue
        char_end = draft.donor.blocks[draft.boundary_block_1based - 1].char_end
        close_text = extract_close_text(text, char_end)
        if close_text is None:
            diagnostics["a_atypical_close_tail"] += 1
            continue
        result = paths_for(
            draft,
            text,
            boundary_block_1based=draft.boundary_block_1based,
            close_text=close_text,
            tag="a",
        )
        if result is None:
            continue
        payload, opener_source, own_openers = result
        global_opener_texts.extend(own_openers)
        a_close_texts.append(close_text)
        a_built.append(
            build_anchor(
                draft,
                payload,
                anchor_type="A",
                boundary_block_1based=draft.boundary_block_1based,
                prompt_text=prompt_text,
                finish_reason=finish_reason,
                close_source="own",
                opener_source=opener_source,
            )
        )

    modal_close = modal_text(a_close_texts)
    if modal_close is None:
        raise RuntimeError(
            "no valid A anchors could be built; diagnostics="
            f"{dict(diagnostics)} (selection sizes: a_drafts={len(a_drafts)}, b_drafts={len(b_drafts)})"
        )

    # --- pass 2: B anchors + C boundaries from the first valid B donors -----
    c_prompts_used = 0
    for draft in b_drafts:
        if len(b_built) >= n_b:
            break
        prepared = prepare(draft)
        if prepared is None:
            continue
        text, prompt_text, finish_reason = prepared
        result = paths_for(
            draft,
            text,
            boundary_block_1based=draft.boundary_block_1based,
            close_text=modal_close,
            tag="b",
        )
        if result is None:
            continue
        payload, opener_source, _own = result
        b_built.append(
            build_anchor(
                draft,
                payload,
                anchor_type="B",
                boundary_block_1based=draft.boundary_block_1based,
                prompt_text=prompt_text,
                finish_reason=finish_reason,
                close_source="modal",
                opener_source=opener_source,
            )
        )
        if c_prompts_used < n_c_prompts:
            c_prompts_used += 1
            for fraction, block in c_boundaries(draft.donor, draft.gold_blocks):
                c_result = paths_for(
                    draft,
                    text,
                    boundary_block_1based=block,
                    close_text=modal_close,
                    tag="c",
                )
                if c_result is None:
                    continue
                c_payload, c_source, _own = c_result
                c_anchor = build_anchor(
                    draft,
                    c_payload,
                    anchor_type="C",
                    boundary_block_1based=block,
                    prompt_text=prompt_text,
                    finish_reason=finish_reason,
                    close_source="modal",
                    opener_source=c_source,
                    same_as_b=block == draft.boundary_block_1based,
                )
                c_anchor.notes["c_fraction"] = fraction
                c_built.append(c_anchor)

    if len(a_built) < n_a:
        diagnostics["a_underfilled"] = n_a - len(a_built)
    if len(b_built) < n_b:
        diagnostics["b_underfilled"] = n_b - len(b_built)

    anchors = a_built + b_built + c_built
    seen_ids = Counter(anchor.anchor_id for anchor in anchors)
    duplicates = [aid for aid, count in seen_ids.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"duplicate anchor ids: {duplicates[:3]}")
    aligned = sum(anchor.boundary_aligned for anchor in anchors)
    manifest = {
        "n_a": len(a_built),
        "n_b": len(b_built),
        "n_c": len(c_built),
        "n_boundary_aligned": aligned,
        "modal_close_tail_text": modal_close,
        "global_opener_text": modal_text(global_opener_texts),
        "gap_text_counts": dict(Counter(anchor.gap_text for anchor in anchors)),
        "diagnostics": dict(diagnostics),
    }
    return anchors, manifest
