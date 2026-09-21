# coding=utf-8
"""Build the E-SupportConditioning anchor set (§8.3) from the frozen eval data.

What an anchor is
-----------------
A record's gold list is cut a few blocks before its end.  The prefix is what
the assistant has "already emitted"; the remainder is what a supported model
still owes.  Manipulating the INPUT's support for that remainder, and reading
the StopMargin at the cut, is the whole instrument.

Why "a few blocks before the end"
---------------------------------
`REMAINDER_SIZES` is 2-4 on purpose.  "Remove ALL remaining support" then costs
one or two endpoint edits instead of a rewrite, which is what lets a neutral
edit of the same size and position exist at all.  On the frozen eval set a
remainder of 2 leaves 335 fully-removable records -- more than the 128 needed.

The two axes, and why E2 can measure the evidence axis when E1 could not
-----------------------------------------------------------------------
* **evidence**: the endpoint's mentions are replaced in the TEXT.  It stays in
  the candidate list, so admissibility is untouched and the violation is pure.
* **admissibility**: the endpoint is removed from the CANDIDATE LIST and a
  length-matched distractor takes its slot.  It stays in the text, so evidence
  is untouched.

In E1 these two axes collapsed together, because every candidate of every eval
record occurs in its own document, so an *output* endpoint could only fail the
evidence test by also failing the admissibility test.  E2 does not have that
problem: it manipulates the input and reads a margin, so the axes are separated
by construction rather than by what the model happens to emit.

Both axes are built on the SAME records, so ACI and ECI are a within-document
contrast -- which is what H2 (the two axes are dissociable) needs.

    python -m tcr.support_probe.build --eval_data .../eval_supportclean_keep8.jsonl \
        --out anchors/anchors.jsonl --tokenizer ./models/Qwen3/Qwen3-4B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from . import protocol
from .protocol import BUILT_ANCHOR_TYPES
from .anchors import (
    CLOSE_CONTINUATION, CONTINUE_CONTINUATION_BLOCK, CONTINUE_CONTINUATION_EMPTY,
    PC_CONTINUE_NAMED, PC_CONTINUE_PLAIN, PC_NEUTRAL, PC_STOP, Anchor, Variant,
    assistant_prefix, occurrence_gap, validate_anchor,
)
from .io_utils import (
    JsonlWriter, read_jsonl, sha256_file,
    stable_json, write_json,
)

from tcr.evaluation.quality import candidate_entities, norm_field  # noqa: E402

from .prompts import build_extraction_relation_prompt  # noqa: E402

LOG = logging.getLogger("tcr.support_probe.build")


def n_removed(level: float, n_remainder: int) -> int:
    """How many remaining blocks level `r` must strip of support.

    `int(x + 0.5)` rather than `round`: banker's rounding would send r=0.5 on a
    3-block remainder to 2 on one Python version's whim, and the level->count
    map has to be stable across machines."""
    return int(level * n_remainder + 0.5)


def render_entities(entities: Sequence[str]) -> str:
    """Back into the `"['a', 'b']"` form the prompt template expects."""
    return str(list(entities))


def endpoints_of(block: Dict[str, Any]) -> Tuple[str, str]:
    return norm_field(block.get("source", "")), norm_field(block.get("target", ""))


def is_supported(block: Dict[str, Any], candidates: set, norm_text: str) -> bool:
    source, target = endpoints_of(block)
    if not source or not target:
        return False
    return (source in candidates and target in candidates
            and source in norm_text and target in norm_text)


# --------------------------------------------------------------- donor pool

def _stable_index(salt: str, length: int, modulus: int) -> int:
    """A start offset that does not move between processes.

    Python's builtin `hash()` is salted per interpreter (PYTHONHASHSEED), so
    `hash((salt, length))` picks a DIFFERENT donor on every run -- the same
    `--seed` would silently produce a different anchor set, and the report
    would not show it.  sha256 of the same bytes always gives the same index.
    """
    digest = hashlib.sha256(f"{salt}|{length}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(modulus, 1)


class DonorPool:
    """Replacement entities, indexed by length.

    Substitutions use real entities from other documents rather than invented
    glyphs: an invented string is a second, uncontrolled manipulation (the model
    can tell it is not a name), while a real entity from another document is
    exactly what ISC used and keeps the edit in-distribution."""

    def __init__(self, entities: Iterable[str], length_of: Callable[[str], int],
                 *, max_length_delta: int = 0) -> None:
        self.by_length: Dict[int, List[str]] = defaultdict(list)
        for entity in entities:
            self.by_length[length_of(entity)].append(entity)
        for values in self.by_length.values():
            values.sort()
        self.length_of = length_of
        self.max_length_delta = max(0, int(max_length_delta))
        """§8.3 asks for a token-length-matched substitution, and the default
        is therefore EXACT: an anchor with no exact donor is dropped rather
        than built with a near one.  A looser build is still reachable
        (`--max_length_delta`), but only on purpose -- with mismatched lengths
        the manipulated and neutral arms no longer share a context length or a
        decision position, which is the premise the whole delta rests on."""

    def pick(self, target: str, *, forbidden_text: str, forbidden: set,
             salt: str) -> Optional[Tuple[str, int]]:
        """A donor of matched length that appears nowhere in this record.

        Returns `(donor, length_delta)`; the delta is 0 for an exact token match
        and is reported so a build with loose matching is visible.

        A donor is rejected when it NESTS with anything already in the record,
        in either direction -- not only when it equals it.  `is_independent`
        applies that rule to the entity being removed; this is the same rule on
        the insertion side, and it is not symmetric bookkeeping but the same
        defect: substituting `银针 -> 中医学会` into a document whose candidate
        list also holds `中医` manufactures a mention of `中医`, and a
        manufactured mention changes that pair's support exactly as a deleted
        one does.  Measured on the frozen eval set, this is what rejected 7 of
        152 evidence anchors -- with `北京` and `中医` the repeat offenders.
        """
        want = self.length_of(target)
        for delta in range(0, self.max_length_delta + 1):
            for length in ({want - delta, want + delta} if delta else {want}):
                bucket = self.by_length.get(length)
                if not bucket:
                    continue
                start = _stable_index(salt, length, len(bucket))
                for offset in range(len(bucket)):
                    donor = bucket[(start + offset) % len(bucket)]
                    # `is_independent` skips an exact match (it asks about
                    # OTHER entities), so equality is tested separately.
                    if donor == target or donor in forbidden:
                        continue
                    if not is_independent(donor, forbidden):
                        continue
                    if norm_field(donor) in forbidden_text:
                        continue
                    return donor, length - want
        return None


# ------------------------------------------------------------ record analysis

class Record:
    """One eval record, pre-parsed once."""

    def __init__(self, row: Dict[str, Any]) -> None:
        self.key = row["key"]
        self.source = row.get("source", "")
        self.text = row.get("text", "") or ""
        self.norm_text = norm_field(self.text)
        self.entities = candidate_entities(row.get("entities_str", ""))
        self.candidates = {norm_field(name) for name in self.entities}
        gold = row.get("output", [])
        if isinstance(gold, str):
            gold = json.loads(gold)
        self.gold = [b for b in gold if isinstance(b, dict)]
        self.used = set()
        for block in self.gold:
            self.used.update(endpoints_of(block))
        # candidates the gold never touches: the only safe material for a
        # neutral edit, since editing them cannot change any block's support
        self.unused = [name for name in self.entities
                       if norm_field(name) and norm_field(name) not in self.used]

    def raw_of(self, normalised: str) -> Optional[str]:
        """The candidate list's own spelling of a normalised entity."""
        for name in self.entities:
            if norm_field(name) == normalised:
                return name
        return None


def editable_endpoints(record: Record, prefix_blocks: Sequence[Dict[str, Any]],
                       remainder: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Endpoint -> which remaining blocks it would kill.

    Only endpoints the prefix never referenced qualify: editing one the prefix
    used would retroactively invalidate already-emitted output."""
    used_by_prefix = set()
    for block in prefix_blocks:
        used_by_prefix.update(endpoints_of(block))
    kills: Dict[str, List[int]] = defaultdict(list)
    for index, block in enumerate(remainder):
        for entity in endpoints_of(block):
            if entity and entity not in used_by_prefix and entity in record.candidates:
                kills[entity].append(index)
    return dict(kills)


def cover_exactly(kills: Dict[str, List[int]], target: frozenset) -> Optional[List[str]]:
    """Endpoints whose combined kill set is EXACTLY `target`.

    An endpoint shared with a block outside the target would strip support the
    level did not ask for, so it is excluded rather than accepted with a note --
    the level->supported-count map is an invariant the validator enforces."""
    usable = {entity: set(blocks) for entity, blocks in kills.items()
              if set(blocks) <= target}
    covered: set = set()
    chosen: List[str] = []
    while covered != target:
        best = None
        best_gain = 0
        for entity, blocks in usable.items():
            gain = len(blocks - covered)
            if gain > best_gain or (gain == best_gain and gain > 0
                                    and (best is None or entity < best)):
                best, best_gain = entity, gain
        if best is None:
            return None
        chosen.append(best)
        covered |= usable[best]
    return sorted(chosen)


def plan_levels(record: Record, prefix_blocks: Sequence[Dict[str, Any]],
                remainder: Sequence[Dict[str, Any]]) -> Optional[Dict[float, List[str]]]:
    """Which endpoints to edit at each level, or None if the anchor is unusable.

    Every level must be realisable exactly, including r=1 (all remaining blocks
    unsupported).  A partially realisable anchor is dropped: a slope fitted over
    levels that do not mean what they say is worse than a missing anchor."""
    kills = editable_endpoints(record, prefix_blocks, remainder)
    if not kills:
        return None
    n = len(remainder)
    plan: Dict[float, List[str]] = {0.0: []}
    order = list(range(n))
    for level in protocol.LEVELS:
        if level == 0.0:
            continue
        want = n_removed(level, n)
        if want == 0:
            plan[level] = []
            continue
        found = None
        # Small remainders make this exhaustive rather than heuristic.
        for size in (want,):
            for subset in _subsets(order, size):
                chosen = cover_exactly(kills, frozenset(subset))
                if chosen is not None:
                    found = chosen
                    break
            if found is not None:
                break
        if found is None:
            return None
        plan[level] = found
    return plan


def _subsets(items: Sequence[int], size: int) -> Iterable[Tuple[int, ...]]:
    from itertools import combinations
    return combinations(items, size)


# ------------------------------------------------------------ edit application

def is_independent(entity: str, candidates: Sequence[str]) -> bool:
    """No other candidate nests inside it, and it nests inside no other.

    Substituting a nested entity would corrupt its container's mentions in the
    text (`需求管理` inside `需求管理流程`), silently breaking a block the edit
    was never supposed to touch.  §6.4 keeps the main results on Independent
    entities for exactly this reason."""
    for other in candidates:
        if other == entity:
            continue
        if entity in other or other in entity:
            return False
    return True


def apply_text_substitutions(text: str, pairs: Sequence[Tuple[str, str]]) -> str:
    """Replace every mention of each endpoint.

    Every occurrence has to go: leaving one behind leaves the pair evidenced,
    and the level would not mean what it claims.  The replacement is a plain
    string substitution because Chinese has no orthographic word boundary --
    without a segmenter there is no "mention boundary" to align to, and a
    segmenter would put a third-party model's decisions inside the manipulation.

    The residual risk is that an endpoint occurs inside a longer NON-candidate
    word, which the substitution would also rewrite.  Three things bound it:
    `is_independent` already excludes any entity that nests in, or contains,
    another candidate; `validate_anchor` re-checks that no OTHER candidate's
    occurrence count moved, which is the collateral that could actually change a
    support relation; and :func:`mention_context` counts, per edit, how many of
    its occurrences sit inside a longer run of script characters -- the ones
    where "part of a longer word" is even possible.  That count is written into
    the anchor file and aggregated in the build report, so the residual risk is
    a number in the artefact rather than a paragraph in a README.

    Collateral confined to non-entity prose remains possible.  It is the same
    KIND of edit the length-matched neutral arm makes, so it differences out of
    every delta only to the extent the two arms touch comparable amounts of
    text -- which is why the neutral partner is now chosen to match the
    manipulated one on occurrence count as well as on entity count.
    """
    for target, donor in pairs:
        text = text.replace(target, donor)
    return text


def _is_wordish(char: str) -> bool:
    """A character that could make the neighbouring text one longer word.

    CJK ideographs, kana and hangul (no spaces between words), plus letters and
    digits.  Punctuation, whitespace and symbols end a word in every script
    here, so an occurrence bounded by them cannot be the inside of a longer
    one.  This is orthography, not segmentation: it says where a longer word is
    POSSIBLE, never where one is."""
    if char.isalnum():
        return True
    code = ord(char)
    return (0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF or 0x20000 <= code <= 0x2FA1F
            or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF)


def mention_context(text: str, target: str) -> Dict[str, int]:
    """How many occurrences of `target` could be inside a longer word.

    An occurrence flanked by punctuation, whitespace or a line break is
    unambiguous; one flanked by more script characters may be part of something
    longer that no candidate list names, and a global substitution would
    rewrite that too.  `is_independent` rules out the case that can change
    support (nesting inside ANOTHER CANDIDATE); this quantifies what is left.
    """
    total = embedded = 0
    start = 0
    while True:
        index = text.find(target, start)
        if index < 0:
            break
        total += 1
        before = text[index - 1] if index > 0 else ""
        after_index = index + len(target)
        after = text[after_index] if after_index < len(text) else ""
        if (before and _is_wordish(before)) or (after and _is_wordish(after)):
            embedded += 1
        start = index + len(target)
    return {"n_occurrences": total, "n_occurrences_embedded": embedded}


def apply_candidate_substitutions(entities: Sequence[str],
                                  pairs: Sequence[Tuple[str, str]]) -> List[str]:
    """Swap in place, so the list keeps both its length and its ordering."""
    out = list(entities)
    lookup = {target: donor for target, donor in pairs}
    for index, name in enumerate(out):
        if name in lookup:
            out[index] = lookup[name]
    return out


def _variant(*, variant_id: str, arm: str, axis: str, level: float, text: str,
             entities: Sequence[str], instruction: str, n_edits: int,
             remainder: Sequence[Dict[str, Any]],
             edits: Sequence[Dict[str, Any]]) -> Variant:
    candidates = {norm_field(name) for name in entities}
    norm_text = norm_field(text)
    supported = sum(1 for block in remainder
                    if is_supported(block, candidates, norm_text))
    return Variant(variant_id=variant_id, arm=arm, axis=axis, level=level,
                   text=text, entities_str=render_entities(entities),
                   instruction=instruction, n_edits=n_edits,
                   n_remaining_supported=supported, edits=tuple(edits))


def _pc_variants(base: Variant, entities: Sequence[str],
                 remainder: Sequence[Dict[str, Any]], anchor_id: str) -> List[Variant]:
    """The §8.3 E controls: an unambiguous stop, an unambiguous continue, and a
    length-matched instruction that pushes neither way.

    `pc_continue` names a relation from the audited gold remainder instead of
    inventing one, so "an explicitly supported new relation" stays inside
    material E0 already audited."""
    if remainder:
        first = remainder[0]
        continue_text = PC_CONTINUE_NAMED.format(source=first.get("source", ""),
                                                 target=first.get("target", ""))
    else:
        continue_text = PC_CONTINUE_PLAIN
    out: List[Variant] = []
    for arm, instruction in (("pc_stop", PC_STOP), ("pc_continue", continue_text),
                             ("pc_neutral", PC_NEUTRAL)):
        out.append(_variant(variant_id=f"{anchor_id}:{arm}", arm=arm, axis="none",
                            level=0.0, text=base.text, entities=entities,
                            instruction=instruction, n_edits=0,
                            remainder=remainder, edits=()))
    return out


def build_preterminal_anchor(record: Record, axis: str, remainder_size: int,
                             pool: DonorPool) -> Optional[Anchor]:
    """One evidence or admissibility anchor, with every level and control."""
    if len(record.gold) <= remainder_size:
        return None
    prefix_blocks = record.gold[:-remainder_size]
    remainder = record.gold[-remainder_size:]
    plan = plan_levels(record, prefix_blocks, remainder)
    if plan is None:
        return None
    max_edits = max((len(entities) for entities in plan.values()), default=0)
    if max_edits == 0:
        return None

    # Nesting only endangers the EVIDENCE axis, which rewrites the document:
    # substituting `需求管理` would corrupt `需求管理流程`, and an entity nested
    # inside another cannot be removed from the text at all while its container
    # stays.  The admissibility axis swaps entries in the candidate LIST and
    # never touches the text, so it carries no such hazard and is not filtered.
    needs_independent = axis == "evidence"
    raw_targets: Dict[str, str] = {}
    for entities in plan.values():
        for entity in entities:
            raw = record.raw_of(entity)
            if raw is None:
                return None
            if needs_independent and not is_independent(raw, record.entities):
                return None
            raw_targets[entity] = raw

    neutral_pool = [name for name in record.unused
                    if (not needs_independent
                        or is_independent(name, record.entities))
                    and (axis == "admissibility"
                         or norm_field(name) in record.norm_text)]
    if len(neutral_pool) < max_edits:
        return None

    # The neutral arm exists to make "the input was edited at all" cancel out
    # of every delta, and on the evidence axis that cancellation is only as
    # good as the match in HOW MUCH TEXT the two arms rewrite: replacing an
    # endpoint mentioned nine times against a neutral one mentioned once is not
    # the same edit.  Entity count is already matched by construction; this
    # orders the pool so the occurrence counts line up too.  A preference, not
    # a requirement -- the paired-axis pool has little headroom (152 records
    # for 128 anchors) and a hard constraint here could empty it.
    if axis == "evidence":
        wanted = sorted((record.text.count(raw_targets[entity])
                         for entities in plan.values() for entity in entities),
                        reverse=True)
        typical = wanted[0] if wanted else 1
        neutral_pool.sort(key=lambda name: (abs(record.text.count(name) - typical),
                                            name))

    anchor_id = f"{axis}:{stable_json(record.key)}:r{remainder_size}"
    prefix = assistant_prefix(prefix_blocks)

    def donors_for(names: Sequence[str], salt: str
                   ) -> Optional[List[Tuple[str, str, int]]]:
        picked: List[Tuple[str, str, int]] = []
        blocked = set(record.entities)
        for name in names:
            choice = pool.pick(name, forbidden_text=record.norm_text,
                               forbidden=blocked, salt=f"{salt}:{name}")
            if choice is None:
                return None
            donor, delta = choice
            blocked.add(donor)
            picked.append((name, donor, delta))
        return picked

    base = _variant(variant_id=f"{anchor_id}:base", arm="base", axis=axis,
                    level=0.0, text=record.text, entities=record.entities,
                    instruction="", n_edits=0, remainder=remainder, edits=())
    if base.n_remaining_supported != len(remainder):
        return None                  # the untouched record must support them all
    variants: List[Variant] = [base]

    for level in protocol.LEVELS:
        if level == 0.0:
            continue
        targets = [raw_targets[entity] for entity in plan[level]]
        n_edits = len(targets)
        manip = donors_for(targets, f"{anchor_id}:manip:{level}")
        neutral = donors_for(neutral_pool[:n_edits], f"{anchor_id}:neutral:{level}")
        if manip is None or neutral is None:
            return None
        for arm, picked in (("manip", manip), ("neutral", neutral)):
            pairs = [(name, donor) for name, donor, _ in picked]
            edits = [{"axis": axis, "arm": arm, "from": name, "to": donor,
                      "length_delta": delta,
                      # how much text this edit actually touched, and how much
                      # of it could have been inside a longer non-candidate
                      # word (the residual risk of a global substitution)
                      **mention_context(record.text, name)}
                     for name, donor, delta in picked]
            if axis == "evidence":
                text = apply_text_substitutions(record.text, pairs)
                entities = list(record.entities)
            else:
                text = record.text
                entities = apply_candidate_substitutions(record.entities, pairs)
            variants.append(_variant(
                variant_id=f"{anchor_id}:{arm}:r{level}", arm=arm, axis=axis,
                level=level, text=text, entities=entities, instruction="",
                n_edits=n_edits, remainder=remainder, edits=edits))

    variants.extend(_pc_variants(base, record.entities, remainder, anchor_id))
    return Anchor(anchor_id=anchor_id, anchor_type=axis, key=record.key,
                  source=record.source, n_gold_blocks=len(record.gold),
                  prefix_blocks=len(prefix_blocks), remainder=tuple(remainder),
                  assistant_prefix=prefix,
                  close_continuation=CLOSE_CONTINUATION,
                  continue_continuation=CONTINUE_CONTINUATION_BLOCK,
                  variants=tuple(variants))


def build_zero_answer_anchor(record: Record, donor: Record) -> Optional[Anchor]:
    """§8.3 D: an input whose correct answer is provably `[]`.

    The document is kept and the candidate list is replaced wholesale by another
    document's, verified to contain no entity occurring in this text.  No listed
    pair can then be evidenced, so `[]` is the only supported answer and the
    margin measured after `[` is an EmptyMargin."""
    entities = [name for name in donor.entities
                if norm_field(name) and norm_field(name) not in record.norm_text]
    if len(entities) < max(8, len(record.entities) // 4):
        return None
    entities = entities[:len(record.entities)]
    anchor_id = f"zero_answer:{stable_json(record.key)}"
    base = _variant(variant_id=f"{anchor_id}:base", arm="base", axis="none",
                    level=0.0, text=record.text, entities=entities,
                    instruction="", n_edits=0, remainder=(), edits=())
    variants = [base, *_pc_variants(base, entities, (), anchor_id)]
    return Anchor(anchor_id=anchor_id, anchor_type="zero_answer", key=record.key,
                  source=record.source, n_gold_blocks=len(record.gold),
                  prefix_blocks=0, remainder=(), assistant_prefix="[",
                  close_continuation=CLOSE_CONTINUATION,
                  continue_continuation=CONTINUE_CONTINUATION_EMPTY,
                  variants=tuple(variants))


# ---------------------------------------------------------------- selection

def stratified_order(items: Sequence[Any], stratum_of: Callable[[Any], str],
                     key_of: Callable[[Any], str], salt: str) -> List[Any]:
    """One ordering whose EVERY prefix is proportional across strata.

    Anchors are stratified by source document so a cut of 128 cannot come
    mostly from one book.  Items are ranked inside their stratum by a stable
    hash and given the fractional position `(i + 0.5) / n_s`; sorting globally
    on that interleaves the strata, so taking the first N is a stratified
    sample for any N -- including a later, larger N."""
    buckets: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        buckets[stratum_of(item)].append(item)
    ranked: List[Tuple[float, str, Any]] = []
    for stratum, members in buckets.items():
        members.sort(key=lambda item: (
            hashlib.sha256(f"{salt}|{stratum}|{key_of(item)}".encode("utf-8")).hexdigest(),
            key_of(item)))
        n = len(members)
        for index, item in enumerate(members):
            ranked.append(((index + 0.5) / n, f"{stratum}|{key_of(item)}", item))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return [item for _fraction, _tie, item in ranked]


def variant_tokens(anchor: Anchor, variant: Variant,
                   length_of: Callable[[str], int], chat_overhead: int,
                   render_chat: Optional[Callable[[str], str]] = None) -> int:
    """The scored context for one variant, in tokens.

    With ``render_chat`` this tokenises the EXACT string the scorer will run --
    the chat-templated user turn plus the assistant prefix, as one string.
    Without it (a pilot build with no tokenizer), it falls back to adding up
    the parts plus ``chat_overhead``, which is an estimate: BPE merges across
    the joins, so the sum of the pieces is not the length of the whole.
    """
    prompt = build_extraction_relation_prompt(text=variant.text,
                                              entities_str=variant.entities_str)
    if render_chat is not None:
        return length_of(render_chat(prompt + variant.instruction)
                         + anchor.assistant_prefix)
    return (length_of(prompt + variant.instruction)
            + length_of(anchor.assistant_prefix) + chat_overhead)


def context_tokens(anchor: Anchor, length_of: Callable[[str], int],
                   chat_overhead: int = 0,
                   render_chat: Optional[Callable[[str], str]] = None) -> int:
    """Size of the LONGEST variant's context, for the length gate.

    Every variant, measured in tokens.  Picking the longest by characters and
    measuring only that one is not the same thing: the candidate list's BPE
    merges differ between the axes (evidence rewrites the document, admissibility
    rewrites the list), so the character-longest variant need not be the
    token-longest, and an over-long admissibility variant could ship behind a
    short evidence one.  An anchor over the limit is DROPPED, not truncated --
    truncation would silently change the very prefix the margin is conditioned
    on -- so a missed variant is a cell that cannot be scored at all."""
    return max(variant_tokens(anchor, variant, length_of, chat_overhead,
                              render_chat)
               for variant in anchor.variants)


# --------------------------------------------------------------------- main

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the E-SupportConditioning anchors (§8.3).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_data", required=True,
                        help="Frozen E-Natural jsonl; anchors come from it so "
                             "the gold blocks are the ones E0 audited.")
    parser.add_argument("--out", required=True, help="anchors.jsonl to write.")
    parser.add_argument("--tokenizer", default=None,
                        help="Qwen3 tokenizer dir.  With it, donor entities are "
                             "matched on TOKEN length and the context gate is "
                             "exact; without it both fall back to characters "
                             "and the report says so.")
    parser.add_argument("--n_per_type", type=int, default=protocol.N_ANCHORS_PER_TYPE)
    parser.add_argument("--report", default=None,
                        help="default: <out>.report.json")
    parser.add_argument("--seed", type=int, default=protocol.BUILD_SEED)
    parser.add_argument("--max_length_delta", type=int, default=0,
                        help="Tokens a donor entity may differ from the one it "
                             "replaces.  0 (the default) makes §8.3's "
                             "length-matched substitution an INVARIANT: an "
                             "anchor with no exact donor is dropped, not built "
                             "with a near one.  Raise it only deliberately -- "
                             "mismatched lengths move the decision position "
                             "between the manipulated and neutral arms.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    length_of: Callable[[str], int]
    token_matched = False
    chat_overhead = 0
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        cache: Dict[str, int] = {}

        def length_of(text: str) -> int:                      # noqa: F811
            hit = cache.get(text)
            if hit is None:
                hit = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                if len(text) < 64:            # only small strings are worth caching
                    cache[text] = hit
            return hit

        token_matched = True

        # The scored context is the CHAT-TEMPLATED user turn plus the assistant
        # prefix.  With a tokenizer the strict gate tokenises exactly that
        # string; `chat_overhead` is only the fallback estimate's allowance for
        # the template, and adding up the pieces is not the same as measuring
        # the whole (BPE merges across the joins).
        def render_chat(user: str) -> str:                    # noqa: F811
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)

        try:
            chat_overhead = length_of(render_chat(""))
        except (TypeError, ValueError) as exc:                # pragma: no cover
            render_chat = None                                # type: ignore
            chat_overhead = 64
            LOG.warning("chat template not usable (%s); the length gate falls "
                        "back to an estimate with a %d-token allowance",
                        exc, chat_overhead)
        LOG.info("chat template overhead: %d tokens", chat_overhead)
    else:
        LOG.warning("no --tokenizer: donor length matching and the context gate "
                    "fall back to characters; this is a pilot build")
        render_chat = None                                    # type: ignore

        def length_of(text: str) -> int:                      # noqa: F811
            return len(text)

    records = [Record(row) for row in read_jsonl(args.eval_data)]
    LOG.info("loaded %d eval records", len(records))

    pool = DonorPool((name for record in records for name in record.entities),
                     length_of, max_length_delta=args.max_length_delta)

    # ---- preterminal anchors: both axes on the same records (H2 needs the
    # within-document contrast), so a record qualifies only if BOTH build.
    feasible: List[Tuple[Record, int]] = []
    axis_only = {"evidence": 0, "admissibility": 0}
    for record in records:
        matched = False
        for size in protocol.REMAINDER_SIZES:
            built = {axis: build_preterminal_anchor(record, axis, size, pool)
                     for axis in ("evidence", "admissibility")}
            if any(anchor is None for anchor in built.values()):
                continue
            # A cheap screen on one axis; the strict all-variant check runs
            # during selection, where being wrong actually costs something.
            if (context_tokens(built["evidence"], length_of, chat_overhead)
                    > protocol.MAX_CONTEXT_TOKENS):
                continue
            feasible.append((record, size))
            matched = True
            break
        # Records that build on ONE axis only are counted for the report: the
        # paired pool is the binding constraint, and knowing how much bigger a
        # single-axis run could be is what tells you whether a shortfall is
        # about the data or about the pairing requirement.
        if not matched:
            for axis in ("evidence", "admissibility"):
                if any(build_preterminal_anchor(record, axis, size, pool) is not None
                       for size in protocol.REMAINDER_SIZES):
                    axis_only[axis] += 1
    LOG.info("records feasible for BOTH preterminal axes: %d "
             "(evidence-only %d, admissibility-only %d more)",
             len(feasible), axis_only["evidence"], axis_only["admissibility"])

    def checked(anchor: Optional[Anchor]) -> Tuple[Optional[Anchor], List[str]]:
        """An anchor and why it is unusable -- validated HERE, during selection.

        Validating only after the first `n_per_type` were taken made an
        invariant failure unrecoverable: the anchor was dropped, nothing
        replaced it, and the build failed as under-filled with no next
        candidate ever tried.  On the frozen eval set that really happens (7
        of 152 evidence anchors, from donors that manufacture a mention of
        another candidate), so an anchor set could not be built at all.

        Now a failure only costs its record: the next one in the stratified
        order takes its place, and the count of skipped records goes into the
        report so a pool quietly eaten by rejections is still visible."""
        if anchor is None:
            return None, ["not constructible"]
        problems = validate_anchor(anchor, norm=norm_field,
                                   parse_candidates=candidate_entities)
        # EVERY variant against the context limit, tokenising the EXACT string
        # the scorer will run.  An over-long variant is a cell that cannot be
        # scored, and the feasibility screen only estimated one variant of one
        # axis.  ~2400 tokenisations, about 20 s -- worth it here, where being
        # wrong costs a cell; the screen above stays cheap.
        longest = context_tokens(anchor, length_of, chat_overhead, render_chat)
        if longest > protocol.MAX_CONTEXT_TOKENS:
            problems = problems + [
                f"longest variant is {longest} tokens, over the "
                f"{protocol.MAX_CONTEXT_TOKENS} limit"]
        return (None, problems) if problems else (anchor, [])

    ordered = stratified_order(feasible, lambda item: item[0].source,
                               lambda item: stable_json(item[0].key),
                               f"{args.seed}:preterminal")

    anchors: List[Anchor] = []
    chosen: List[Tuple[Record, int]] = []
    skipped: List[Dict[str, Any]] = []
    for record, size in ordered:
        if len(chosen) >= args.n_per_type:
            break
        pair: List[Anchor] = []
        problems: List[str] = []
        for axis in ("evidence", "admissibility"):
            anchor, issues = checked(build_preterminal_anchor(record, axis,
                                                              size, pool))
            if anchor is None:
                problems.append(f"{axis}: {issues[0]}")
            else:
                pair.append(anchor)
        # Both axes or neither: ACI and ECI are a within-document contrast
        # (H2), so a record that only half builds is not usable for either.
        if len(pair) != 2:
            skipped.append({"key": record.key, "problems": problems})
            continue
        chosen.append((record, size))
        anchors.extend(pair)
    # ---- zero-answer anchors
    by_index = {index: record for index, record in enumerate(records)}
    zero_candidates: List[Tuple[Record, Record]] = []
    for index, record in enumerate(records):
        for offset in (211, 433, 617, 809, 97):
            donor = by_index[(index + offset) % len(records)]
            anchor = build_zero_answer_anchor(record, donor)
            if anchor is None:
                continue
            if (context_tokens(anchor, length_of, chat_overhead)
                    > protocol.MAX_CONTEXT_TOKENS):
                continue
            zero_candidates.append((record, donor))
            break
    LOG.info("records with a clean zero-answer donor: %d", len(zero_candidates))
    zero_ordered = stratified_order(zero_candidates, lambda item: item[0].source,
                                    lambda item: stable_json(item[0].key),
                                    f"{args.seed}:zero")
    n_zero = 0
    for record, donor in zero_ordered:
        if n_zero >= args.n_per_type:
            break
        anchor, issues = checked(build_zero_answer_anchor(record, donor))
        if anchor is None:
            skipped.append({"key": record.key, "problems": [f"zero: {issues[0]}"]})
            continue
        anchors.append(anchor)
        n_zero += 1
    if skipped:
        LOG.warning("passed over %d record(s) whose anchors failed an "
                    "invariant; the next in the stratified order took their "
                    "place.  e.g. %s", len(skipped), skipped[0])

    # ---- §E2 instrument criterion 4: nothing ships that fails an invariant.
    # Every anchor was already validated during selection, so this is the
    # last-line re-check: `n_rejected` feeding gate 4 must be 0, and if it ever
    # is not, the two code paths have drifted apart.
    kept: List[Anchor] = []
    rejected: List[Dict[str, Any]] = []
    for anchor in anchors:
        problems = validate_anchor(anchor, norm=norm_field,
                                   parse_candidates=candidate_entities)
        if problems:
            rejected.append({"anchor_id": anchor.anchor_id, "problems": problems})
        else:
            kept.append(anchor)
    if rejected:
        LOG.error("%d anchor(s) failed validation, e.g. %s", len(rejected),
                  rejected[0])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written under a temporary name and renamed only once the report is in
    # place.  A build killed halfway must not leave a short anchors.jsonl
    # behind: the launcher skips construction when the file exists, so a
    # truncated file would be reused in silence for the rest of the experiment.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with JsonlWriter(tmp_path, mode="w") as handle:
        for anchor in kept:
            handle.write(anchor.as_dict())

    counts: Dict[str, int] = defaultdict(int)
    variant_counts: Dict[str, int] = defaultdict(int)
    edit_deltas: List[int] = []
    # Residual risk of the global text substitution (§6.4 / the honest list):
    # how many evidence-axis edits had at least one occurrence that could sit
    # inside a longer non-candidate word, and how well the neutral arm matches
    # the manipulated one on the amount of text rewritten.
    embedded_edits = evidence_edits = matched_anchors = 0
    occurrence_gaps: List[int] = []
    for anchor in kept:
        counts[anchor.anchor_type] += 1
        occurrences: Dict[Tuple[float, str], int] = {}
        for variant in anchor.variants:
            variant_counts[variant.arm] += 1
            edit_deltas.extend(int(edit.get("length_delta", 0))
                               for edit in variant.edits)
            if anchor.anchor_type == "evidence":
                for edit in variant.edits:
                    evidence_edits += 1
                    if int(edit.get("n_occurrences_embedded", 0)) > 0:
                        embedded_edits += 1
                if variant.arm in ("manip", "neutral"):
                    occurrences[(variant.level, variant.arm)] = sum(
                        int(edit.get("n_occurrences", 0)) for edit in variant.edits)
        for level in {level for level, _arm in occurrences}:
            manip_n = occurrences.get((level, "manip"))
            neutral_n = occurrences.get((level, "neutral"))
            if manip_n is not None and neutral_n is not None:
                occurrence_gaps.append(abs(manip_n - neutral_n))
        # The same quantity the readouts carry and the analysis filters on:
        # matched at EVERY level, not per level.  Reported here so the size of
        # the sensitivity subset is known before any GPU time is spent.
        if occurrence_gap(anchor) == 0:
            matched_anchors += 1

    report = {
        "protocol": protocol.describe(),
        "eval_data": str(args.eval_data),
        "eval_data_sha256": sha256_file(args.eval_data),
        "seed": args.seed,
        "token_matched": token_matched,
        "tokenizer": args.tokenizer,
        "n_records": len(records),
        "n_feasible_preterminal_records": len(feasible),
        "n_feasible_single_axis_only": dict(axis_only),
        "pool_headroom": round(len(feasible) / max(args.n_per_type, 1), 2),
        # Records the stratified order reached and had to pass over because an
        # anchor failed an invariant.  It comes straight off the headroom, so a
        # pool being eaten by rejections is visible before it runs out.
        "n_skipped_by_validation": len(skipped),
        "skipped_by_validation": skipped[:20],
        "n_zero_answer_candidates": len(zero_candidates),
        "anchors_by_type": dict(counts),
        "variants_by_arm": dict(variant_counts),
        "n_variants": sum(len(anchor.variants) for anchor in kept),
        "max_length_delta_allowed": args.max_length_delta,
        "donor_length_delta": {
            "exact": sum(1 for value in edit_deltas if value == 0),
            "total": len(edit_deltas),
            "max_abs": max((abs(value) for value in edit_deltas), default=0),
        },
        # The evidence axis rewrites the document with a plain string
        # substitution (no segmenter -- see `apply_text_substitutions`).  These
        # two numbers are the residual risk, carried as data:
        #   * how many edits touched at least one occurrence that could be
        #     inside a longer word no candidate list names;
        #   * how far the neutral arm's rewritten-occurrence count sits from
        #     its manipulated counterpart's, which is how much of that
        #     collateral actually cancels in the delta.
        "evidence_substitution_risk": {
            "edits": evidence_edits,
            "edits_with_embedded_occurrence": embedded_edits,
            "share_embedded": (round(embedded_edits / evidence_edits, 4)
                               if evidence_edits else 0.0),
            "neutral_occurrence_gap_mean": (
                round(sum(occurrence_gaps) / len(occurrence_gaps), 3)
                if occurrence_gaps else 0.0),
            "neutral_occurrence_gap_max": max(occurrence_gaps, default=0),
            "neutral_occurrence_exact": sum(1 for gap in occurrence_gaps if gap == 0),
            "neutral_occurrence_pairs": len(occurrence_gaps),
            # anchors matched at EVERY level -- the subset the analysis reports
            # `slope_occurrence_matched` on
            "occurrence_matched_anchors": matched_anchors,
        },
        "rejected": rejected[:20],
        "n_rejected": len(rejected),
        "n_anchors": len(kept),
        "n_per_type_target": args.n_per_type,
        "anchors_sha256": sha256_file(tmp_path),
        "output": str(out_path),
    }
    report_path = Path(args.report) if args.report else \
        out_path.with_suffix(out_path.suffix + ".report.json")

    # A shortfall is a HARD failure, not a warning.  An under-filled type still
    # produces a perfectly ordinary-looking summary -- fewer anchors only widen
    # the CI -- so nothing downstream would ever reveal that one axis had been
    # measured on a tenth of the intended sample, or not at all.
    missing = {name: counts.get(name, 0) for name in BUILT_ANCHOR_TYPES
               if counts.get(name, 0) < args.n_per_type}
    # Cheap, but it pins the property rather than describing it: with the
    # default --max_length_delta 0 the pool cannot return a loose donor, and
    # this makes that an assertion instead of a statistic in the report.
    over_length = report["donor_length_delta"]["max_abs"] > args.max_length_delta
    if missing or rejected or over_length or not kept:
        tmp_path.unlink(missing_ok=True)
        if missing:
            LOG.error("under-filled anchor types %s (wanted %d each); refusing "
                      "to ship a partial anchor set -- pass a smaller "
                      "--n_per_type explicitly if that is really the intent",
                      missing, args.n_per_type)
        if over_length:
            LOG.error("a donor differs by %d token(s) from the entity it "
                      "replaces, above the allowed %d",
                      report["donor_length_delta"]["max_abs"],
                      args.max_length_delta)
        if rejected:
            LOG.error("%d anchor(s) failed validation; nothing written",
                      len(rejected))
        write_json(report_path.with_suffix(".failed.json"), report)
        return 1

    write_json(report_path, report)
    os.replace(tmp_path, out_path)          # report lands first, then anchors

    LOG.info("wrote %d anchors (%s) / %d variants -> %s", len(kept),
             dict(counts), report["n_variants"], out_path)
    LOG.info("report -> %s", report_path)
    if len(feasible) < 1.25 * args.n_per_type:
        LOG.warning("only %d records qualify for %d paired anchors: the "
                    "stratification has little room, so a later change to the "
                    "eval set or the levels could push it under the target",
                    len(feasible), args.n_per_type)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
