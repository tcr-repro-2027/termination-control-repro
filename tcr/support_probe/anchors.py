# coding=utf-8
"""The anchor data model, its legality invariants, and the control texts.

An **anchor** is one place in one record where the model has already emitted a
legal partial answer and now stands at a block boundary.  A **variant** is that
same anchor with the INPUT edited: only the user prompt ever changes, never the
assistant prefix, which is what makes a difference between two variants
attributable to the edit (see `protocol`).

The invariants in :func:`validate_anchor` are the code form of §E2's instrument
criterion 4 ("anchor parser、prefix legality 和 token 对齐 100% 通过").  They
are checked at build time and again before scoring, because an anchor that
violates one of them produces a number that looks fine and means nothing:

* the assistant prefix must serialise EXACTLY as training serialises gold, and
  must end at a block boundary (right after `}`);
* every block in the prefix must remain supported in EVERY variant -- an edit
  that retroactively invalidates already-emitted output is measuring something
  else;
* the manipulated variant must actually remove the support it claims to, and
  the neutral variant must remove none, on the same axis;
* the neutral variant must make the same NUMBER of edits as its manipulated
  counterpart at the same level.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------ serialising

def serialize_blocks(blocks: Sequence[Dict[str, Any]]) -> str:
    """Exactly how the training target renders a relation list.

    `build_training_formats.py` writes `json.dumps(output, ensure_ascii=False)`,
    whose default separators are `', '` and `': '`; a list is therefore
    `"[" + ", ".join(dumps(block)) + "]"`.  The prefix below is built from the
    same join, so it is a genuine prefix of a training target rather than a
    lookalike."""
    return "[" + ", ".join(json.dumps(block, ensure_ascii=False)
                           for block in blocks) + "]"


def assistant_prefix(blocks: Sequence[Dict[str, Any]]) -> str:
    """The emitted answer up to and including the last block's `}`."""
    if not blocks:
        return "["
    return "[" + ", ".join(json.dumps(block, ensure_ascii=False)
                           for block in blocks)


#: What the two branches of the decision look like as text.  The scorer finds
#: the first token at which they diverge, so these only need to be long enough
#: to differ at their first character.
CLOSE_CONTINUATION = "]"
CONTINUE_CONTINUATION_BLOCK = ", {"
"""After `...}`: either the list closes, or another object follows."""
CONTINUE_CONTINUATION_EMPTY = '{"'
"""After `[`: either the list closes empty, or the first object follows.

Measured on Qwen3, `[]` is a single token, so this decision actually resolves
ON the `[` rather than after it: the branches are `[]` against `[`.  That is
the on-distribution form of §8.3 D's "measure the EmptyMargin after `[`" -- an
empty answer's canonical tokenisation never contains a standalone `[`, so a
context ending in one could not be reached by generation."""


# ------------------------------------------------------------- control texts

#: NONE of these may name the close bracket.  What they control is the score of
#: the CLOSE branch, and both control texts used to say it outright ("请立即输出
#: `]`" / "不要输出 `]`").  For a model that follows instructions that is
#: harmless.  For one that has narrowed onto a fixed output format it is a
#: confound pointing the SAME way in both arms -- the token is simply present in
#: recent context -- so the control measures token priming instead of
#: instruction following, and its two arms stop being opposites.
#:
#: That is what the first calibration run looks like.  Relative to the neutral
#: instruction, BOTH arms moved the two SFT'd 4B models toward closing
#: (pc_stop +0.11 / +0.02 nat, pc_continue +0.40 / +0.36), which makes
#: PC = stop - continue negative and fails gate 2.  The untrained model, which
#: does follow the instruction, moved the other way on the continue arm
#: (-1.71 / -1.27 nat) and passed.  Removing the bracket does not prove that
#: was the cause; it removes the only reading under which the control cannot
#: work at all.
PC_STOP = ("\n\n# 补充指令\n"
           "关系已经抽取完整，没有需要补充的关系对象。"
           "请立即结束并关闭当前 JSON 列表。")

PC_CONTINUE_NAMED = ("\n\n# 补充指令\n"
                     "列表尚未完成。原文中还存在「{source}」与「{target}」"
                     "之间的关系，请继续输出下一个关系对象。")

PC_CONTINUE_PLAIN = ("\n\n# 补充指令\n"
                     "列表尚未完成，还有关系对象没有输出。"
                     "请继续输出下一个关系对象。")

PC_NEUTRAL = ("\n\n# 补充指令\n"
              "输出请保持标准 JSON 列表格式，字段顺序与示例一致，"
              "不要添加额外的说明文字。")

#: `pc_continue` names a relation that is already in the audited gold remainder
#: rather than inventing one, so "an explicitly supported new relation" (§8.3 E)
#: stays inside audited material.  Terminal and zero-answer anchors have no
#: remainder and fall back to the plain instruction.
PC_ARMS = ("pc_stop", "pc_continue", "pc_neutral")
SUPPORT_ARMS = ("base", "manip", "neutral")


# ------------------------------------------------------------------ data model

@dataclasses.dataclass(frozen=True)
class Variant:
    """One (anchor, edit) cell: a complete model input."""
    variant_id: str
    arm: str                      # base | manip | neutral | pc_*
    axis: str                     # evidence | admissibility | evidence_add | none
    level: float                  # normalised manipulation strength r
    text: str                     # the document as the model will see it
    entities_str: str             # the candidate list as the model will see it
    instruction: str              # appended control instruction ("" if none)
    n_edits: int                  # positions edited on this axis
    n_remaining_supported: int    # remaining gold blocks still fully supported
    edits: Tuple[Dict[str, Any], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        row = dataclasses.asdict(self)
        row["edits"] = list(self.edits)
        return row


@dataclasses.dataclass(frozen=True)
class Anchor:
    """One decision point, with every variant measured at it."""
    anchor_id: str
    anchor_type: str              # evidence | admissibility | evidence_add | zero_answer
    key: Any
    source: str
    n_gold_blocks: int
    prefix_blocks: int            # how many blocks the assistant already emitted
    remainder: Tuple[Dict[str, Any], ...]
    assistant_prefix: str
    close_continuation: str
    continue_continuation: str
    variants: Tuple[Variant, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "anchor_type": self.anchor_type,
            "key": self.key,
            "source": self.source,
            "n_gold_blocks": self.n_gold_blocks,
            "prefix_blocks": self.prefix_blocks,
            "n_remainder": len(self.remainder),
            "remainder": list(self.remainder),
            "assistant_prefix": self.assistant_prefix,
            "close_continuation": self.close_continuation,
            "continue_continuation": self.continue_continuation,
            "variants": [variant.as_dict() for variant in self.variants],
        }


def occurrence_gap(anchor: "Anchor") -> Optional[int]:
    """How unequal the manipulated and neutral arms are in TEXT REWRITTEN.

    The neutral arm exists so that "the input was edited at all" cancels out of
    every delta.  It is matched on the NUMBER of entities edited by
    construction; it is only preferred, not required, to match on how many
    mentions those entities have.  So the cancellation is exact where this is
    0 and approximate where it is not, and the analysis reports the slope on
    the matched subset separately (§9.5 sensitivity) rather than claiming a
    perfectly clean evidence-axis effect.

    Returns the worst gap over the levels, or None when the anchor edits no
    text (the admissibility axis rewrites the candidate list, not the
    document, and zero-answer anchors edit nothing).
    """
    if anchor.anchor_type != "evidence":
        return None
    totals: Dict[Tuple[float, str], int] = {}
    for variant in anchor.variants:
        if variant.arm not in ("manip", "neutral"):
            continue
        totals[(variant.level, variant.arm)] = sum(
            int(edit.get("n_occurrences", 0)) for edit in variant.edits)
    gaps = [abs(totals[(level, "manip")] - totals[(level, "neutral")])
            for level, arm in totals if arm == "manip"
            and (level, "neutral") in totals]
    return max(gaps) if gaps else None


def anchor_from_dict(row: Dict[str, Any]) -> Anchor:
    return Anchor(
        anchor_id=row["anchor_id"], anchor_type=row["anchor_type"],
        key=row["key"], source=row.get("source", ""),
        n_gold_blocks=int(row["n_gold_blocks"]),
        prefix_blocks=int(row["prefix_blocks"]),
        remainder=tuple(row.get("remainder", [])),
        assistant_prefix=row["assistant_prefix"],
        close_continuation=row["close_continuation"],
        continue_continuation=row["continue_continuation"],
        variants=tuple(
            Variant(variant_id=v["variant_id"], arm=v["arm"], axis=v["axis"],
                    level=float(v["level"]), text=v["text"],
                    entities_str=v["entities_str"],
                    instruction=v.get("instruction", ""),
                    n_edits=int(v.get("n_edits", 0)),
                    n_remaining_supported=int(v.get("n_remaining_supported", 0)),
                    edits=tuple(v.get("edits", ())))
            for v in row["variants"]),
    )


# ------------------------------------------------------------------ invariants

def _supported(block: Dict[str, Any], candidates: set, norm_text: str,
               norm) -> bool:
    source = norm(block.get("source", ""))
    target = norm(block.get("target", ""))
    if not source or not target:
        return False
    admissible = source in candidates and target in candidates
    evidenced = source in norm_text and target in norm_text
    return admissible and evidenced


def _mention_counts(text: str, names) -> Dict[str, int]:
    return {name: text.count(name) for name in names}


def validate_anchor(anchor: Anchor, *, norm, parse_candidates) -> List[str]:
    """Every way an anchor can be silently wrong.  Empty list = usable."""
    problems: List[str] = []
    prefix = anchor.assistant_prefix

    # --- shape of the decision point
    if not prefix.startswith("["):
        problems.append("assistant prefix does not open the JSON list")
    if anchor.prefix_blocks == 0:
        if prefix != "[":
            problems.append("zero-block prefix must be exactly '['")
        if anchor.continue_continuation != CONTINUE_CONTINUATION_EMPTY:
            problems.append("empty-list anchor must continue with an object")
    else:
        if not prefix.endswith("}"):
            problems.append("assistant prefix does not end at a block boundary")
        if anchor.continue_continuation != CONTINUE_CONTINUATION_BLOCK:
            problems.append("block anchor must continue with ', {'")
    if anchor.close_continuation != CLOSE_CONTINUATION:
        problems.append("close continuation must be ']'")

    # Closing the list here must yield exactly the blocks already emitted --
    # that is both a parse check and the way the prefix blocks are recovered
    # without storing them a second time.
    prefix_blocks: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(prefix + anchor.close_continuation)
        if not isinstance(parsed, list):
            problems.append("prefix + close is not a JSON list")
        else:
            prefix_blocks = [b for b in parsed if isinstance(b, dict)]
            if len(prefix_blocks) != len(parsed):
                problems.append("prefix holds a non-object item")
            if len(prefix_blocks) != anchor.prefix_blocks:
                problems.append(f"prefix holds {len(prefix_blocks)} blocks but "
                                f"declares {anchor.prefix_blocks}")
    except json.JSONDecodeError as exc:
        problems.append(f"prefix + close does not parse as JSON: {exc}")

    # --- one prefix for every variant is the whole point
    arms = [variant.arm for variant in anchor.variants]
    if "base" not in arms:
        problems.append("no base variant")
    for arm in PC_ARMS:
        if arm not in arms:
            problems.append(f"missing positive control {arm}")

    base = next((v for v in anchor.variants if v.arm == "base"), None)
    if base is not None and base.n_edits != 0:
        problems.append("base variant edits the input")

    # --- per-variant support accounting
    for variant in anchor.variants:
        candidates = {norm(name) for name in parse_candidates(variant.entities_str)}
        norm_text = norm(variant.text)
        # An edit that retroactively invalidates already-emitted output would
        # change what the prefix MEANS, so the margin would no longer be
        # comparable across variants.  This is the single most important check.
        broken = [index for index, block in enumerate(prefix_blocks)
                  if not _supported(block, candidates, norm_text, norm)]
        if broken:
            problems.append(
                f"{variant.variant_id}: the edit invalidated {len(broken)} "
                f"already-emitted prefix block(s), e.g. #{broken[0]}")
        remaining_ok = sum(
            1 for block in anchor.remainder
            if _supported(block, candidates, norm_text, norm))
        if remaining_ok != variant.n_remaining_supported:
            problems.append(
                f"{variant.variant_id}: declares {variant.n_remaining_supported} "
                f"supported remaining blocks but has {remaining_ok}")
        if variant.arm == "neutral" and anchor.remainder and \
                remaining_ok != len(anchor.remainder):
            problems.append(f"{variant.variant_id}: a neutral edit removed support")
        if variant.arm == "manip" and anchor.remainder:
            expected = len(anchor.remainder) - round(variant.level * len(anchor.remainder))
            if remaining_ok != expected:
                problems.append(
                    f"{variant.variant_id}: level {variant.level} should leave "
                    f"{expected} supported, left {remaining_ok}")

    # --- the edit must not have touched a candidate it did not name
    # Evidence-axis edits are plain string substitutions over the document, so
    # an endpoint that happens to occur inside another candidate would rewrite
    # that one too and silently remove support the level never asked for.
    # `is_independent` is supposed to prevent it; this re-checks the OUTCOME,
    # because a guard that is never verified is a guard that can be regressed.
    if base is not None:
        base_counts = _mention_counts(base.text, parse_candidates(base.entities_str))
        for variant in anchor.variants:
            if variant.arm == "base" or variant.text == base.text:
                continue
            edited = {edit.get("from") for edit in variant.edits}
            after = _mention_counts(variant.text,
                                    parse_candidates(base.entities_str))
            collateral = sorted(
                name for name, count in base_counts.items()
                if name not in edited and after.get(name, 0) != count)
            if collateral:
                problems.append(
                    f"{variant.variant_id}: the edit also changed mentions of "
                    f"{len(collateral)} candidate(s) it did not name, e.g. "
                    f"{collateral[0]!r}")

    # --- neutral must match its manipulated counterpart's edit count
    by_level: Dict[float, Dict[str, Variant]] = {}
    for variant in anchor.variants:
        if variant.arm in ("manip", "neutral"):
            by_level.setdefault(variant.level, {})[variant.arm] = variant
    for level, pair in by_level.items():
        if set(pair) != {"manip", "neutral"}:
            problems.append(f"level {level}: missing "
                            f"{'neutral' if 'neutral' not in pair else 'manip'}")
        elif pair["manip"].n_edits != pair["neutral"].n_edits:
            problems.append(
                f"level {level}: neutral makes {pair['neutral'].n_edits} edits "
                f"but manip makes {pair['manip'].n_edits}")
    return problems
