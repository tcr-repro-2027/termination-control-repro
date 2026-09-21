# coding: utf-8
"""The four training arms that complete the problem-4 2x2.

E1 leaves one cell of a 2x2 approximated rather than measured:

    filter   (P4 present, P5-11 present)   capture 36.17%
    clean    (P4 removed, P5-11 present)            7.18%
    cleanv2  (P4 removed, P5-11 removed)            5.52%
    cleanv2 + OBR (P4 REINSERTED at 24.3%)          8.40%

Each defect alone is worth about +3 points over cleanv2; together they are worth
+31.  Either that 6.7x is a real interaction, or OBR's reinsertion is not a
faithful stand-in for a corpus that simply never had problem 4 cleaned.  Only
the latter can be settled cheaply, and only by building that corpus.

    keep4     filter with problems 5-11 cleaned and problem 4 LEFT ALONE.
              Pairs with cleanv2: the two differ in problem 4 and nothing else.

    randdrop  filter with the same NUMBER of blocks removed per record as the
              problem-4 cleaning removes, chosen at random from all of that
              record's blocks, and nothing else done.  Pairs with clean: same
              records, same per-record block counts, same amount removed, so
              the only difference is WHICH blocks went.  Without it, "capture
              fell from 36% to 7%" has a second reading -- the targets also got
              24% shorter -- and no way to tell the two apart.

    keep4_a   keep4 with only the problem-4 blocks whose violation is purely
    keep4_ae  admissibility (endpoint absent from the candidate list), and
              keep4 with only those violating BOTH axes.  Worth training only
              if keep4 itself comes out positive.

The axis pair is A vs AE and not A vs E for a measured reason: on this corpus
there are **no pure evidence violations at all** -- 156,383 blocks of kind A,
47,660 of kind AE, zero of kind E.  Every candidate entity occurs in its own
document, so "listed but not in the text" cannot arise, and a `keep4_e` arm
comes out byte-identical to `cleanv2`.  (The same degeneracy is already on
record for the eval split, where it is why E1 could not test the evidence axis
at all.)  So the only evidence contrast this data supports is "admissibility
alone" against "admissibility and evidence together", which is what A vs AE is.
`build_p4_arms.py` refuses to write an arm that reproduces an existing stage
file, so a degenerate arm is reported rather than trained.

`randdrop` is deliberately NOT given the problems 5-11 pass: its comparison
partner is `clean`, which still carries them.  Cleaning it further would make
the pair incomparable.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from tcr.data.stage_build.stages import (           # noqa: E402
    Record, StageStats, canonical_block, parse_entities, read_jsonl,
)

from .rules import P4_KINDS, apply_later_problems, label_problem4  # noqa: E402

__all__ = ["ARMS", "AXIS_ARMS", "QUEUE_ARMS", "load_stage", "build_keep4", "build_randdrop", "build_axis_arm",
           "removal_counts", "BUILD_SEED"]

BUILD_SEED = 20260901

#: name -> the file stem its training data is written under.
ARMS = {
    "keep4": "train_keep4.jsonl",
    "randdrop": "train_randdrop.jsonl",
    "keep4_a": "train_keep4_a.jsonl",
    "keep4_ae": "train_keep4_ae.jsonl",
    # buildable on request, but degenerate on this corpus (no pure evidence
    # violations exist) -- the build refuses to write it and says why.
    "keep4_e": "train_keep4_e.jsonl",
}

#: The arms meant for the training queue.  `keep4_e` is excluded: asking for it
#: is a diagnostic, not a run.
QUEUE_ARMS = ("keep4", "randdrop", "keep4_a", "keep4_ae")

#: arm -> which problem-4 kinds it keeps.  "keep4" keeps them all.
AXIS_ARMS = {"keep4_a": ("A",), "keep4_ae": ("AE",), "keep4_e": ("E",)}


# --------------------------------------------------------------------- input

def load_stage(data_path: Path, split: str) -> list[Record]:
    """Rebuild `Record`s from a frozen stage file plus its row map.

    The row map carries `rec_id`, which is the identity every diff in this
    project pairs on; it is never recomputed from content.  `filter_index` is
    the block's position within THIS file, which is what `filter` positions
    mean for every stage derived from it.
    """
    rowmap_path = data_path.with_name(data_path.stem + "_rowmap.jsonl")
    rowmap = list(read_jsonl(rowmap_path)) if rowmap_path.is_file() else []
    records: list[Record] = []
    for row_index, row in enumerate(read_jsonl(data_path)):
        entities, method = parse_entities(row.get("entities_str"))
        if entities is None:
            raise ValueError(f"{data_path}:{row_index}: entities_str "
                             f"unparseable ({method}); the stage files are "
                             "supposed to be past problem 2")
        blocks: list[dict[str, str]] = []
        for item in row.get("output") or []:
            block, _aliases, problem = canonical_block(item)
            if block is None:
                raise ValueError(f"{data_path}:{row_index}: unusable block "
                                 f"({problem}); stage files are past problem 1")
            blocks.append(block)
        extra = {k: v for k, v in row.items()
                 if k not in ("text", "entities_str", "output")}
        rec_id = (rowmap[row_index]["rec_id"] if row_index < len(rowmap)
                  else f"{split}:row_{row_index + 1:06d}")
        records.append(Record(
            rec_id=rec_id, split=split, row_index=row_index, extra=extra,
            text=row.get("text", ""), entities=entities,
            entities_str=row.get("entities_str", ""), blocks=blocks,
            filter_index=list(range(len(blocks))),
        ))
    return records


def removal_counts(filter_records: Sequence[Record],
                   clean_path: Path) -> dict[str, int]:
    """How many blocks the problem-4 cleaning removed, per record.

    Read off the FROZEN pair of files rather than recomputed: `randdrop` has to
    remove exactly as many blocks as `clean` did, and the authority on that is
    the file `clean` actually is.  A record `clean` dropped entirely counts as
    all of its blocks.
    """
    rowmap_path = clean_path.with_name(clean_path.stem + "_rowmap.jsonl")
    kept = {row["rec_id"]: int(row["blocks"]) for row in read_jsonl(rowmap_path)}
    return {record.rec_id: len(record.blocks) - kept.get(record.rec_id, 0)
            for record in filter_records}


# --------------------------------------------------------------------- arms

def build_keep4(filter_records: Sequence[Record], stats: StageStats,
                protected_descriptions: set[str]) -> list[Record]:
    """Problems 5-11 on a corpus that still has problem 4."""
    return apply_later_problems(filter_records, stats, protected_descriptions)


def build_axis_arm(filter_records: Sequence[Record], stats: StageStats,
                   protected_descriptions: set[str], *,
                   keep_kinds: Sequence[str],
                   axis_stats: dict[str, int]) -> list[Record]:
    """keep4 restricted to some kinds of problem-4 violation.

    Every problem-4 block of another kind is removed before the problems 5-11
    pass, so the arm carries exactly the violation it names.  The per-kind
    counts go into `axis_stats`, which is how the absence of pure evidence
    violations became visible in the first place.
    """
    unknown = set(keep_kinds) - set(P4_KINDS)
    if unknown:
        raise ValueError(f"unknown problem-4 kind(s) {sorted(unknown)}")
    trimmed: list[Record] = []
    for record in filter_records:
        kinds = label_problem4(record)
        blocks, index = [], []
        for position, (block, kind) in enumerate(zip(record.blocks, kinds)):
            axis_stats[f"seen_{kind}"] += 1
            if kind == "Valid" or kind in keep_kinds:
                blocks.append(block)
                index.append(record.filter_index[position])
            else:
                axis_stats[f"dropped_{kind}"] += 1
        if not blocks:
            axis_stats["records_emptied"] += 1
            continue
        record.blocks = blocks
        record.filter_index = index
        trimmed.append(record)
    return apply_later_problems(trimmed, stats, protected_descriptions)


def build_randdrop(filter_records: Sequence[Record], counts: dict[str, int],
                   stats: StageStats, *, seed: int = BUILD_SEED) -> list[Record]:
    """`filter` with a matched-size RANDOM removal instead of the problem-4 one.

    Per record, `counts[rec_id]` blocks are drawn uniformly without replacement
    from ALL of that record's blocks -- not only from the good ones.  Drawing
    from the good ones alone would raise the problem-4 share instead of leaving
    it alone, and the arm is meant to hold composition fixed while matching the
    amount.

    The RNG is seeded per record from its `rec_id`, so the draw does not depend
    on iteration order, on how many records precede it, or on the arm being
    built alone or in a batch.
    """
    stats.records_in = len(filter_records)
    stats.blocks_in = sum(len(r.blocks) for r in filter_records)
    kept: list[Record] = []
    for record in filter_records:
        wanted = counts.get(record.rec_id, 0)
        if wanted <= 0:
            kept.append(record)
            continue
        if wanted >= len(record.blocks):
            stats.record_drops["no_blocks_after_randdrop"] += 1
            stats.block_drops["random_matched_drop"] += len(record.blocks)
            continue
        digest = hashlib.sha256(f"{seed}|{record.rec_id}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        drop = set(rng.sample(range(len(record.blocks)), wanted))
        stats.block_drops["random_matched_drop"] += wanted
        survivors = [(block, record.filter_index[i])
                     for i, block in enumerate(record.blocks) if i not in drop]
        record.blocks = [item[0] for item in survivors]
        record.filter_index = [item[1] for item in survivors]
        kept.append(record)
    stats.records_out = len(kept)
    stats.blocks_out = sum(len(r.blocks) for r in kept)
    return kept
