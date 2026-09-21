# coding: utf-8
"""Problem-4 labelling and the problems 5-11 pass, reusable off the clean chain.

Why this exists as new code instead of a flag on `tcr/data/stage_build`
----------------------------------------------------------------------
The frozen pipeline runs `filter -> clean -> cleanv2` in that order, and
`stage_cleanv2` contains one line that is only correct BECAUSE problem 4 already
ran:

    # problem 10 fallout: an endpoint whose candidate entry is gone.
    if block["source"] not in entity_set or block["target"] not in entity_set:
        stats.block_drops["problem10_endpoint_entity_removed"] += 1

`entity_set` is the candidate list.  On the normal chain every out-of-candidate
endpoint is already gone, so this only catches blocks whose endpoint was removed
by problem 10 in that same step -- which is what the comment says it does.  Run
it on `filter`, where problem 4 has NOT run, and the same line silently deletes
every out-of-candidate block and files it under `problem10`.  A naive "skip the
clean stage" therefore produces something close to `clean`, not to `keep4`, and
nothing in the logs says so.

So the ORCHESTRATION is rewritten here with that one predicate corrected, while
every PER-BLOCK RULE is imported from the frozen module rather than restated:
`relation_leak`, `is_low_quality_entity`, `block_quality`, `joint_type`, `norm`,
`contains`.  Those are the parts that must not drift.

The rewrite is not trusted either.  :func:`apply_later_problems` run on the
frozen `clean` records must reproduce the frozen `cleanv2` file byte for byte;
`experiments/1_data/build_p4_arms.py` checks that before it builds anything, and refuses to
continue otherwise.  If that holds, the corrected fallout is a strict
generalisation of the original and the new arms are the frozen cleaning applied
to a corpus that still carries problem 4.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from tcr.data.stage_build.stages import (           # noqa: E402  (frozen rules, reused)
    Record, StageStats, block_quality, contains, is_low_quality_entity,
    joint_type, norm, relation_leak,
)

__all__ = ["label_problem4", "apply_later_problems", "P4_KINDS"]

#: `stage_clean`'s own vocabulary: A = the endpoint is not in the candidate list
#: (admissibility), E = it is not in the text (evidence), AE = both.
P4_KINDS = ("A", "E", "AE")


def label_problem4(record: Record) -> list[str]:
    """Per block: "Valid" | "A" | "E" | "AE", exactly as `stage_clean` decides.

    Reproduced rather than imported because `stage_clean` mutates the record it
    is given and returns only the survivors; the arms here need the label on
    every block, including the ones it would drop.  The decision itself is
    `joint_type` on the same four booleans, so there is one place where the rule
    lives and this is not it -- `experiments/1_data/build_p4_arms.py` checks that keeping
    only the "Valid" blocks reproduces the frozen `clean` file.
    """
    entity_set = set(record.entities)
    kinds: list[str] = []
    for block in record.blocks:
        kinds.append(joint_type(
            block["source"] in entity_set,
            block["target"] in entity_set,
            contains(record.text, block["source"]),
            contains(record.text, block["target"]),
        ))
    return kinds


def apply_later_problems(records: Sequence[Record], stats: StageStats,
                         protected_descriptions: set[str]) -> list[Record]:
    """Problems 5, 6, 7, 9, 10, 11.  Problem 8 (nesting) is kept, as in cleanv2.

    Identical to `stage_cleanv2` except for the problem-10 fallout, which asks
    the question the comment asks: was this endpoint's candidate entry removed
    *by problem 10, here*?  The original asks "is this endpoint in the candidate
    list at all", which is the same question only after problem 4 has run.
    """
    stats.records_in = len(records)
    stats.blocks_in = sum(len(r.blocks) for r in records)
    kept: list[Record] = []
    for record in records:
        # problem 10: drop low-quality entities from the candidate list.
        before = set(record.entities)
        entities: list[str] = []
        for entity in record.entities:
            if is_low_quality_entity(entity):
                stats.repairs["low_quality_entity_removed"] += 1
                continue
            entities.append(entity)
        entity_set = set(entities)
        # THE ONE CHANGED LINE'S INPUT: endpoints this step just removed, not
        # every endpoint that happens to be absent from the list.
        removed_by_problem10 = before - entity_set

        survivors: list[dict[str, str]] = []
        survivor_index: list[int] = []
        for index, block in enumerate(record.blocks):
            block = dict(block)
            # problem 7: relation/description leakage.
            reason, swapped = relation_leak(block)
            if swapped:
                block["relation"], block["description"] = (
                    block["description"], block["relation"])
                stats.repairs["relation_description_swapped"] += 1
                reason, swapped = relation_leak(block)
            if reason:
                stats.block_drops[f"problem7_{reason}"] += 1
                continue
            # problem 10 fallout, corrected.
            if (block["source"] in removed_by_problem10
                    or block["target"] in removed_by_problem10):
                stats.block_drops["problem10_endpoint_entity_removed"] += 1
                continue
            # problem 5: self loop.
            if norm(block["source"]) == norm(block["target"]):
                stats.block_drops["problem5_self_loop"] += 1
                continue
            # problem 9: description grounded in neither endpoint.
            if (not contains(block["description"], block["source"])
                    and not contains(block["description"], block["target"])):
                stats.block_drops["problem9_weak_description"] += 1
                continue
            # problem 11: train description reused from eval.
            if norm(block["description"]) in protected_descriptions:
                stats.block_drops["problem11_cross_split_description"] += 1
                continue
            survivors.append(block)
            survivor_index.append(record.filter_index[index])

        # problem 6: reverse-direction duplicates.
        groups: dict[frozenset[str], list[int]] = defaultdict(list)
        for index, block in enumerate(survivors):
            groups[frozenset((norm(block["source"]), norm(block["target"])))].append(index)
        keep_indices: set[int] = set()
        for indices in groups.values():
            keep_indices.add(max(indices, key=lambda i: block_quality(survivors[i], i)))
            if len(indices) > 1:
                stats.block_drops["problem6_reverse_duplicate"] += len(indices) - 1

        # problem 11: duplicate description inside one record.
        by_description: dict[str, list[int]] = defaultdict(list)
        for index in sorted(keep_indices):
            by_description[norm(survivors[index]["description"])].append(index)
        final_indices: set[int] = set()
        for indices in by_description.values():
            final_indices.add(max(indices, key=lambda i: block_quality(survivors[i], i)))
            if len(indices) > 1:
                stats.block_drops["problem11_duplicate_description"] += len(indices) - 1

        final = [(survivors[i], survivor_index[i]) for i in sorted(final_indices)]
        if not final:
            stats.record_drops["no_blocks_after_cleanv2"] += 1
            continue
        record.entities = entities
        record.entities_str = repr(entities)
        record.blocks = [item[0] for item in final]
        record.filter_index = [item[1] for item in final]
        kept.append(record)
    stats.records_out = len(kept)
    stats.blocks_out = sum(len(r.blocks) for r in kept)
    return kept
