# coding: utf-8
"""What each arm must be, on data whose answer is known by construction.

The reproductions in `build_p4_arms.py` prove the rules did not drift from the
frozen chain; these prove the arms are what their names say on inputs small
enough to check by hand.  The one that matters most is
`test_keep4_keeps_out_of_candidate_blocks`: a naive skip of the clean stage
deletes exactly those blocks under the name `problem10`, and every other
property of the resulting file looks normal.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcr.data.stage_build.stages import Record, StageStats               # noqa: E402
from tcr.data.raw_arms.arms import build_axis_arm, build_keep4, build_randdrop  # noqa: E402
from tcr.data.raw_arms.rules import apply_later_problems, label_problem4      # noqa: E402


def record(rec_id: str, entities, blocks, text=None) -> Record:
    """A record whose text mentions every entity unless told otherwise."""
    text = text if text is not None else "。".join(entities) + "。"
    return Record(rec_id=rec_id, split="train", row_index=0, extra={},
                  text=text, entities=list(entities),
                  entities_str=repr(list(entities)), blocks=[dict(b) for b in blocks],
                  filter_index=list(range(len(blocks))))


def block(source, target, relation="关系", description=None):
    return {"source": source, "target": target, "relation": relation,
            "description": description or f"{source}与{target}之间的说明。"}


# ------------------------------------------------------- problem-4 labelling

def test_problem4_kinds():
    """A = not in the candidate list, E = not in the text, AE = both."""
    rec = record("r1", ["甲公司", "乙机构", "丙中心"],
                 [block("甲公司", "乙机构"),          # both listed and in text -> Valid
                  block("甲公司", "丁单位"),          # 丁 not listed, not in text -> AE
                  block("甲公司", "丙中心")],
                 text="甲公司。乙机构。丙中心。丁单位。")     # 丁 IS in the text, not listed -> A
    kinds = label_problem4(rec)
    assert kinds[0] == "Valid"
    assert kinds[1] == "A"          # in text, absent from the list
    assert kinds[2] == "Valid"

    hidden = record("r2", ["甲公司", "乙机构"], [block("甲公司", "乙机构")], text="甲公司。")
    assert label_problem4(hidden)[0] == "E"     # listed, absent from the text


# ------------------------------------------------------------------- keep4

def test_keep4_keeps_out_of_candidate_blocks():
    """THE regression.  `stage_cleanv2`'s problem-10 fallout asks "is this
    endpoint in the candidate list", which on an uncleaned corpus deletes every
    problem-4 block and files it under problem 10.  Here it must survive."""
    rec = record("r1", ["甲公司", "乙机构"],
                 [block("甲公司", "乙机构"), block("甲公司", "丁单位")],
                 text="甲公司。乙机构。丁单位。")
    stats = StageStats()
    kept = build_keep4([rec], stats, set())
    assert len(kept) == 1
    assert len(kept[0].blocks) == 2, "the out-of-candidate block was deleted"
    assert stats.block_drops.get("problem10_endpoint_entity_removed", 0) == 0


def test_problem10_fallout_still_fires_for_its_own_removals():
    """The corrected predicate is narrower, not absent: an endpoint whose
    candidate entry problem 10 removes must still take its block with it."""
    rec = record("r1", ["甲公司", "乙机构", "丙中心"],
                 [block("甲公司", "乙机构"), block("甲公司", "丙中心")])
    rec.entities = ["甲公司", "乙机构", "丙中心"]
    stats = StageStats()
    # a single character is a low-quality entity, so problem 10 removes it
    rec = record("r2", ["甲公司", "乙机构", "X"],
                 [block("甲公司", "乙机构"), block("甲公司", "X")],
                 text="甲公司。乙机构。X。")
    kept = build_keep4([rec], stats, set())
    assert len(kept[0].blocks) == 1
    assert stats.block_drops["problem10_endpoint_entity_removed"] == 1


def test_keep4_still_cleans_problems_5_to_11():
    rec = record("r1", ["甲公司", "乙机构", "丙中心"],
                 [block("甲公司", "甲公司"),                       # problem 5, self loop
                  block("甲公司", "乙机构"),
                  block("乙机构", "甲公司"),                       # problem 6, reversed
                  block("甲公司", "丙中心", description="毫无关联的文字")])  # problem 9
    stats = StageStats()
    kept = build_keep4([rec], stats, set())
    assert stats.block_drops["problem5_self_loop"] == 1
    assert stats.block_drops["problem9_weak_description"] == 1
    assert stats.block_drops["problem6_reverse_duplicate"] == 1
    assert len(kept[0].blocks) == 1


# ---------------------------------------------------------------- randdrop

def test_randdrop_removes_exactly_the_matched_count():
    rec = record("r1", [f"e{i}" for i in range(8)],
                 [block(f"e{i}", f"e{i + 1}") for i in range(7)])
    stats = StageStats()
    kept = build_randdrop([rec], {"r1": 3}, stats)
    assert len(kept[0].blocks) == 4
    assert stats.block_drops["random_matched_drop"] == 3


def test_randdrop_is_reproducible_and_order_independent():
    """Seeded per rec_id, so building one arm alone and building it inside a
    batch give the same file."""
    def make():
        return [record("a", ["x", "y", "z"],
                       [block("x", "y"), block("y", "z"), block("x", "z")]),
                record("b", ["p", "q", "r"],
                       [block("p", "q"), block("q", "r"), block("p", "r")])]
    counts = {"a": 1, "b": 1}
    first = build_randdrop(make(), counts, StageStats())
    second = build_randdrop(list(reversed(make())), counts, StageStats())
    by_id = {r.rec_id: r.blocks for r in second}
    for rec in first:
        assert rec.blocks == by_id[rec.rec_id]


def test_randdrop_draws_from_all_blocks_not_only_the_good_ones():
    """Drawing only from the valid blocks would RAISE the problem-4 share; the
    arm exists to hold composition fixed while matching the amount."""
    entities = [f"e{i}" for i in range(6)]
    blocks = ([block(f"e{i}", f"e{i + 1}") for i in range(5)]
              + [block("e0", "外来实体")])          # one problem-4 block
    dropped_p4 = 0
    for trial in range(40):
        rec = record(f"r{trial}", entities, blocks, text="。".join(entities + ["外来实体"]))
        kept = build_randdrop([rec], {f"r{trial}": 3}, StageStats())
        if not any(b["target"] == "外来实体" for b in kept[0].blocks):
            dropped_p4 += 1
    assert dropped_p4 > 0, "the problem-4 block was never eligible for removal"


def test_a_record_losing_every_block_is_dropped():
    rec = record("r1", ["甲公司", "乙机构"], [block("甲公司", "乙机构")])
    stats = StageStats()
    assert build_randdrop([rec], {"r1": 1}, stats) == []
    assert stats.record_drops["no_blocks_after_randdrop"] == 1


# --------------------------------------------------------------- axis arms

def test_axis_arms_carry_exactly_one_kind():
    rec = record("r1", ["甲公司", "乙机构"],
                 [block("甲公司", "乙机构"),          # Valid
                  block("甲公司", "丁单位"),          # A: 丁 in text, not listed
                  block("甲公司", "戊部门")],         # AE: 戊 in neither
                 text="甲公司。乙机构。丁单位。")
    assert label_problem4(rec) == ["Valid", "A", "AE"]

    axis: Counter = Counter()
    only_a = build_axis_arm([record("r1", ["甲公司", "乙机构"],
                                    [block("甲公司", "乙机构"), block("甲公司", "丁单位"),
                                     block("甲公司", "戊部门")], text="甲公司。乙机构。丁单位。")],
                            StageStats(), set(), keep_kinds=("A",), axis_stats=axis)
    assert label_problem4(only_a[0]) == ["Valid", "A"]
    assert axis["dropped_AE"] == 1

    axis = Counter()
    only_ae = build_axis_arm([record("r1", ["甲公司", "乙机构"],
                                     [block("甲公司", "乙机构"), block("甲公司", "丁单位"),
                                      block("甲公司", "戊部门")], text="甲公司。乙机构。丁单位。")],
                             StageStats(), set(), keep_kinds=("AE",), axis_stats=axis)
    assert label_problem4(only_ae[0]) == ["Valid", "AE"]
    assert axis["dropped_A"] == 1


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        build_axis_arm([], StageStats(), set(), keep_kinds=("Z",),
                       axis_stats=Counter())


# ------------------------------------------------- the pass itself is frozen

def test_later_problems_leaves_a_clean_record_alone():
    rec = record("r1", ["甲公司", "乙机构", "丙中心"],
                 [block("甲公司", "乙机构"), block("甲公司", "丙中心")])
    stats = StageStats()
    kept = apply_later_problems([rec], stats, set())
    assert len(kept[0].blocks) == 2
    assert not stats.block_drops


def test_problem11_protects_eval_descriptions():
    rec = record("r1", ["甲公司", "乙机构"], [block("甲公司", "乙机构", description="甲公司与乙机构的说明。")])
    stats = StageStats()
    assert apply_later_problems([rec], stats, {"甲公司与乙机构的说明。"}) == []
    assert stats.block_drops["problem11_cross_split_description"] == 1
