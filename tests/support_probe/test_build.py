# coding=utf-8
"""The anchor builder and the invariants that decide whether an anchor ships.

The invariants are the code form of §E2's instrument criterion 4.  Each test
below breaks exactly one of them, because an anchor that violates one still
produces a perfectly ordinary-looking number.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict

import pytest

from tcr.evaluation.quality import candidate_entities, norm_field
from tcr.support_probe.anchors import assistant_prefix, serialize_blocks, validate_anchor
from tcr.support_probe.build import (
    DonorPool, Record, build_preterminal_anchor, build_zero_answer_anchor,
    cover_exactly, is_independent, n_removed, render_entities,
)
from tcr.support_probe.build import main as build_main
from tcr.support_probe.io_utils import sha256_file

VALIDATE = dict(norm=norm_field, parse_candidates=candidate_entities)


def block(source: str, target: str, relation: str = "关系") -> Dict[str, str]:
    return {"source": source, "target": target, "relation": relation,
            "description": f"{source}与{target}之间的说明文字。"}


def make_record(*, n_blocks: int = 5, extra_candidates: int = 8) -> Record:
    """A record whose remainder is editable: the last blocks use endpoints the
    prefix never touches."""
    blocks = [block(f"甲{i}", f"乙{i}") for i in range(n_blocks)]
    entities = [f"甲{i}" for i in range(n_blocks)] + [f"乙{i}" for i in range(n_blocks)]
    entities += [f"闲{i}" for i in range(extra_candidates)]
    text = "。".join(entities) + "。"
    return Record({"key": 1, "source": "doc", "text": text,
                   "entities_str": render_entities(entities), "output": blocks})


@pytest.fixture
def pool() -> DonorPool:
    donors = [f"外来实体{i:03d}" for i in range(200)]
    donors += [f"短{i}" for i in range(200)]
    return DonorPool(donors, len)


# ------------------------------------------------------------------ helpers

def test_n_removed_is_stable_across_python_versions():
    """`round` is banker's rounding: round(1.5) == 2 but round(2.5) == 2.  The
    level -> count map must not depend on that."""
    assert n_removed(0.5, 2) == 1
    assert n_removed(0.5, 3) == 2
    assert n_removed(0.5, 4) == 2
    assert n_removed(1.0, 3) == 3
    assert n_removed(0.0, 4) == 0


def test_is_independent_rejects_nesting():
    candidates = ["需求管理", "需求管理流程", "评审"]
    assert not is_independent("需求管理", candidates)
    assert not is_independent("需求管理流程", candidates)
    assert is_independent("评审", candidates)


def test_cover_exactly_refuses_collateral_damage():
    """An endpoint shared with a block outside the target would strip support
    the level did not ask for, so it must not be selected."""
    kills = {"a": [0], "b": [0, 1], "c": [1]}
    assert cover_exactly(kills, frozenset({0})) == ["a"]
    assert cover_exactly(kills, frozenset({0, 1})) in (["b"], ["a", "c"])
    assert cover_exactly({"b": [0, 1]}, frozenset({0})) is None


def test_serialisation_matches_the_training_target():
    """The prefix must be a genuine prefix of a training target, not a
    lookalike: `json.dumps` on the list and the join must agree."""
    blocks = [block("甲", "乙"), block("丙", "丁")]
    assert serialize_blocks(blocks) == json.dumps(blocks, ensure_ascii=False)
    assert assistant_prefix(blocks) == serialize_blocks(blocks)[:-1]
    assert assistant_prefix(blocks).endswith("}")
    assert assistant_prefix([]) == "["


# ------------------------------------------------------------- the builder

@pytest.mark.parametrize("axis", ["evidence", "admissibility"])
def test_builder_produces_a_valid_anchor(pool, axis):
    anchor = build_preterminal_anchor(make_record(), axis, 2, pool)
    assert anchor is not None
    assert validate_anchor(anchor, **VALIDATE) == []
    assert anchor.anchor_type == axis
    assert anchor.prefix_blocks == 3 and len(anchor.remainder) == 2
    assert anchor.assistant_prefix.endswith("}")

    arms = [variant.arm for variant in anchor.variants]
    assert arms.count("base") == 1
    assert arms.count("manip") == 2 and arms.count("neutral") == 2
    for name in ("pc_stop", "pc_continue", "pc_neutral"):
        assert name in arms


@pytest.mark.parametrize("axis", ["evidence", "admissibility"])
def test_levels_remove_exactly_what_they_promise(pool, axis):
    anchor = build_preterminal_anchor(make_record(), axis, 2, pool)
    by_key = {(v.arm, v.level): v for v in anchor.variants}
    assert by_key[("base", 0.0)].n_remaining_supported == 2
    assert by_key[("manip", 0.5)].n_remaining_supported == 1
    assert by_key[("manip", 1.0)].n_remaining_supported == 0
    # the neutral edit is the same SIZE and removes nothing
    for level in (0.5, 1.0):
        assert by_key[("neutral", level)].n_remaining_supported == 2
        assert (by_key[("neutral", level)].n_edits
                == by_key[("manip", level)].n_edits)


def test_evidence_edits_the_text_and_leaves_the_candidate_list_alone(pool):
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    base = next(v for v in anchor.variants if v.arm == "base")
    manip = next(v for v in anchor.variants if v.arm == "manip" and v.level == 1.0)
    assert manip.text != base.text
    assert manip.entities_str == base.entities_str
    # the endpoint is gone from the document but still listed: a PURE evidence
    # violation, which is what E1 could not measure on E-Natural
    gone = manip.edits[0]["from"]
    assert gone not in manip.text
    assert gone in candidate_entities(manip.entities_str)


def test_admissibility_edits_the_list_and_leaves_the_text_alone(pool):
    anchor = build_preterminal_anchor(make_record(), "admissibility", 2, pool)
    base = next(v for v in anchor.variants if v.arm == "base")
    manip = next(v for v in anchor.variants if v.arm == "manip" and v.level == 1.0)
    assert manip.text == base.text
    assert manip.entities_str != base.entities_str
    gone = manip.edits[0]["from"]
    assert gone not in candidate_entities(manip.entities_str)
    assert gone in manip.text                     # evidence untouched
    # the candidate list keeps its length: a shorter list is a second, silent
    # manipulation
    assert len(candidate_entities(manip.entities_str)) == \
        len(candidate_entities(base.entities_str))


def test_pc_continue_names_a_relation_from_the_audited_remainder(pool):
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    variant = next(v for v in anchor.variants if v.arm == "pc_continue")
    first = anchor.remainder[0]
    assert first["source"] in variant.instruction
    assert first["target"] in variant.instruction


def test_controls_do_not_touch_the_input(pool):
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    base = next(v for v in anchor.variants if v.arm == "base")
    for variant in anchor.variants:
        if variant.arm.startswith("pc_"):
            assert variant.text == base.text
            assert variant.entities_str == base.entities_str
            assert variant.n_edits == 0
            assert variant.instruction


def test_zero_answer_anchor_has_no_supported_pair(pool):
    host = make_record()
    donor = Record({"key": 2, "source": "other", "text": "无关文本。",
                    "entities_str": render_entities(
                        [f"异域实体{i}" for i in range(20)]),
                    "output": []})
    anchor = build_zero_answer_anchor(host, donor)
    assert anchor is not None
    assert validate_anchor(anchor, **VALIDATE) == []
    assert anchor.assistant_prefix == "["
    assert anchor.continue_continuation == '{"'
    base = next(v for v in anchor.variants if v.arm == "base")
    norm_text = norm_field(base.text)
    # no listed candidate occurs in the document, so no pair can be evidenced
    assert all(norm_field(name) not in norm_text
               for name in candidate_entities(base.entities_str))


def test_builder_refuses_when_the_remainder_shares_endpoints_with_the_prefix(pool):
    """Editing an endpoint the prefix used would retroactively invalidate
    already-emitted output, so such a record must not become an anchor."""
    # the last two blocks recombine endpoints the first two already emitted, so
    # nothing in the remainder can be edited without touching the prefix
    blocks = [block("甲", "乙"), block("丙", "丁"),
              block("甲", "丁"), block("丙", "乙")]
    entities = ["甲", "乙", "丙", "丁"] + [f"闲{i}" for i in range(6)]
    record = Record({"key": 3, "source": "doc",
                     "text": "。".join(entities) + "。",
                     "entities_str": render_entities(entities), "output": blocks})
    assert build_preterminal_anchor(record, "evidence", 2, pool) is None
    assert build_preterminal_anchor(record, "admissibility", 2, pool) is None


# --------------------------------------------------------------- invariants

def test_validator_catches_an_edit_that_invalidates_the_prefix(pool):
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    broken = list(anchor.variants)
    victim = broken[1]
    # remove an entity the PREFIX depends on
    broken[1] = dataclasses.replace(victim, text=victim.text.replace("甲0", "陌生"))
    tampered = dataclasses.replace(anchor, variants=tuple(broken))
    problems = validate_anchor(tampered, **VALIDATE)
    assert any("already-emitted prefix block" in problem for problem in problems)


def test_validator_catches_a_neutral_edit_that_removes_support(pool):
    anchor = build_preterminal_anchor(make_record(), "admissibility", 2, pool)
    variants = list(anchor.variants)
    index = next(i for i, v in enumerate(variants)
                 if v.arm == "neutral" and v.level == 1.0)
    manip = next(v for v in variants if v.arm == "manip" and v.level == 1.0)
    variants[index] = dataclasses.replace(
        variants[index], entities_str=manip.entities_str)
    problems = validate_anchor(dataclasses.replace(
        anchor, variants=tuple(variants)), **VALIDATE)
    assert any("neutral edit removed support" in problem for problem in problems)


def test_validator_catches_a_mismatched_edit_count(pool):
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    variants = list(anchor.variants)
    index = next(i for i, v in enumerate(variants)
                 if v.arm == "neutral" and v.level == 1.0)
    variants[index] = dataclasses.replace(variants[index], n_edits=99)
    problems = validate_anchor(dataclasses.replace(
        anchor, variants=tuple(variants)), **VALIDATE)
    assert any("neutral makes 99 edits" in problem for problem in problems)


def test_validator_catches_a_missing_positive_control(pool):
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    kept = tuple(v for v in anchor.variants if v.arm != "pc_stop")
    problems = validate_anchor(dataclasses.replace(anchor, variants=kept),
                               **VALIDATE)
    assert any("pc_stop" in problem for problem in problems)


def test_validator_catches_a_prefix_that_is_not_at_a_block_boundary(pool):
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    tampered = dataclasses.replace(
        anchor, assistant_prefix=anchor.assistant_prefix[:-8])
    problems = validate_anchor(tampered, **VALIDATE)
    assert problems      # either the boundary or the JSON parse check fires


# --------------------------------------------- shipping an anchor set at all

def eval_file(tmp_path, n_records: int = 8):
    """A tiny but genuinely buildable eval set."""
    rows = []
    for index in range(n_records):
        blocks = [block(f"甲{index}_{i}", f"乙{index}_{i}") for i in range(5)]
        entities = ([f"甲{index}_{i}" for i in range(5)]
                    + [f"乙{index}_{i}" for i in range(5)]
                    + [f"闲{index}_{i}" for i in range(10)])
        rows.append({"key": index, "source": f"doc{index}",
                     "text": "。".join(entities) + "。",
                     "entities_str": render_entities(entities),
                     "output": blocks})
    path = tmp_path / "eval.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
                    + "\n", encoding="utf-8")
    return path


def run_build(tmp_path, *, n_per_type: int, n_records: int = 8):
    out = tmp_path / "anchors" / "anchors.jsonl"
    code = build_main(["--eval_data", str(eval_file(tmp_path, n_records)),
                       "--out", str(out), "--n_per_type", str(n_per_type)])
    return code, out


def test_an_under_filled_anchor_set_is_not_shipped(tmp_path):
    """The failure this prevents: fewer anchors only widen the CI, so an axis
    measured on a tenth of the intended sample -- or not at all -- produces a
    summary that looks entirely normal."""
    code, out = run_build(tmp_path, n_per_type=500)
    assert code == 1
    assert not out.exists()
    assert not out.with_suffix(out.suffix + ".tmp").exists()
    report = out.with_suffix(out.suffix + ".report.json")
    assert report.with_suffix(".failed.json").is_file()   # evidence, not silence
    assert not report.is_file()


def test_a_complete_build_ships_and_its_report_describes_it(tmp_path):
    code, out = run_build(tmp_path, n_per_type=4)
    assert code == 0
    report_path = out.with_suffix(out.suffix + ".report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["anchors_sha256"] == sha256_file(out)
    assert report["n_rejected"] == 0
    for name in ("evidence", "admissibility", "zero_answer"):
        assert report["anchors_by_type"][name] >= 4
    assert not out.with_suffix(out.suffix + ".tmp").exists()


def test_the_anchor_file_is_never_left_half_written(tmp_path):
    """Construction is atomic, so a killed build cannot leave a short file that
    the launcher then skips construction for."""
    code, out = run_build(tmp_path, n_per_type=4)
    assert code == 0
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert all(json.loads(line)["anchor_id"] for line in lines)


def test_every_edit_records_how_much_text_it_touched(pool):
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    manip = next(v for v in anchor.variants if v.arm == "manip" and v.level == 1.0)
    assert all(edit["n_occurrences"] >= 1 for edit in manip.edits)


def test_an_edit_that_rewrites_another_candidate_is_rejected(pool):
    """`is_independent` is meant to prevent it; the validator re-checks the
    OUTCOME, because an unverified guard is one that can be regressed away."""
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    base = next(v for v in anchor.variants if v.arm == "base")
    victim = next(name for name in candidate_entities(base.entities_str)
                  if name not in {e["from"] for v in anchor.variants
                                  for e in v.edits})
    broken = []
    for variant in anchor.variants:
        if variant.arm == "manip" and variant.level == 1.0:
            variant = dataclasses.replace(
                variant, text=variant.text.replace(victim, "陌生替换"))
        broken.append(variant)
    problems = validate_anchor(dataclasses.replace(anchor, variants=tuple(broken)),
                               **VALIDATE)
    assert any("did not name" in problem for problem in problems)


# ------------------------------------- what the global text substitution risks

def test_mention_context_counts_occurrences_a_longer_word_could_contain():
    """The evidence axis rewrites the document with a plain `.replace()`.  The
    residual risk is an endpoint sitting inside a longer NON-candidate word;
    this quantifies where that is orthographically possible -- punctuation and
    whitespace close a word in every script here, more script characters do
    not."""
    from tcr.support_probe.build import mention_context
    # bounded by punctuation on both sides: no longer word can contain it
    assert mention_context("。甲公司。乙。", "甲公司") == {
        "n_occurrences": 1, "n_occurrences_embedded": 0}
    # followed by another Hanzi: 「甲公司与」 could be one longer unit
    assert mention_context("甲公司与乙签约。", "甲公司") == {
        "n_occurrences": 1, "n_occurrences_embedded": 1}
    # the case the docstring names: nested inside a longer term
    assert mention_context("需求管理流程。", "需求管理") == {
        "n_occurrences": 1, "n_occurrences_embedded": 1}
    # counted per occurrence, not per entity
    assert mention_context("甲公司与乙。甲公司。", "甲公司") == {
        "n_occurrences": 2, "n_occurrences_embedded": 1}
    # latin and digits are word characters too
    assert mention_context("ABCD ABC.", "ABC") == {
        "n_occurrences": 2, "n_occurrences_embedded": 1}


def test_every_evidence_edit_carries_its_mention_risk(pool):
    """Carried as data in the anchor file, not as a paragraph in the README:
    a reader of the artefacts can see how much of the text each edit rewrote
    and how much of that could have been collateral."""
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    for variant in anchor.variants:
        for edit in variant.edits:
            assert edit["n_occurrences"] >= 1
            assert 0 <= edit["n_occurrences_embedded"] <= edit["n_occurrences"]


def test_the_neutral_arm_is_matched_on_text_touched_not_just_edit_count(tmp_path):
    """The neutral arm only cancels 'the input was edited at all' to the extent
    it rewrites a comparable amount of text.  Entity count was already matched;
    occurrence count is now preferred too."""
    code, out = run_build(tmp_path, n_per_type=4)
    assert code == 0
    report = json.loads(
        out.with_suffix(out.suffix + ".report.json").read_text(encoding="utf-8"))
    risk = report["evidence_substitution_risk"]
    assert risk["neutral_occurrence_pairs"] > 0
    assert risk["neutral_occurrence_gap_max"] == 0
    assert "share_embedded" in risk


# ----------------------------------------- length matching as an invariant

def test_a_donor_of_the_wrong_length_is_not_offered_by_default():
    """§8.3 asks for a length-matched substitution.  It used to be a statistic
    in the report (`±3` tolerated, differences merely counted); an anchor with
    no exact donor is now dropped instead."""
    pool = DonorPool(["一二三", "一二三四"], len)
    assert pool.pick("甲乙丙", forbidden_text="", forbidden=set(),
                     salt="s") == ("一二三", 0)
    assert pool.pick("甲乙丙丁戊", forbidden_text="", forbidden=set(),
                     salt="s") is None


def test_a_looser_build_is_reachable_but_only_on_purpose():
    pool = DonorPool(["一二三"], len, max_length_delta=2)
    donor, delta = pool.pick("甲乙丙丁", forbidden_text="", forbidden=set(),
                             salt="s")
    assert (donor, delta) == ("一二三", -1)


def test_the_build_refuses_to_ship_a_loose_donor(tmp_path, monkeypatch):
    """Defence in depth: with --max_length_delta 0 the pool cannot return one,
    so this pins the property instead of describing it."""
    import tcr.support_probe.build as build_module
    original = build_module.DonorPool.pick

    def loose(self, target, **kwargs):
        result = original(self, target, **kwargs)
        return (result[0], 2) if result else result

    monkeypatch.setattr(build_module.DonorPool, "pick", loose)
    code, out = run_build(tmp_path, n_per_type=4)
    assert code == 1
    assert not out.exists()


# ------------------------------- collateral the substitution can CREATE

def test_a_donor_that_contains_another_candidate_is_not_offered():
    """`is_independent` protects the entity being REMOVED.  The same rule has
    to hold for the one being inserted: substituting in `中医学会` where the
    candidate list also holds `中医` manufactures a mention of `中医`, and a
    manufactured mention changes that pair's support exactly as a deleted one
    does.  Measured on the frozen eval set this rejected 7 of 152 evidence
    anchors, `北京` and `中医` being the repeat offenders."""
    pool = DonorPool(["中医学会", "钢铁公司"], lambda name: len(name))
    forbidden = {"银针", "中医"}
    donor, _delta = pool.pick("银针针灸", forbidden_text="", forbidden=forbidden,
                              salt="s")
    assert donor == "钢铁公司"


def test_a_donor_nested_inside_a_candidate_is_not_offered_either():
    pool = DonorPool(["中医", "钢铁"], lambda name: len(name))
    donor, _delta = pool.pick("银针", forbidden_text="",
                              forbidden={"银针", "中医学会"}, salt="s")
    assert donor == "钢铁"


def test_a_donor_already_in_the_candidate_list_is_still_refused():
    """`is_independent` skips the exact match (it asks about OTHER entities),
    so equality has to be tested on its own."""
    pool = DonorPool(["中医"], lambda name: len(name))
    assert pool.pick("银针", forbidden_text="", forbidden={"中医"},
                     salt="s") is None


def test_a_record_whose_anchor_fails_an_invariant_is_replaced_not_dropped(
        tmp_path, monkeypatch):
    """Validation used to run only AFTER the first n_per_type were taken, so a
    failure was unrecoverable: nothing replaced the anchor and the build died
    as under-filled with candidates still queued.  On the frozen eval set that
    really happened, and no anchor set could be built at all."""
    import tcr.support_probe.build as build_module
    original = build_module.build_preterminal_anchor

    def poison(record, axis, size, pool):
        # By record, not by call order: the feasibility sweep calls this too,
        # so a counter would spend itself before selection ever starts.
        anchor = original(record, axis, size, pool)
        if anchor is not None and axis == "evidence" and record.key in (0, 1):
            broken = list(anchor.variants)
            victim = next(i for i, v in enumerate(broken) if v.arm == "manip")
            broken[victim] = dataclasses.replace(broken[victim], n_edits=999)
            return dataclasses.replace(anchor, variants=tuple(broken))
        return anchor

    monkeypatch.setattr(build_module, "build_preterminal_anchor", poison)
    code, out = run_build(tmp_path, n_per_type=4, n_records=10)
    assert code == 0, "the build must draw the next records instead of failing"
    report = json.loads(
        out.with_suffix(out.suffix + ".report.json").read_text(encoding="utf-8"))
    assert report["n_skipped_by_validation"] == 2
    assert report["n_rejected"] == 0                 # gate 4 still sees none
    assert report["anchors_by_type"]["evidence"] == 4


def test_a_skipped_record_is_named_in_the_report(tmp_path, monkeypatch):
    """A pool quietly eaten by rejections has to be visible before it runs
    out: 152 records for 128 paired anchors is not much headroom."""
    import tcr.support_probe.build as build_module
    original = build_module.build_zero_answer_anchor

    def poison(record, donor):
        anchor = original(record, donor)
        if anchor is not None and record.key == 0:
            return dataclasses.replace(anchor, assistant_prefix="NOT A LIST")
        return anchor

    monkeypatch.setattr(build_module, "build_zero_answer_anchor", poison)
    code, out = run_build(tmp_path, n_per_type=4, n_records=10)
    assert code == 0
    report = json.loads(
        out.with_suffix(out.suffix + ".report.json").read_text(encoding="utf-8"))
    assert any(entry["key"] == 0 for entry in report["skipped_by_validation"])


# ------------------------------- the context ceiling, on EVERY variant

def test_the_length_gate_measures_every_variant_of_both_axes(pool):
    """It used to pick the longest variant by CHARACTERS and measure only that
    one, on the evidence axis only.  The two axes edit different things -- one
    the document, one the candidate list -- so the character-longest variant
    need not be the token-longest, and an over-long admissibility variant could
    ship behind a short evidence one.  An over-long anchor cannot be scored at
    all: it is dropped, never truncated."""
    from tcr.support_probe.build import context_tokens, variant_tokens
    anchor = build_preterminal_anchor(make_record(), "admissibility", 2, pool)
    per_variant = [variant_tokens(anchor, v, len, 0) for v in anchor.variants]
    assert context_tokens(anchor, len, 0) == max(per_variant)
    assert len(set(per_variant)) > 1              # the variants really differ


def test_the_chat_template_counts_towards_the_ceiling(pool):
    """The scored context is the chat-templated prompt plus the prefix, so the
    template's own tokens are part of the length the gate is about."""
    from tcr.support_probe.build import context_tokens
    anchor = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    assert (context_tokens(anchor, len, 40)
            == context_tokens(anchor, len, 0) + 40)


def test_an_over_long_anchor_is_passed_over(tmp_path, monkeypatch):
    from tcr.support_probe import protocol as protocol_module
    monkeypatch.setattr(protocol_module, "MAX_CONTEXT_TOKENS", 10)
    code, out = run_build(tmp_path, n_per_type=1)
    assert code == 1                       # nothing fits, nothing ships
    assert not out.exists()


# ------------------------------------- the magnitude gap the analysis needs

def test_occurrence_gap_is_the_worst_level_and_evidence_only(pool):
    """The readouts carry it per anchor, and the analysis filters the evidence
    slope on `== 0`; an anchor matched at r=0.5 but not at r=1 is not matched."""
    from tcr.support_probe.anchors import occurrence_gap
    evidence = build_preterminal_anchor(make_record(), "evidence", 2, pool)
    assert occurrence_gap(evidence) == 0
    # the admissibility axis rewrites the list, not the document
    assert occurrence_gap(
        build_preterminal_anchor(make_record(), "admissibility", 2, pool)) is None

    variants = list(evidence.variants)
    index = next(i for i, v in enumerate(variants)
                 if v.arm == "neutral" and v.level == 1.0)
    edits = [dict(edit, n_occurrences=edit["n_occurrences"] + 4)
             for edit in variants[index].edits]
    variants[index] = dataclasses.replace(variants[index], edits=tuple(edits))
    # it is the TOTAL mentions rewritten by the arm, so every edit counts
    assert occurrence_gap(
        dataclasses.replace(evidence, variants=tuple(variants))) == 4 * len(edits)


def test_no_positive_control_names_the_token_it_controls():
    """The control measures the CLOSE branch's score.  A text that says `]`
    outright primes the very token under measurement, and for a model that has
    narrowed onto a fixed format that priming is the only signal left -- both
    arms then move the same way and the control stops being a control."""
    from tcr.support_probe.anchors import (PC_STOP, PC_CONTINUE_NAMED, PC_CONTINUE_PLAIN,
                            PC_NEUTRAL, CLOSE_CONTINUATION)
    for name, text in (("PC_STOP", PC_STOP),
                       ("PC_CONTINUE_NAMED", PC_CONTINUE_NAMED),
                       ("PC_CONTINUE_PLAIN", PC_CONTINUE_PLAIN),
                       ("PC_NEUTRAL", PC_NEUTRAL)):
        assert CLOSE_CONTINUATION not in text, name
    # and they must still be distinguishable instructions
    assert "结束" in PC_STOP and "继续" in PC_CONTINUE_PLAIN
