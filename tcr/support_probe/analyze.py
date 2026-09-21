# coding=utf-8
"""Turn one model's readouts into the §9.5 measures and the §E2 gate verdict.

What is computed, and why in this order
---------------------------------------
1. **delta** (§9.5).  Per anchor and level, the manipulated margin minus its
   token- and position-matched neutral counterpart.  Subtracting the neutral is
   what removes "the input was edited at all" from "the input lost support";
   it also cancels the presence penalty, which is identical across an anchor's
   variants.
2. **slope** `b = d(delta)/dr` over r in {0, 0.5, 1}.  `delta(0) = 0` holds by
   construction (both arms are the untouched record), so the fit is through the
   origin and `b` reads directly as "predicted StopMargin change from no
   manipulation to full manipulation", in nat -- the plan's own wording.
   `ACI-logit` is the slope on the admissibility axis, `ECI-logit` on evidence.
3. **positive control** (§8.3 E, §9.5).  `PC = SM(pc_stop) - SM(pc_continue)`,
   sign-unified so it must be positive.  A model that does not respond to an
   unambiguous instruction cannot be said to have lost *support* conditioning
   specifically -- that is the whole point of H3's negation condition -- so PC
   gates the normalised readout and never enters it as a support measure.
4. **nSCI** = `b / max(|PC|, tau)`: how much of the general input response the
   support response retains.  Secondary by construction; §9.5 forbids it from
   replacing the raw logit.
5. **baseline strata**: everything again by quintile of the untouched margin,
   because an effect that exists only in a deep floor is not a support effect.

The six instrument gates are §E2's own pass criteria, evaluated verbatim.  A
failed gate is not a result; it means the instrument is not ready and E3 must
not start.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import protocol
from .io_utils import (
    read_json, read_jsonl, sanitize_filename, sha256_file, write_json,
)

LOG = logging.getLogger("tcr.support_probe.analyze")

MARGINS = ("sm_primary", "sm_raw", "sm_unpenalised")
AXES = ("evidence", "admissibility")

#: The margin family measured in the state the SHORT CONTINUATIONS are sampled
#: in.  vLLM is handed the assistant prefix as prompt and penalises output
#: tokens only, so the hazard's first step sees no presence penalty on the
#: prefix while `sm_primary` does (see protocol.HAZARD_PRESENCE_STATE).  Gate 7
#: compares like with like by reading this one.
HAZARD_MARGIN = "sm_unpenalised"


# ------------------------------------------------------------------- helpers

def bootstrap_mean(values: Sequence[float], *, n_boot: int = protocol.BOOTSTRAP,
                   seed: int = protocol.BOOTSTRAP_SEED) -> Dict[str, Optional[float]]:
    """Anchor-clustered mean with a percentile CI.

    The resampling unit is the anchor: an anchor contributes several levels and
    several arms, and treating those as independent would report an interval
    several times too narrow."""
    clean = np.array([v for v in values if v is not None and np.isfinite(v)],
                     dtype=float)
    if clean.size == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, clean.size, size=(n_boot, clean.size))
    means = clean[draws].mean(axis=1)
    return {"n": int(clean.size), "mean": float(clean.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "sd": float(clean.std(ddof=1)) if clean.size > 1 else 0.0}


def slope_through_origin(levels: Sequence[float],
                         deltas: Sequence[float]) -> Optional[float]:
    """OLS of delta on r with no intercept.

    delta(0) = 0 is exact here (the r=0 manipulated and neutral arms are the
    same untouched record), so forcing the line through the origin is not an
    assumption -- and it makes b equal the predicted delta at full
    manipulation, which is how §9.5 asks for it to be read."""
    pairs = [(r, d) for r, d in zip(levels, deltas)
             if d is not None and np.isfinite(d)]
    if len(pairs) < 2:
        return None
    numerator = sum(r * d for r, d in pairs)
    denominator = sum(r * r for r, _ in pairs)
    if denominator <= 0:
        return None
    return numerator / denominator


# ------------------------------------------------------------- per-anchor view

class AnchorView:
    """Every cell of one anchor, indexed the way the measures need it."""

    def __init__(self, anchor_id: str) -> None:
        self.anchor_id = anchor_id
        self.anchor_type = ""
        self.axis = "none"
        self.source = ""
        self.key: Any = None
        self.n_remainder = 0
        self.occurrence_gap: Optional[int] = None
        """Evidence axis only: how unequal the manipulated and neutral arms are
        in the amount of text they rewrite.  0 means the edit magnitude is
        matched exactly, so "the input was edited at all" cancels exactly; see
        `anchors.occurrence_gap`.  None on readouts written before this was
        recorded, and on axes that do not touch the document."""
        self.by_arm_level: Dict[Tuple[str, float], Dict[str, Any]] = {}

    def add(self, row: Dict[str, Any]) -> None:
        self.anchor_type = row["anchor_type"]
        self.source = row.get("source", "")
        self.key = row.get("key")
        self.n_remainder = int(row.get("n_remainder", 0))
        if row.get("occurrence_gap") is not None:
            self.occurrence_gap = int(row["occurrence_gap"])
        if row["axis"] != "none":
            self.axis = row["axis"]
        self.by_arm_level[(row["arm"], float(row["level"]))] = row

    def get(self, arm: str, level: float = 0.0) -> Optional[Dict[str, Any]]:
        return self.by_arm_level.get((arm, level))

    def margin(self, arm: str, level: float, field: str) -> Optional[float]:
        """Absent is None, not an error: the hazard is measured on a subset of
        the arms, so a hazard field is legitimately missing on the controls."""
        row = self.get(arm, level)
        if row is None:
            return None
        value = row.get(field)
        return None if value is None else float(value)

    # -- the measures ----------------------------------------------------

    def deltas(self, field: str) -> Dict[float, Optional[float]]:
        """delta(r) = SM(manip) - SM(neutral), expected POSITIVE (§9.5)."""
        out: Dict[float, Optional[float]] = {0.0: 0.0}
        for level in protocol.LEVELS:
            if level == 0.0:
                continue
            manip = self.margin("manip", level, field)
            neutral = self.margin("neutral", level, field)
            out[level] = None if manip is None or neutral is None else manip - neutral
        return out

    def neutral_drift(self, field: str) -> Dict[float, Optional[float]]:
        """SM(neutral) - SM(base): gate 1 wants this indistinguishable from 0."""
        base = self.margin("base", 0.0, field)
        out: Dict[float, Optional[float]] = {}
        for level in protocol.LEVELS:
            if level == 0.0:
                continue
            neutral = self.margin("neutral", level, field)
            out[level] = None if neutral is None or base is None else neutral - base
        return out

    def slope(self, field: str) -> Optional[float]:
        deltas = self.deltas(field)
        levels = [r for r in sorted(deltas) if deltas[r] is not None]
        return slope_through_origin(levels, [deltas[r] for r in levels])

    def pc(self, field: str) -> Dict[str, Optional[float]]:
        stop = self.margin("pc_stop", 0.0, field)
        cont = self.margin("pc_continue", 0.0, field)
        neutral = self.margin("pc_neutral", 0.0, field)
        return {
            "pc": None if stop is None or cont is None else stop - cont,
            "pc_stop_effect": None if stop is None or neutral is None else stop - neutral,
            "pc_continue_effect": None if cont is None or neutral is None else neutral - cont,
        }

    def base_row(self) -> Optional[Dict[str, Any]]:
        return self.get("base", 0.0)

    def informative(self, prob_field: str = "close_prob_pre_filter"
                    ) -> Tuple[bool, str]:
        """§E2 criterion 5: floor / ceiling anchors must be flagged, not hidden.

        An anchor whose close token is unreachable even when ALL remaining
        support is gone can never show a support response, and one that is
        already certain to close can never show more.

        Asked on the PRE-FILTER probability, not the post-top-k/top-p one, and
        that is not a loosening -- it is the difference between a graded
        quantity and a step function.  `close_sampler_prob` is exactly 0.0
        whenever the close token falls outside the nucleus, and at a structured
        JSON boundary the model is peaked enough that top_p=0.8 keeps little
        more than the argmax.  Measured on the frozen anchor set, that rejected
        78-81% of anchors on all three calibration models while their margins
        sat at only about -1.6 nat and moved by +1.0 to +2.3 nat under
        manipulation: anchors that respond perfectly well, discarded for a
        property of the sampler rather than of the measurement.  Sampler
        reachability is still reported, as `sampler_reachable_share`; it is a
        deployment caveat, not an exclusion rule.

        `prob_field` selects the SAMPLING STATE the question is asked in.  The
        margins use the penalised state (what E1's generation sees); the hazard
        has to use the unpenalised one, because that is the distribution its
        draws actually came from."""
        base = self.base_row()
        if base is None:
            return False, "no_base"
        probability = base.get(prob_field)
        if probability is None:
            return False, "no_base"
        full = self.get("manip", 1.0)
        if float(probability) >= 0.999:
            return False, "ceiling"
        if full is not None and float(full.get(prob_field) or 0.0) <= 1e-6:
            return False, "floor"
        if float(probability) <= 1e-6 and full is None:
            return False, "floor"
        return True, "informative"


#: Fields the hazard stage contributes to the cell it shares with the scorer.
HAZARD_FIELDS = ("stop_hazard", "stop_hazard_decided", "n_closed", "n_continued",
                 "n_other", "n_noncanonical", "n_samples")


#: A hazard row is only a measurement if it carries these.  Without them the
#: cell contributes nothing to any rate, so counting it as merged would let a
#: half-written file pass a completeness check by volume.
HAZARD_REQUIRED = ("stop_hazard", "n_samples", "n_closed", "n_continued",
                   "n_other", "n_noncanonical")


def _count(value: Any) -> Optional[int]:
    """A draw count, or None if this is not one.

    `int(x)` is not a check: it accepts floats, numeric strings and bools, and
    silently truncates.  A hazard file that was concatenated or hand-edited is
    exactly where those turn up, and every one of them would flow into a rate."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def hazard_row_problem(row: Dict[str, Any]) -> Optional[str]:
    """Why this hazard row is not a usable measurement, if it is not.

    The writer produces all of these by construction, so any violation means
    the file was corrupted, concatenated from another run, or written by a
    different protocol.  None of that is visible downstream: a `stop_hazard`
    of 1.7 or a row of 8 samples would flow into the rates and the CI as if it
    were a measurement.
    """
    for field in HAZARD_REQUIRED:
        if row.get(field) is None:
            return f"no {field}"
    n_samples = _count(row.get("n_samples"))
    if n_samples != protocol.HAZARD_SAMPLES:
        return (f"n_samples={row.get('n_samples')!r}, not the protocol's "
                f"{protocol.HAZARD_SAMPLES}")
    counts = {}
    for field in ("n_closed", "n_continued", "n_other", "n_noncanonical"):
        # Present, not defaulted: the writer always emits these, so a row that
        # omits one is a corrupted or hand-edited row, and `.get(field, 0)`
        # would let it through as a legitimate zero.
        value = _count(row.get(field))
        if value is None:
            return f"{field}={row.get(field)!r} is not a count"
        if value > n_samples:
            return f"{field}={value} exceeds n_samples={n_samples}"
        counts[field] = value
    labelled = counts["n_closed"] + counts["n_continued"] + counts["n_other"]
    if labelled != n_samples:
        return (f"closed+continued+other={labelled} but n_samples={n_samples}: "
                "the draws do not add up")
    for field in ("stop_hazard", "stop_hazard_decided"):
        rate = row.get(field)
        if rate is None:
            continue                       # decided is None when nothing decided
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            return f"{field}={rate!r} is not a number"
        if not 0.0 <= float(rate) <= 1.0:
            return f"{field}={rate!r} is not a rate in [0, 1]"
    if abs(float(row["stop_hazard"]) - counts["n_closed"] / n_samples) > 1e-6:
        return (f"stop_hazard={row['stop_hazard']} disagrees with "
                f"n_closed={counts['n_closed']} of {n_samples}")
    decided = n_samples - counts["n_other"]
    expected_decided = (counts["n_closed"] / decided) if decided else None
    got = row.get("stop_hazard_decided")
    if expected_decided is None:
        if got is not None:
            return "stop_hazard_decided is set but no draw decided anything"
    elif got is None or abs(float(got) - expected_decided) > 1e-6:
        return (f"stop_hazard_decided={got!r} disagrees with "
                f"{counts['n_closed']}/{decided}")
    return None


def merge_hazard(by_cell: Dict[str, Dict[str, Any]],
                 hazard_path: str | Path,
                 expect_tag: Optional[str] = None) -> Dict[str, Any]:
    """Fold the hazard rows into the scorer's cells; count what happened.

    Counted by DISTINCT cell, and only for rows that survive
    :func:`hazard_row_problem`.  Rows, not cells, is the wrong unit twice over:
    a duplicated `cell_id` would pad the total and hide a cell that never
    landed, and a row whose `stop_hazard` is missing or out of range would be
    counted as measured while contributing nothing -- or worse, contributing
    nonsense.

    The counts are returned rather than logged away because a hazard file that
    joins onto NOTHING is the silent failure this guards: `has_hazard` would
    come out false, gate 7 would be absent rather than failed, and a report
    showing every gate green would be describing a run whose behaviour column
    was never measured.
    """
    merged: set = set()
    duplicates = 0
    malformed = 0
    unmatched = 0
    examples: List[str] = []
    reasons: List[str] = []
    presence_states: set = set()
    foreign_tags: set = set()
    for row in read_jsonl(hazard_path):
        cell_id = row.get("cell_id")
        if expect_tag is not None and row.get("tag") not in (None, expect_tag):
            foreign_tags.add(str(row.get("tag")))
            continue
        target = by_cell.get(cell_id)
        if target is None:
            unmatched += 1
            if len(examples) < 5:
                examples.append(str(cell_id))
            continue
        problem = hazard_row_problem(row)
        if problem is not None:
            malformed += 1
            if len(reasons) < 5:
                reasons.append(f"{cell_id}: {problem}")
            continue
        if cell_id in merged:
            duplicates += 1
        presence_states.add(row.get("presence_state"))
        for field in HAZARD_FIELDS:
            if row.get(field) is not None:
                target[field] = row[field]
        merged.add(cell_id)
    return {"merged": len(merged), "duplicates": duplicates,
            "malformed": malformed, "unmatched": unmatched,
            "unmatched_examples": examples, "malformed_examples": reasons,
            "presence_states": presence_states, "foreign_tags": foreign_tags}


def expected_hazard_cells(by_cell: Dict[str, Dict[str, Any]],
                          arms: Optional[str]) -> Optional[int]:
    """How many cells the hazard stage should have covered, from its own arms.

    `arms` comes out of the hazard sentinel (it is part of that stage's
    identity), so this is "what the finished run said it measured" rather than
    a guess -- which is what makes `merged < expected` a real shortfall."""
    if not arms:
        return None
    if arms == "all":
        return len(by_cell)
    wanted = {name.strip() for name in arms.split(",") if name.strip()}
    return sum(1 for row in by_cell.values() if row.get("arm") in wanted)


def load_views(path: str | Path,
               hazard_path: Optional[str | Path] = None,
               *, hazard_arms: Optional[str] = None,
               strict_hazard: bool = False,
               expect_cells: Optional[int] = None,
               expect_presence_state: Optional[str] = None,
               expect_tag: Optional[str] = None,
               readouts_sha256: Optional[str] = None,
               hazard_sha256: Optional[str] = None) -> Dict[str, AnchorView]:
    """Readouts, with the hazard folded into the SAME cells when it was run.

    Merging rather than summarising separately is what makes the hazard usable:
    every measure below is parameterised by a field name, so once `stop_hazard`
    sits on the cell, the slope, the neutral drift and the positive control all
    fall out of the machinery that already exists for the margins -- and they
    are computed over exactly the same anchors, which is the only way the two
    can be compared for the §9.5 "moved together" reading.

    With ``strict_hazard`` the join must succeed: a hazard file that was asked
    for and did not land is an error, not an empty column.
    """
    # The manifest has to prove the FILE, not just its path and row count: a
    # readout file swapped in by hand has the same cell ids, the same count and
    # the same manifest beside it.
    if readouts_sha256 and sha256_file(path) != readouts_sha256:
        raise SystemExit(
            f"{path} does not match the sha256 its manifest recorded; it has "
            "been replaced or appended to since the score run finished")

    views: Dict[str, AnchorView] = {}
    by_cell: Dict[str, Dict[str, Any]] = {}
    n_rows = 0
    foreign_tags: set = set()
    duplicate_cells: List[str] = []
    for row in read_jsonl(path):
        n_rows += 1
        if expect_tag is not None and row.get("tag") not in (None, expect_tag):
            foreign_tags.add(str(row.get("tag")))
        view = views.setdefault(row["anchor_id"], AnchorView(row["anchor_id"]))
        view.add(row)
        cell_id = row.get("cell_id")
        if cell_id:
            if cell_id in by_cell and len(duplicate_cells) < 5:
                duplicate_cells.append(str(cell_id))
            by_cell[cell_id] = row
    # A repeated score cell is not a resume either: the scorer skips cells the
    # file already holds, so a repeat means two runs' output in one file and the
    # later row silently replacing the earlier.  `expect_cells` (the count the
    # score manifest recorded) turns a short file into a failure rather than a
    # thinner sample -- fewer anchors only widen a CI, so nothing else would say.
    problems: List[str] = []
    if foreign_tags:
        problems.append(f"{path} holds rows tagged {sorted(foreign_tags)}, but "
                        f"this analysis is {expect_tag!r}")
    if duplicate_cells:
        problems.append(
            f"{n_rows - len(by_cell)} duplicate score cell(s) in {path}, e.g. "
            f"{duplicate_cells[:2]}; the file holds output from more than one "
            "run and the later row silently overwrites the earlier")
    if expect_cells is not None and len(by_cell) != expect_cells:
        problems.append(
            f"{path} holds {len(by_cell)} distinct cell(s), but its manifest "
            f"records {expect_cells}")
    if problems:
        raise SystemExit("readouts are not usable: " + "; ".join(problems))

    if hazard_path and not Path(hazard_path).is_file():
        if strict_hazard:
            raise SystemExit(f"hazard file requested but missing: {hazard_path}")
        LOG.warning("no hazard file at %s; the behaviour column stays empty",
                    hazard_path)
    elif hazard_path:
        if hazard_sha256 and sha256_file(hazard_path) != hazard_sha256:
            raise SystemExit(
                f"{hazard_path} does not match the sha256 its manifest "
                "recorded; it has been replaced or appended to since the "
                "hazard run finished")
        stats = merge_hazard(by_cell, hazard_path, expect_tag=expect_tag)
        expected = expected_hazard_cells(by_cell, hazard_arms)
        LOG.info("merged %d distinct hazard cell(s) from %s (expected %s; "
                 "%d unmatched, %d duplicate, %d malformed)",
                 stats["merged"], hazard_path, expected, stats["unmatched"],
                 stats["duplicates"], stats["malformed"])
        if strict_hazard:
            problems = []
            if stats["merged"] == 0:
                examples = stats["unmatched_examples"][:2]
                problems.append(
                    f"no hazard row joined onto a scored cell "
                    f"({stats['unmatched']} rows name other cells, e.g. "
                    f"{examples}); the file belongs to another model or "
                    "another anchor set")
            elif expected is not None and stats["merged"] < expected:
                problems.append(
                    f"only {stats['merged']} of {expected} hazard cell(s) "
                    "joined; the hazard run did not cover this readout set")
            if stats["unmatched"]:
                problems.append(f"{stats['unmatched']} hazard row(s) match no "
                                "scored cell")
            if stats["malformed"]:
                problems.append(
                    f"{stats['malformed']} hazard row(s) are not a usable "
                    f"measurement, e.g. {stats['malformed_examples'][:2]}")
            if stats["foreign_tags"]:
                problems.append(
                    f"hazard rows are tagged {sorted(stats['foreign_tags'])}, "
                    f"not {expect_tag!r}: this file is another model's")
            if (expect_presence_state is not None
                    and stats["presence_states"]
                    and stats["presence_states"] != {expect_presence_state}):
                problems.append(
                    f"hazard rows were sampled in "
                    f"{sorted(stats['presence_states'])}, this analysis "
                    f"compares against {expect_presence_state!r}")
            if stats["duplicates"]:
                # The writer skips cells already in the file, so a repeat is
                # not a resume: it is two runs' output in one file, and the
                # last line silently wins.
                problems.append(
                    f"{stats['duplicates']} cell(s) appear more than once; the "
                    "file holds output from more than one run and the later "
                    "row silently overwrites the earlier")
            if problems:
                raise SystemExit("hazard merge failed: " + "; ".join(problems))
    return views


# ---------------------------------------------------------------- summarise

def _quintile_edges(values: Sequence[float]) -> List[float]:
    clean = np.array([v for v in values if v is not None and np.isfinite(v)])
    if clean.size == 0:
        return []
    return [float(np.quantile(clean, q)) for q in protocol.BASELINE_QUANTILES]


def _stratum_of(value: Optional[float], edges: Sequence[float]) -> Optional[int]:
    if value is None or not edges:
        return None
    index = 0
    for edge in edges:
        if value > edge:
            index += 1
    return index


def axis_summary(views: Sequence[AnchorView], axis: str, field: str, *,
                 n_boot: int = protocol.BOOTSTRAP) -> Dict[str, Any]:
    """Slopes, deltas, neutral drift and strata for one support axis."""
    members = [v for v in views if v.axis == axis and v.anchor_type == axis]
    informative = [v for v in members if v.informative()[0]]
    out: Dict[str, Any] = {
        "axis": axis, "margin": field,
        "n_anchors": len(members),
        "n_informative": len(informative),
        "uninformative": _count_flags(members),
    }
    if not informative:
        return out

    out["slope"] = bootstrap_mean([v.slope(field) for v in informative], n_boot=n_boot)
    for level in protocol.LEVELS:
        if level == 0.0:
            continue
        out[f"delta_r{level}"] = bootstrap_mean(
            [v.deltas(field)[level] for v in informative], n_boot=n_boot)
        out[f"neutral_drift_r{level}"] = bootstrap_mean(
            [v.neutral_drift(field)[level] for v in informative], n_boot=n_boot)

    # deployment-facing readouts at full manipulation (§9.5 非饱和读出)
    rank_delta, prob_delta, survive_delta = [], [], []
    for view in informative:
        manip, neutral = view.get("manip", 1.0), view.get("neutral", 1.0)
        if manip is None or neutral is None:
            continue
        rank_delta.append(float(neutral["close_rank"] - manip["close_rank"]))
        prob_delta.append(float(manip["close_sampler_prob"]
                                - neutral["close_sampler_prob"]))
        survive_delta.append(float(bool(manip["close_survives_top_p"])
                                   - bool(neutral["close_survives_top_p"])))
    # What the OLD floor rule was really measuring: can the sampler draw the
    # close token at all?  A deployment caveat worth reporting -- a margin that
    # moves while the token stays outside the nucleus changes no behaviour --
    # but not a reason to drop the anchor from the margin analysis.
    reachable = [bool(row.get("close_survives_top_p"))
                 for row in (view.get("manip", 1.0) for view in informative)
                 if row is not None]
    out["sampler_reachable_share"] = (sum(reachable) / len(reachable)
                                      if reachable else None)
    out["close_rank_gain_r1"] = bootstrap_mean(rank_delta, n_boot=n_boot)
    out["close_sampler_prob_gain_r1"] = bootstrap_mean(prob_delta, n_boot=n_boot)
    out["close_top_p_survival_gain_r1"] = bootstrap_mean(survive_delta, n_boot=n_boot)

    # positive control and the normalised readout
    pcs = [view.pc(field)["pc"] for view in informative]
    out["pc"] = bootstrap_mean(pcs, n_boot=n_boot)
    out["pc_stop_effect"] = bootstrap_mean(
        [view.pc(field)["pc_stop_effect"] for view in informative],
        n_boot=n_boot)
    out["pc_continue_effect"] = bootstrap_mean(
        [view.pc(field)["pc_continue_effect"] for view in informative],
        n_boot=n_boot)

    for tau in (protocol.PC_TAU, *protocol.PC_TAU_ROBUST):
        qualified = [view for view in informative
                     if (view.pc(field)["pc"] or 0.0) and
                     abs(view.pc(field)["pc"]) >= tau]
        nsci = [(view.slope(field) or 0.0) / max(abs(view.pc(field)["pc"]), tau)
                for view in qualified if view.slope(field) is not None]
        key = "nsci" if tau == protocol.PC_TAU else f"nsci_tau{tau}"
        out[key] = {**bootstrap_mean(nsci, n_boot=n_boot), "tau": tau,
                    "n_qualified": len(qualified)}

    # baseline strata: an effect that lives only in a deep floor is not a
    # support effect (§9.5), so the slope is reported by quintile of the
    # untouched margin as well as pooled.
    base_margins = [view.margin("base", 0.0, field) for view in informative]
    edges = _quintile_edges(base_margins)
    strata: Dict[str, Any] = {}
    for view, base in zip(informative, base_margins):
        index = _stratum_of(base, edges)
        if index is None:
            continue
        strata.setdefault(str(index), []).append(view.slope(field))
    out["baseline_edges"] = edges
    out["baseline_margin"] = bootstrap_mean(base_margins, n_boot=n_boot)
    out["by_baseline_quintile"] = {
        name: bootstrap_mean(values, n_boot=n_boot) for name, values in sorted(strata.items())}

    # Sensitivity: the same slope on anchors whose manipulated and neutral arms
    # rewrote EXACTLY the same amount of text.  The evidence axis substitutes
    # over the whole document, so collateral outside the entity is possible; the
    # claim that it differences out holds exactly only where the two arms are
    # matched in edit magnitude.  Reported beside the pooled slope rather than
    # replacing it -- a subset is a smaller sample, not a better estimator --
    # so a reader can see whether the effect survives the strict matching.
    gaps = [view.occurrence_gap for view in informative]
    if any(gap is not None for gap in gaps):
        matched = [view for view in informative if view.occurrence_gap == 0]
        out["occurrence_gap"] = bootstrap_mean(
            [float(gap) for gap in gaps if gap is not None], n_boot=n_boot)
        out["n_occurrence_matched"] = len(matched)
        out["slope_occurrence_matched"] = bootstrap_mean(
            [v.slope(field) for v in matched], n_boot=n_boot)
    return out


def _count_flags(views: Sequence[AnchorView],
                 prob_field: str = "close_prob_pre_filter") -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for view in views:
        ok, reason = view.informative(prob_field)
        if not ok:
            counts[reason] += 1
    return dict(counts)


def zero_answer_summary(views: Sequence[AnchorView], field: str, *,
                        n_boot: int = protocol.BOOTSTRAP) -> Dict[str, Any]:
    """§8.3 D: the EmptyMargin at `[` when the answer really is `[]`.

    This is the reference point the preterminal margins are read against: it is
    what "the model can tell there is nothing to say" looks like on this
    instrument."""
    members = [v for v in views if v.anchor_type == "zero_answer"]
    if not members:
        return {"n_anchors": 0}
    return {
        "n_anchors": len(members),
        "empty_margin": bootstrap_mean(
            [v.margin("base", 0.0, field) for v in members], n_boot=n_boot),
        "close_sampler_prob": bootstrap_mean(
            [float(v.base_row()["close_sampler_prob"]) for v in members
             if v.base_row()], n_boot=n_boot),
        "close_is_top1_rate": bootstrap_mean(
            [float(v.base_row()["close_rank"] == 1) for v in members
             if v.base_row()], n_boot=n_boot),
        "pc": bootstrap_mean([v.pc(field)["pc"] for v in members], n_boot=n_boot),
    }


def hazard_summary(views: Sequence[AnchorView], axis: str, *,
                   n_boot: int = protocol.BOOTSTRAP) -> Dict[str, Any]:
    """What the model DOES at the boundary, on the same anchors as the margin.

    Deliberately not `axis_summary`: nSCI divides by a positive control measured
    in logits, and a rate in [0, 1] does not live in those units.  What carries
    over is the shape of the measurement -- per-anchor slope of
    manip-minus-neutral against the removal level, plus the neutral drift that
    says the edit itself did nothing.

    `other_rate` and `noncanonical_rate` are reported because they are the
    failure mode that matters here: if the model stops producing a recognisable
    structural decision, the hazard is not a small effect, it is not a
    measurement at all.

    Floor/ceiling is judged in the hazard's OWN sampling state (the prefix
    carries no presence penalty there), because that is the distribution the
    draws came from.  Judging it by the penalised probabilities would count
    anchors as informative whose close token the hazard's sampler could never
    have produced, and report "the model never closes" for an anchor where the
    honest answer is "we asked in a state where it could not".
    """
    field = "stop_hazard"
    prob_field = "close_prob_pre_filter_unpenalised"
    members = [v for v in views if v.axis == axis and v.anchor_type == axis]
    with_hazard = [v for v in members if v.margin("base", 0.0, field) is not None]
    out: Dict[str, Any] = {"axis": axis, "n_anchors": len(members),
                           "n_with_hazard": len(with_hazard),
                           "presence_state": protocol.HAZARD_PRESENCE_STATE,
                           "margin_family": HAZARD_MARGIN}
    if not with_hazard:
        return out
    # Readouts written before the two states were separated carry only the
    # penalised probability; fall back rather than silently drop every anchor.
    if any(v.base_row() and v.base_row().get(prob_field) is not None
           for v in with_hazard):
        out["floor_state"] = prob_field
    else:
        prob_field = "close_prob_pre_filter"
        out["floor_state"] = prob_field + " (readouts predate the split)"
    informative = [v for v in with_hazard if v.informative(prob_field)[0]]
    out["n_informative"] = len(informative)
    out["uninformative"] = _count_flags(with_hazard, prob_field)
    # How much of the anchor set the hazard's own state can even reach.  A low
    # share here means a flat hazard is about the state, not about the model.
    out["reachable_share"] = (len(informative) / len(with_hazard)
                              if with_hazard else 0.0)
    if not informative:
        return out

    out["slope"] = bootstrap_mean([v.slope(field) for v in informative], n_boot=n_boot)
    for level in protocol.LEVELS:
        if level == 0.0:
            continue
        out[f"delta_r{level}"] = bootstrap_mean(
            [v.deltas(field)[level] for v in informative], n_boot=n_boot)
        out[f"neutral_drift_r{level}"] = bootstrap_mean(
            [v.neutral_drift(field)[level] for v in informative], n_boot=n_boot)
    out["base_rate"] = bootstrap_mean(
        [v.margin("base", 0.0, field) for v in informative], n_boot=n_boot)
    out["manip_r1_rate"] = bootstrap_mean(
        [v.margin("manip", 1.0, field) for v in informative], n_boot=n_boot)
    out["neutral_r1_rate"] = bootstrap_mean(
        [v.margin("neutral", 1.0, field) for v in informative], n_boot=n_boot)

    # Per ANCHOR, not per cell.  An anchor contributes several cells and they
    # are anything but independent (same document, same prefix, same decision
    # token), so pooling cells straight into the bootstrap reports an interval
    # several times too narrow -- and weights anchors by how many hazard cells
    # they happen to have.  Each anchor's cells are averaged first; the
    # resampling unit is then the anchor, as everywhere else in this file.
    other, noncanonical = [], []
    for view in informative:
        cell_other, cell_noncanonical = [], []
        for row in view.by_arm_level.values():
            if row.get("n_samples"):
                cell_other.append(float(row.get("n_other", 0)) / row["n_samples"])
                cell_noncanonical.append(
                    float(row.get("n_noncanonical", 0)) / row["n_samples"])
        if cell_other:
            other.append(sum(cell_other) / len(cell_other))
            noncanonical.append(sum(cell_noncanonical) / len(cell_noncanonical))
    out["other_rate"] = bootstrap_mean(other, n_boot=n_boot)
    out["noncanonical_rate"] = bootstrap_mean(noncanonical, n_boot=n_boot)
    out["n_hazard_cells"] = sum(
        1 for view in informative for row in view.by_arm_level.values()
        if row.get("n_samples"))
    # only meaningful when the hazard was run with --arms all
    pcs = [v.pc(field)["pc"] for v in informative]
    if any(value is not None for value in pcs):
        out["pc"] = bootstrap_mean(pcs, n_boot=n_boot)
    return out


def summarize(views: Dict[str, AnchorView], *, tag: str,
              n_boot: int = protocol.BOOTSTRAP) -> Dict[str, Any]:
    """Every §9.5 measure for one model.

    ``n_boot`` is threaded down to every interval rather than read from the
    protocol at each call site, so `--bootstrap 100` really does produce a
    100-resample run -- and the number used is written into the summary, since
    a CI whose resample count is unknown is not reproducible."""
    all_views = list(views.values())
    summary: Dict[str, Any] = {
        "tag": tag,
        "protocol": protocol.describe(),
        # what the ANALYSIS produced, not what the instrument measured: a
        # summary from older analysis code has a perfectly matching measurement
        # identity and incomparable numbers.
        "summary_schema": protocol.SUMMARY_SCHEMA_VERSION,
        "bootstrap": int(n_boot),
        "n_anchors": len(all_views),
        "n_cells": sum(len(v.by_arm_level) for v in all_views),
    }
    for field in MARGINS:
        summary[field] = {
            axis: axis_summary(all_views, axis, field, n_boot=n_boot)
            for axis in AXES}
        summary[field]["zero_answer"] = zero_answer_summary(all_views, field,
                                                            n_boot=n_boot)

    hazard = {axis: hazard_summary(all_views, axis, n_boot=n_boot)
              for axis in AXES}
    summary["stop_hazard"] = hazard
    summary["has_hazard"] = any(block.get("n_with_hazard") for block in hazard.values())
    # §9.5: the claim needs the margin and the behaviour to move together --
    # measured in the SAME sampling state.  The hazard's prefix carries no
    # presence penalty, so it is read against `sm_unpenalised`, not
    # `sm_primary`.  (The two have identical SLOPES, because the penalty is an
    # anchor-constant that cancels from every delta; they differ in level, and
    # it is the level that decides whether the close token is reachable at all.)
    agreement: Dict[str, Any] = {}
    for axis in AXES:
        margin_slope = _get(summary[HAZARD_MARGIN], axis, "slope", "mean")
        hazard_slope = _get(hazard, axis, "slope", "mean")
        agreement[axis] = (None if margin_slope is None or hazard_slope is None
                           else bool(margin_slope * hazard_slope > 0))
    summary["hazard_agrees_with_margin"] = agreement
    summary["hazard_margin_family"] = HAZARD_MARGIN
    return summary


# -------------------------------------------------------------------- gates

def _ci_excludes_zero_positive(block: Optional[Dict[str, Any]]) -> bool:
    return bool(block and block.get("ci_low") is not None
                and block["ci_low"] > 0)


def _ci_contains_zero(block: Optional[Dict[str, Any]]) -> bool:
    return bool(block and block.get("ci_low") is not None
                and block["ci_low"] <= 0 <= block["ci_high"])


def instrument_gates(summary: Dict[str, Any], *,
                     anchor_report: Optional[Dict[str, Any]] = None,
                     score_manifest: Optional[Dict[str, Any]] = None,
                     neutral_tolerance: float = 0.25) -> List[Dict[str, Any]]:
    """§E2 仪器通过标准, evaluated one by one.

    These decide whether the instrument may be used, not whether the hypothesis
    is true.  A failure means anchors and controls get fixed and E3 does not
    start -- `filter->clean` being significant or not has no vote here."""
    main = summary.get("sm_primary", {})
    gates: List[Dict[str, Any]] = []

    # 1 -- a matched edit that removes no support must move nothing
    drifts = []
    for axis in AXES:
        for level in protocol.LEVELS:
            if level == 0.0:
                continue
            block = main.get(axis, {}).get(f"neutral_drift_r{level}")
            if block and block.get("mean") is not None:
                drifts.append((f"{axis}@r{level}", block))
    ok = all(_ci_contains_zero(block) or abs(block["mean"]) < neutral_tolerance
             for _name, block in drifts)
    gates.append({
        "gate": "1_neutral_effect_near_zero", "pass": bool(drifts) and ok,
        "detail": {name: {"mean": block["mean"], "ci_low": block["ci_low"],
                          "ci_high": block["ci_high"]} for name, block in drifts},
        "note": "a token- and position-matched edit that removes no support "
                "must not move the margin; if it does, the manipulation is "
                "confounded with editing itself",
    })

    # 2 -- the model responds to an unambiguous instruction, in the right sign
    pc_ok, pc_detail = True, {}
    for axis in AXES:
        block = main.get(axis, {})
        for name in ("pc", "pc_stop_effect", "pc_continue_effect"):
            entry = block.get(name)
            pc_detail[f"{axis}.{name}"] = None if not entry else {
                "mean": entry.get("mean"), "ci_low": entry.get("ci_low")}
            if not _ci_excludes_zero_positive(entry):
                pc_ok = False
    gates.append({
        "gate": "2_positive_control_direction", "pass": pc_ok,
        "detail": pc_detail,
        "note": "PC = SM(pc_stop) - SM(pc_continue) must be positive with the "
                "CI excluding 0; without it, a flat support response cannot be "
                "read as a SUPPORT failure (H3's negation condition)",
    })

    # 3 -- the manipulation does something on this model
    axis_pass = {axis: _ci_excludes_zero_positive(main.get(axis, {}).get("slope"))
                 for axis in AXES}
    gates.append({
        "gate": "3_support_manipulation_responds", "pass": any(axis_pass.values()),
        "detail": {axis: {"pass": passed,
                          "slope": main.get(axis, {}).get("slope")}
                   for axis, passed in axis_pass.items()},
        "note": "evaluated per model; §E2 requires it on a base or clean model. "
                "A trained-arm model failing it is a RESULT, not an instrument "
                "failure -- read it together with gate 2",
    })

    # 4 -- construction and alignment
    build_ok = anchor_report is None or int(anchor_report.get("n_rejected", 1)) == 0
    prefix_ok = score_manifest is None or bool(score_manifest.get("chat_prefix_ok"))
    gates.append({
        "gate": "4_parser_prefix_token_alignment",
        "pass": bool(build_ok and prefix_ok),
        "detail": {"anchor_validation_failures":
                   None if anchor_report is None else anchor_report.get("n_rejected"),
                   "chat_prefix_ok": None if score_manifest is None else
                   score_manifest.get("chat_prefix_ok")},
        "note": "every anchor passed its invariants at build time and the chat "
                "prefix matched the training-time one at score time",
    })

    # 5 -- floor / uninformative anchors are surfaced, not hidden
    flags = {axis: main.get(axis, {}).get("uninformative", {}) for axis in AXES}
    total = {axis: main.get(axis, {}).get("n_anchors", 0) for axis in AXES}
    share = {axis: (sum(flags[axis].values()) / total[axis]) if total[axis] else 0.0
             for axis in AXES}
    gates.append({
        "gate": "5_floor_anchors_flagged",
        "pass": all(value < 0.5 for value in share.values()),
        "detail": {"counts": flags, "share": share},
        "note": "anchors that cannot move (close already certain, or still "
                "impossible at full manipulation) are excluded and counted; "
                "more than half being uninformative means the anchor set, not "
                "the model, is the problem",
    })

    # 6 -- no systematic sign flip between the readout families
    flips = {}
    for axis in AXES:
        primary = summary.get("sm_primary", {}).get(axis, {}).get("slope", {})
        raw = summary.get("sm_raw", {}).get(axis, {}).get("slope", {})
        rank = summary.get("sm_primary", {}).get(axis, {}).get("close_rank_gain_r1", {})
        # Only readouts whose CI excludes zero get a vote.  A "sign flip"
        # between two quantities that are both indistinguishable from zero is
        # noise, and §9.5's rule is about a SYSTEMATIC reversal -- a rank gain
        # of -0.04 positions against a slope whose own CI spans zero says
        # nothing about either.
        decided = [block for block in (primary, raw, rank)
                   if block and block.get("mean") is not None
                   and block.get("ci_low") is not None
                   and (block["ci_low"] > 0 or block["ci_high"] < 0)]
        signs = {np.sign(block["mean"]) for block in decided}
        flips[axis] = {"primary": primary.get("mean"), "raw": raw.get("mean"),
                       "rank_gain": rank.get("mean"),
                       "n_decided": len(decided),
                       "consistent": len(signs) <= 1}
    gates.append({
        "gate": "6_no_systematic_sign_flip",
        "pass": all(value["consistent"] for value in flips.values()),
        "detail": flips,
        "note": "§9.5 only permits the phrase 'support conditioning fell' when "
                "the raw logit, the calibrated margin and a rank/sampler "
                "readout move the same way",
    })

    # 7 -- the hazard, when it was measured, is a measurement at all and points
    #      the same way as the margin.  Skipped (not failed) when NO_HAZARD=1.
    if summary.get("has_hazard"):
        hazard = summary.get("stop_hazard", {})
        other = max((_get(hazard, axis, "other_rate", "mean") or 0.0)
                    for axis in AXES)
        noncanonical = max((_get(hazard, axis, "noncanonical_rate", "mean") or 0.0)
                           for axis in AXES)
        agrees = summary.get("hazard_agrees_with_margin", {})
        decided = [value for value in agrees.values() if value is not None]
        # How much of the anchor set the hazard's own sampling state can reach.
        # The prefix carries no presence penalty there, so the close token sits
        # ~2.1 logits lower than `sm_primary` implies; if that pushes it out of
        # the sampler on most anchors, a flat hazard says nothing about the
        # model and the gate must not read as a disagreement.
        reachable = min((_get(hazard, axis, "reachable_share") or 0.0)
                        for axis in AXES)
        measurable = other < 0.5 and reachable >= 0.5
        gates.append({
            "gate": "7_hazard_is_measurable_and_agrees",
            "pass": bool(measurable and all(decided)) if decided else False,
            "detail": {"other_rate": other, "noncanonical_rate": noncanonical,
                       "agrees": agrees, "reachable_share": reachable,
                       "presence_state": protocol.HAZARD_PRESENCE_STATE,
                       "compared_against": HAZARD_MARGIN,
                       "measurable": measurable},
            "note": "a high `other` rate means the continuations stopped being "
                    "structural decisions, so the hazard is not a small effect "
                    "but no measurement; a low `reachable_share` means the same "
                    "thing for a different reason -- the hazard samples with "
                    "the prefix UNPENALISED (protocol.HAZARD_PRESENCE_STATE), "
                    "so it is compared against `sm_unpenalised`, and where the "
                    "close token is out of reach in that state a flat hazard is "
                    "about the state, not the model.  `agrees` is the §9.5 "
                    "requirement that what the model scores and what it does "
                    "move together",
        })
    return gates


# ------------------------------------------------------------------ CSV view

#: One row per model.  Kept narrow on purpose: the full distributions live in
#: the summary json, and a wide CSV invites reading a number without its CI.
CSV_FIELDS: List[str] = [
    "tag", "protocol_version", "n_anchors", "n_cells",
    # admissibility axis (ACI)
    "aci_slope", "aci_ci_low", "aci_ci_high", "aci_delta_r1",
    "aci_n_informative", "aci_rank_gain_r1", "aci_prob_gain_r1",
    "aci_baseline_margin", "aci_nsci", "aci_nsci_n_qualified",
    # evidence axis (ECI)
    "eci_slope", "eci_ci_low", "eci_ci_high", "eci_delta_r1",
    "eci_n_informative", "eci_rank_gain_r1", "eci_prob_gain_r1",
    "eci_baseline_margin", "eci_nsci", "eci_nsci_n_qualified",
    # sensitivity: the evidence axis substitutes over the whole document, so
    # the slope is repeated on the anchors whose manipulated and neutral arms
    # rewrote exactly as much text (`occurrence_gap == 0`)
    "eci_slope_matched", "eci_slope_matched_ci_low", "eci_n_matched",
    "eci_occurrence_gap_mean",
    # controls and floor
    "pc", "pc_ci_low", "pc_stop_effect", "pc_continue_effect",
    "neutral_drift_r1_admissibility", "neutral_drift_r1_evidence",
    "empty_margin", "empty_close_top1_rate",
    # raw-margin duplicates, for the §9.5 "no sign flip" reading
    "aci_slope_raw", "eci_slope_raw",
    # what the model DOES (empty when the hazard stage was skipped)
    "has_hazard",
    "aci_hazard_slope", "aci_hazard_base_rate", "aci_hazard_delta_r1",
    "eci_hazard_slope", "eci_hazard_base_rate", "eci_hazard_delta_r1",
    "hazard_other_rate", "hazard_noncanonical_rate", "hazard_agrees",
    # which of the four §E2 anchor types this run actually measured
    "built_anchor_types", "unbuilt_anchor_types",
    # gates
    "gates_passed", "gates_failed", "gate_fail_names",
]


def _get(block: Dict[str, Any], *path: str) -> Any:
    value: Any = block
    for step in path:
        if not isinstance(value, dict):
            return None
        value = value.get(step)
    return value


def summary_to_row(summary: Dict[str, Any],
                   gates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    main = summary.get("sm_primary", {})
    raw = summary.get("sm_raw", {})
    failed = [gate["gate"] for gate in gates if not gate["pass"]]
    row: Dict[str, Any] = {
        "tag": summary.get("tag"),
        "protocol_version": protocol.PROTOCOL_VERSION,
        "n_anchors": summary.get("n_anchors"),
        "n_cells": summary.get("n_cells"),
        "pc": _get(main, "admissibility", "pc", "mean"),
        "pc_ci_low": _get(main, "admissibility", "pc", "ci_low"),
        "pc_stop_effect": _get(main, "admissibility", "pc_stop_effect", "mean"),
        "pc_continue_effect": _get(main, "admissibility", "pc_continue_effect", "mean"),
        "neutral_drift_r1_admissibility":
            _get(main, "admissibility", "neutral_drift_r1.0", "mean"),
        "neutral_drift_r1_evidence":
            _get(main, "evidence", "neutral_drift_r1.0", "mean"),
        "empty_margin": _get(main, "zero_answer", "empty_margin", "mean"),
        "empty_close_top1_rate": _get(main, "zero_answer", "close_is_top1_rate", "mean"),
        "aci_slope_raw": _get(raw, "admissibility", "slope", "mean"),
        "eci_slope_raw": _get(raw, "evidence", "slope", "mean"),
        "has_hazard": summary.get("has_hazard"),
        "built_anchor_types": ";".join(
            _get(summary, "protocol", "built_anchor_types") or []),
        "unbuilt_anchor_types": ";".join(
            _get(summary, "protocol", "unbuilt_anchor_types") or []),
        "gates_passed": sum(1 for gate in gates if gate["pass"]),
        "gates_failed": len(failed),
        "gate_fail_names": ";".join(failed),
    }
    for axis, prefix in (("admissibility", "aci"), ("evidence", "eci")):
        block = main.get(axis, {})
        row[f"{prefix}_slope"] = _get(block, "slope", "mean")
        row[f"{prefix}_ci_low"] = _get(block, "slope", "ci_low")
        row[f"{prefix}_ci_high"] = _get(block, "slope", "ci_high")
        row[f"{prefix}_delta_r1"] = _get(block, "delta_r1.0", "mean")
        row[f"{prefix}_n_informative"] = block.get("n_informative")
        row[f"{prefix}_rank_gain_r1"] = _get(block, "close_rank_gain_r1", "mean")
        row[f"{prefix}_prob_gain_r1"] = _get(block, "close_sampler_prob_gain_r1", "mean")
        row[f"{prefix}_baseline_margin"] = _get(block, "baseline_margin", "mean")
        row[f"{prefix}_nsci"] = _get(block, "nsci", "mean")
        row[f"{prefix}_nsci_n_qualified"] = _get(block, "nsci", "n_qualified")
        if axis == "evidence":
            # Only this axis rewrites the document, so only this one has an
            # edit magnitude for the neutral arm to match.
            row[f"{prefix}_slope_matched"] = _get(
                block, "slope_occurrence_matched", "mean")
            row[f"{prefix}_slope_matched_ci_low"] = _get(
                block, "slope_occurrence_matched", "ci_low")
            row[f"{prefix}_n_matched"] = block.get("n_occurrence_matched")
            row[f"{prefix}_occurrence_gap_mean"] = _get(block, "occurrence_gap",
                                                        "mean")
        hazard = summary.get("stop_hazard", {}).get(axis, {})
        row[f"{prefix}_hazard_slope"] = _get(hazard, "slope", "mean")
        row[f"{prefix}_hazard_base_rate"] = _get(hazard, "base_rate", "mean")
        row[f"{prefix}_hazard_delta_r1"] = _get(hazard, "delta_r1.0", "mean")

    hazard_blocks = [block for block in summary.get("stop_hazard", {}).values()
                     if block.get("other_rate")]
    if hazard_blocks:
        row["hazard_other_rate"] = max(
            _get(block, "other_rate", "mean") or 0.0 for block in hazard_blocks)
        row["hazard_noncanonical_rate"] = max(
            _get(block, "noncanonical_rate", "mean") or 0.0
            for block in hazard_blocks)
    agrees = summary.get("hazard_agrees_with_margin", {})
    decided = [value for value in agrees.values() if value is not None]
    row["hazard_agrees"] = all(decided) if decided else None
    return row


# --------------------------------------------------------------------- main

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slopes, positive control, nSCI, strata and the §E2 gates "
                    "for one model's readouts.")
    parser.add_argument("--readouts", required=True)
    parser.add_argument("--hazard", default=None,
                        help="<tag>_hazard.jsonl.  Folded into the same cells "
                             "as the readouts, so the margin and the behaviour "
                             "are summarised over identical anchors.  Passing "
                             "it makes the join MANDATORY: a hazard file that "
                             "matches nothing is an error, not an empty column.")
    parser.add_argument("--hazard_manifest", default=None,
                        help="<tag>_hazard.done.json.  REQUIRED with --hazard: "
                             "its identity must agree with the scorer's, and "
                             "its `arms` say how many cells the merge is "
                             "supposed to cover.")
    parser.add_argument("--allow_unverified_hazard", action="store_true",
                        help="Merge a hazard file with no manifest.  Debugging "
                             "only: nothing then proves the file is this "
                             "model's, and cell ids match across models.")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--anchor_report", default=None,
                        help="anchors.jsonl.report.json, for gate 4.")
    parser.add_argument("--score_manifest", default=None,
                        help="<tag>_readouts.done.json, for gate 4.")
    parser.add_argument("--task_json", default=None,
                        help="Registry metadata for this model, so the CSV and "
                             "the report can group by data variant.")
    parser.add_argument("--csv", default=None,
                        help="Append the one-row summary here as well.")
    parser.add_argument("--bootstrap", type=int, default=protocol.BOOTSTRAP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tag = args.tag or Path(args.readouts).stem.replace("_readouts", "")

    anchor_report = read_json(args.anchor_report) if args.anchor_report and \
        Path(args.anchor_report).is_file() else None
    score_manifest = read_json(args.score_manifest) if args.score_manifest and \
        Path(args.score_manifest).is_file() else None
    hazard_manifest = read_json(args.hazard_manifest) if args.hazard_manifest \
        and Path(args.hazard_manifest).is_file() else None

    # The hazard file must belong to THIS measurement before its rows are
    # allowed to join.  Merging is a cell_id lookup, and cell ids are stable
    # across models -- so a hazard file from the wrong model would join
    # perfectly and put another model's behaviour beside this one's margins.
    if args.hazard and hazard_manifest:
        wanted = (score_manifest or {}).get("identity", {})
        found = hazard_manifest.get("identity", {})
        clash = [key for key in ("anchors_sha256", "model_path",
                                 "protocol_version", "prompt_rendered_sha256")
                 if wanted.get(key) is not None and found.get(key) != wanted[key]]
        if clash:
            raise SystemExit(
                f"[{tag}] the hazard was measured under a different identity "
                f"({', '.join(clash)}); its rows would put another "
                "measurement's behaviour beside these margins")
        if hazard_manifest.get("tag") not in (None, tag):
            raise SystemExit(
                f"[{tag}] hazard manifest is tagged "
                f"{hazard_manifest.get('tag')!r}")
    elif args.hazard and not args.allow_unverified_hazard:
        raise SystemExit(
            f"[{tag}] --hazard needs --hazard_manifest.  Cell ids are the same "
            "across models, so an unidentified hazard file joins perfectly and "
            "puts another model's behaviour beside these margins; without the "
            "manifest neither that nor completeness can be checked.\n"
            f"  expected: {Path(args.hazard).with_suffix('.done.json')}\n"
            "  --allow_unverified_hazard overrides this, for debugging only.")
    elif args.hazard:
        # A warning is not a guard: the run went on to write the summary and
        # append the CSV row anyway, and the collector took it.  The bypass is
        # for looking at a file by hand, so it may not produce a shippable
        # artefact -- the summary is stamped unverified and the collector
        # refuses it, and it may not touch the unified CSV at all.
        if args.csv:
            raise SystemExit(
                f"[{tag}] --allow_unverified_hazard cannot be combined with "
                "--csv: an unidentified hazard file must not reach the unified "
                "table.  Drop --csv to inspect it, or supply "
                "--hazard_manifest.")
        LOG.warning("[%s] --allow_unverified_hazard: nothing proves this hazard "
                    "file is this model's, and completeness is unchecked.  The "
                    "summary is stamped hazard_verified=false and the collector "
                    "will refuse it.", tag)

    views = load_views(
        args.readouts, args.hazard,
        hazard_arms=(hazard_manifest or {}).get("arms"),
        strict_hazard=bool(args.hazard),
        expect_cells=(score_manifest or {}).get("n_cells"),
        expect_presence_state=protocol.HAZARD_PRESENCE_STATE,
        expect_tag=tag,
        readouts_sha256=(score_manifest or {}).get("readouts_sha256"),
        hazard_sha256=(hazard_manifest or {}).get("hazard_sha256"))
    # The manifest's own count has to be the whole anchor set, not just
    # self-consistent: a score run that stopped after 40 cells writes a manifest
    # saying 40, and every check downstream would agree with it.
    expected_total = (anchor_report or {}).get("n_variants")
    if expected_total and score_manifest:
        recorded = score_manifest.get("n_cells")
        if recorded != expected_total:
            raise SystemExit(
                f"[{tag}] the score manifest records {recorded} cell(s) but the "
                f"anchor set has {expected_total}; this model was not scored on "
                "the whole anchor set")
    if not views:
        raise SystemExit(f"no readouts in {args.readouts}")
    summary = summarize(views, tag=tag, n_boot=args.bootstrap)
    if args.hazard:
        summary["hazard_verified"] = bool(hazard_manifest)

    # Provenance the collector needs to prove every model was measured with
    # the SAME instrument; without it a mixed CSV looks perfectly healthy.
    if score_manifest:
        summary["anchors_sha256"] = (score_manifest.get("identity", {})
                                     .get("anchors_sha256"))
        summary["score_minutes"] = score_manifest.get("score_minutes")
        # WHICH score run this summary read.  A summary is derived data; without
        # this, a re-scored model keeps the summary of the readouts it replaced.
        summary["score_identity"] = score_manifest.get("identity", {})
        summary["prompt_rendered_sha256"] = (
            score_manifest.get("prompt_rendered_sha256")
            or score_manifest.get("identity", {}).get("prompt_rendered_sha256"))
    if hazard_manifest:
        summary["hazard_arms"] = hazard_manifest.get("arms")
        summary["hazard_minutes"] = hazard_manifest.get("hazard_minutes")
        summary["hazard_identity"] = hazard_manifest.get("identity", {})
    if args.task_json and Path(args.task_json).is_file():
        summary["task"] = read_json(args.task_json)

    gates = instrument_gates(summary, anchor_report=anchor_report,
                             score_manifest=score_manifest)
    summary["gates"] = gates
    summary["gates_passed"] = all(gate["pass"] for gate in gates)

    out = Path(args.output_root) / "analysis" / f"{sanitize_filename(tag)}_e2_summary.json"
    write_json(out, summary)

    row = summary_to_row(summary, gates)
    if args.csv:
        from .io_utils import append_csv_row
        append_csv_row(args.csv, row, CSV_FIELDS)

    print("=" * 72)
    print(f"{tag}   anchors={summary['n_anchors']}  cells={summary['n_cells']}")
    for axis, prefix in (("admissibility", "ACI"), ("evidence", "ECI")):
        block = summary["sm_primary"][axis]
        slope = block.get("slope") or {}
        print(f"  {prefix}-logit  slope={_fmt(slope.get('mean'))} "
              f"[{_fmt(slope.get('ci_low'))}, {_fmt(slope.get('ci_high'))}] nat "
              f"(n={block.get('n_informative')})")
    pc = summary["sm_primary"]["admissibility"].get("pc") or {}
    print(f"  PC           {_fmt(pc.get('mean'))} "
          f"[{_fmt(pc.get('ci_low'))}, {_fmt(pc.get('ci_high'))}] nat")
    print("  gates: " + ("ALL PASS" if summary["gates_passed"] else "FAILED " +
                         ",".join(g["gate"] for g in gates if not g["pass"])))
    print("=" * 72)
    print(f"summary -> {out}")
    return 0 if summary["gates_passed"] else 2


def _fmt(value: Optional[float]) -> str:
    return "  n/a " if value is None else f"{value:+.3f}"


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
