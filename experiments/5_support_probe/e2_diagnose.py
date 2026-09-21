# coding=utf-8
"""Read-only diagnostics for an E2 result directory.

Nothing here writes, rebuilds or re-measures anything: it reads the summaries
and the raw readouts and prints the numbers needed to tell an instrument
problem from a model property.  Four sections, in the order they answer
questions:

1. provenance     -- which anchors, which protocol, which analysis version
2. gates          -- gate 1's drift and gate 2's decomposition, with CIs
3. pc_criterion   -- the positive control computed under BOTH floor criteria
                     (pre-filter, which the analysis uses now, and the
                     post-sampler one it used before) on the SAME readouts.
                     This is what separates "the control text changed" from
                     "the anchor set the control is averaged over changed".
4. pc_quintile    -- the positive control by quintile of the untouched margin.
                     A control that only works where the margin is not already
                     saturated is a control with a range, not a broken one; a
                     control that fails in every quintile is broken.

    RESULT_ROOT=./outputs/support_probe_v12 python scripts/e2_diagnose.py
    python scripts/e2_diagnose.py --result_root ./outputs/support_probe_v12
    python scripts/e2_diagnose.py --sections pc_quintile
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

AXES = ("admissibility", "evidence")
#: (label, per-cell field the floor/ceiling question is asked on)
CRITERIA = (("prefilter(now)", "close_prob_pre_filter"),
            ("sampler(before)", "close_sampler_prob"))
CEILING = 0.999
FLOOR = 1e-6


def log(message=""):
    print(message, flush=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean(values):
    return statistics.mean(values) if values else float("nan")


# ------------------------------------------------------------------ loading

def load_anchors(path):
    """{anchor_id: {"t": anchor_type, "x": axis, "c": {(arm, level): row}}}"""
    anchors = {}
    n_rows = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            anchor = anchors.setdefault(
                row["anchor_id"],
                {"t": row.get("anchor_type"), "x": None, "c": {}})
            if row.get("axis") not in (None, "none"):
                anchor["x"] = row["axis"]
            anchor["c"][(row.get("arm"), float(row.get("level", 0.0)))] = row
    return anchors, n_rows


def informative(cells, field):
    """The floor/ceiling question, asked on `field`.

    Same thresholds `analyze.AnchorView.informative` uses; kept here as a plain
    function so the two criteria can be compared on one set of readouts."""
    base = cells.get(("base", 0.0))
    if base is None or base.get(field) is None:
        return False
    if float(base[field]) >= CEILING:
        return False
    full = cells.get(("manip", 1.0))
    if full is not None and float(full.get(field) or 0.0) <= FLOOR:
        return False
    if float(base[field]) <= FLOOR and full is None:
        return False
    return True


def pc_parts(cells):
    """(PC, stop_effect, continue_effect) on sm_primary, or None."""
    stop = cells.get(("pc_stop", 0.0))
    cont = cells.get(("pc_continue", 0.0))
    neutral = cells.get(("pc_neutral", 0.0))
    if stop is None or cont is None or neutral is None:
        return None
    return (stop["sm_primary"] - cont["sm_primary"],
            stop["sm_primary"] - neutral["sm_primary"],
            neutral["sm_primary"] - cont["sm_primary"])


def readout_files(root):
    return sorted(glob.glob(os.path.join(root, "readouts", "*_readouts.jsonl")))


def summary_files(root):
    return sorted(glob.glob(os.path.join(root, "analysis", "*_e2_summary.json")))


# ----------------------------------------------------------------- sections

def section_provenance(root):
    log("=" * 72)
    log("1. provenance")
    report = os.path.join(root, "anchors", "anchors.jsonl.report.json")
    if os.path.isfile(report):
        data = read_json(report)
        log("  anchors      %s" % str(data.get("anchors_sha256"))[:12])
        log("  protocol     %s" % (data.get("protocol") or {}).get("protocol_version"))
        log("  by type      %s   rejected=%s skipped=%s"
            % (data.get("anchors_by_type"), data.get("n_rejected"),
               data.get("n_skipped_by_validation")))
        log("  donor length %s   max_delta_allowed=%s"
            % (data.get("donor_length_delta"), data.get("max_length_delta_allowed")))
        risk = data.get("evidence_substitution_risk") or {}
        if risk:
            log("  evidence risk share_embedded=%s matched_anchors=%s"
                % (risk.get("share_embedded"), risk.get("occurrence_matched_anchors")))
    else:
        log("  (no anchor report at %s)" % report)
    for path in summary_files(root):
        s = read_json(path)
        log("  %-24s protocol=%s schema=%s bootstrap=%s hazard=%s"
            % (s.get("tag"), (s.get("protocol") or {}).get("protocol_version"),
               s.get("summary_schema"), s.get("bootstrap"), s.get("has_hazard")))


def section_gates(root):
    log("=" * 72)
    log("2. gates 1 and 2, with the numbers behind the verdict")
    for path in summary_files(root):
        s = read_json(path)
        gates = {g["gate"]: g for g in s.get("gates", [])}
        log("-- %s" % s.get("tag"))
        failed = [name for name, g in gates.items() if not g.get("pass")]
        log("   failed: %s" % (", ".join(sorted(failed)) or "none"))

        gate1 = gates.get("1_neutral_effect_near_zero")
        if gate1:
            log("   gate1 neutral drift (should be ~0; threshold |mean|<0.25 or CI covers 0)")
            for name in sorted(gate1.get("detail", {})):
                d = gate1["detail"][name] or {}
                log("     %-22s %+7.3f  [%+.3f, %+.3f]"
                    % (name, d.get("mean", float("nan")),
                       d.get("ci_low", float("nan")), d.get("ci_high", float("nan"))))

        gate2 = gates.get("2_positive_control_direction")
        if gate2:
            log("   gate2 positive control (all three should be > 0)")
            for name in sorted(gate2.get("detail", {})):
                d = gate2["detail"][name]
                if not d:
                    log("     %-30s n/a" % name)
                    continue
                log("     %-30s %+7.3f  ci_low %+.3f"
                    % (name, d.get("mean", float("nan")),
                       d.get("ci_low", float("nan"))))

        for axis in AXES:
            block = (s.get("sm_primary") or {}).get(axis) or {}
            slope = block.get("slope") or {}
            log("   %-14s n_inf=%-4s reachable=%-8s slope=%+.3f [%+.3f, %+.3f]"
                % (axis, block.get("n_informative"),
                   block.get("sampler_reachable_share"),
                   slope.get("mean", float("nan")),
                   slope.get("ci_low", float("nan")),
                   slope.get("ci_high", float("nan"))))
            strata = block.get("by_baseline_quintile") or {}
            if strata:
                cells = []
                for key in sorted(strata):
                    value = (strata[key] or {}).get("mean")
                    cells.append("Q%s=%s" % (int(key) + 1,
                                             "n/a" if value is None else "%+.2f" % value))
                log("     slope by baseline quintile: %s" % "  ".join(cells))


def section_pc_criterion(root):
    log("=" * 72)
    log("3. the positive control under BOTH floor criteria, same readouts")
    log("   (a difference here is the ANCHOR SET changing, not the control text)")
    for path in readout_files(root):
        tag = os.path.basename(path).replace("_readouts.jsonl", "")
        anchors, n_rows = load_anchors(path)
        log("-- %s  rows=%d anchors=%d" % (tag, n_rows, len(anchors)))
        for axis in AXES:
            for label, field in CRITERIA:
                pcs, stops, conts = [], [], []
                for anchor in anchors.values():
                    if anchor["t"] != axis or anchor["x"] != axis:
                        continue
                    if not informative(anchor["c"], field):
                        continue
                    parts = pc_parts(anchor["c"])
                    if parts is None:
                        continue
                    pcs.append(parts[0])
                    stops.append(parts[1])
                    conts.append(parts[2])
                if not pcs:
                    log("   %-14s %-16s n=0" % (axis, label))
                    continue
                log("   %-14s %-16s n=%-4d PC=%+.3f  stop_eff=%+.3f  cont_eff=%+.3f"
                    % (axis, label, len(pcs), mean(pcs), mean(stops), mean(conts)))


def section_pc_quintile(root):
    log("=" * 72)
    log("4. the positive control by quintile of the untouched margin")
    log("   Q1 = most negative base margin (most saturated), Q5 = closest to 0.")
    log("   If PC turns positive as the margin approaches 0, the control works")
    log("   where it CAN work and the pooled negative is a saturation artefact.")
    log("   If every quintile is negative, the control itself is wrong.")
    for path in readout_files(root):
        tag = os.path.basename(path).replace("_readouts.jsonl", "")
        anchors, n_rows = load_anchors(path)
        dropped = {"type": 0, "axis": 0, "no_base": 0, "no_pc": 0}
        rows = []
        for anchor in anchors.values():
            if anchor["t"] not in AXES:
                dropped["type"] += 1
                continue
            if anchor["x"] != anchor["t"]:
                dropped["axis"] += 1
                continue
            base = anchor["c"].get(("base", 0.0))
            if base is None:
                dropped["no_base"] += 1
                continue
            parts = pc_parts(anchor["c"])
            if parts is None:
                dropped["no_pc"] += 1
                continue
            rows.append((base["sm_primary"], parts[0], parts[1], parts[2],
                         1 if base.get("close_survives_top_p") else 0,
                         anchor["t"]))
        log("-- %s  rows=%d anchors=%d usable=%d dropped=%s"
            % (tag, n_rows, len(anchors), len(rows), dropped))
        if not rows:
            sample = next(iter(anchors.values()), None)
            if sample is not None:
                log("   sample anchor: t=%r x=%r arms=%s"
                    % (sample["t"], sample["x"],
                       sorted(str(k) for k in sample["c"])))
            continue
        rows.sort(key=lambda item: item[0])
        width = len(rows) // 5
        log("   %-4s %5s %9s %9s %10s %10s %7s"
            % ("q", "n", "baseSM", "PC", "stop_eff", "cont_eff", "reach"))
        for index in range(5):
            chunk = (rows[index * width:(index + 1) * width] if index < 4
                     else rows[4 * width:])
            if not chunk:
                continue
            log("   %-4s %5d %9.2f %+9.3f %+10.3f %+10.3f %7.2f"
                % ("Q%d" % (index + 1), len(chunk),
                   mean([r[0] for r in chunk]), mean([r[1] for r in chunk]),
                   mean([r[2] for r in chunk]), mean([r[3] for r in chunk]),
                   mean([r[4] for r in chunk])))


SECTIONS = {
    "provenance": section_provenance,
    "gates": section_gates,
    "pc_criterion": section_pc_criterion,
    "pc_quintile": section_pc_quintile,
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result_root", default=os.environ.get("RESULT_ROOT"),
                        help="default: $RESULT_ROOT")
    parser.add_argument("--sections", default="all",
                        help="comma-separated: %s" % ",".join(SECTIONS))
    args = parser.parse_args(argv)

    if not args.result_root:
        parser.error("pass --result_root or set RESULT_ROOT")
    root = args.result_root
    if not os.path.isdir(root):
        log("FATAL: no such directory: %s" % root)
        return 1
    log("result root: %s" % root)
    if not readout_files(root):
        log("WARNING: no readouts under %s/readouts" % root)

    wanted = (list(SECTIONS) if args.sections == "all"
              else [v.strip() for v in args.sections.split(",") if v.strip()])
    unknown = [name for name in wanted if name not in SECTIONS]
    if unknown:
        parser.error("unknown section(s): %s" % unknown)
    for name in wanted:
        SECTIONS[name](root)
    log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
