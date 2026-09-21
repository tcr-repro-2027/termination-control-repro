# coding=utf-8
"""Rebuild `e2_metrics.csv` from the per-model summaries, and write the verdict.

Two outputs:

* `e2_metrics.csv` -- one row per model, rebuilt from `analysis/*.json` so it is
  ordered, complete and free of duplicates from hand re-runs;
* `e2_instrument_report.md` -- the §E2 pass criteria, model by model, with the
  one distinction that matters for reading them:

      gates 1, 2, 4, 5, 6 are about the INSTRUMENT.  If they fail on the
      calibration model, nothing downstream is interpretable and E3 must not
      start.

      gate 3 ("the support manipulation does something") is about the MODEL.
      §E2 requires it on a base or clean model; a trained arm failing it while
      passing gate 2 is a RESULT -- support conditioning is gone while general
      instruction following is intact, which is exactly H3's shape.

The comparability checks are the same discipline as E1's: every model must have
been measured on the same anchors, the same prompt and the same protocol, or
the ACI/ECI numbers are not on one scale.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcr.support_probe.analyze import CSV_FIELDS, summary_to_row              # noqa: E402
from tcr.support_probe.io_utils import (                                      # noqa: E402
    FileLock, read_csv_rows, read_json, write_csv,
)

#: Reading order: the calibration references first, then the controlled arms.
VARIANT_ORDER = {name: index for index, name in enumerate(
    ("none", "base", "dedup", "filter", "clean", "cleanv2", "benign_input",
     "generic_noise", "obr_p10", "obr_p15", "obr", "isc_a", "isc_e", "isc_ae"))}

#: Gates whose failure means the INSTRUMENT is not usable.  Gate 7 is here even
#: though it also reads the model: a high `other` rate is not a small effect, it
#: is the continuations having stopped being structural decisions at all.  It is
#: the only one that can be legitimately ABSENT (NO_HAZARD=1 skips it), so it
#: renders as three states, not two -- a run that never measured the behaviour
#: must not look the same as one that measured it and passed.
INSTRUMENT_GATES = ("1_neutral_effect_near_zero", "2_positive_control_direction",
                    "4_parser_prefix_token_alignment", "5_floor_anchors_flagged",
                    "6_no_systematic_sign_flip", "7_hazard_is_measurable_and_agrees")
GATE_HEADERS = ("1 中性≈0", "2 正向控制", "4 解析/对齐", "5 floor 标记",
                "6 无反号", "7 短续写")
MODEL_GATES = ("3_support_manipulation_responds",)


def build_rows(analysis_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(analysis_dir.glob("*_e2_summary.json")):
        summary = read_json(path)
        row = summary_to_row(summary, summary.get("gates", []))
        row["_variant"] = VARIANT_ORDER.get(
            str(summary.get("task", {}).get("data_variant", "")), 99)
        row["_tag"] = str(row.get("tag"))
        rows.append(row)
    rows.sort(key=lambda row: (row["_variant"], row["_tag"]))
    for row in rows:
        row.pop("_variant", None)
        row.pop("_tag", None)
    return rows


#: Everything that has to be the same, or the ACI/ECI numbers are not on one
#: scale.  The prompt digest and the resample count belong here for the same
#: reason the anchor hash does: a margin measured under a different instruction
#: is a different margin, and a CI over a different number of resamples is a
#: different interval.
SHARED_FIELDS = ("protocol_version", "summary_schema", "anchors_sha256",
                 "n_anchors", "prompt_rendered_sha256", "bootstrap")


def comparability(analysis_dir: Path) -> Tuple[List[str], List[str]]:
    """Every model must have been measured with the same instrument.

    Returns `(problems, notes)`.  The split is what makes `--strict` usable:
    differing anchors, prompt, protocol, analysis schema or resample count make
    the table incomparable and must stop a run; running the short continuations
    on only the key arms is a documented choice (§5) and must not.
    """
    problems: List[str] = []
    notes: List[str] = []
    seen: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    hazard_tags: List[str] = []
    no_hazard_tags: List[str] = []
    for path in sorted(analysis_dir.glob("*_e2_summary.json")):
        summary = read_json(path)
        tag = str(summary.get("tag"))
        values = {
            "protocol_version": (summary.get("protocol") or {}).get("protocol_version"),
            "summary_schema": summary.get("summary_schema"),
            "anchors_sha256": summary.get("anchors_sha256"),
            "n_anchors": summary.get("n_anchors"),
            "prompt_rendered_sha256": summary.get("prompt_rendered_sha256"),
            "bootstrap": summary.get("bootstrap"),
        }
        for field in SHARED_FIELDS:
            seen[field][str(values[field])].append(tag)
        # Whether a model has a hazard column is a legitimate choice (§5's
        # "短续写只补关键臂"), so it is reported, not flagged.  The ARMS used by
        # the models that do have one are not: two arm sets make two different
        # hazard measures wearing the same column name.
        if summary.get("has_hazard"):
            hazard_tags.append(tag)
            seen["hazard_arms"][str(summary.get("hazard_arms"))].append(tag)
            seen["hazard_presence_state"][
                str((summary.get("stop_hazard", {}).get("evidence", {}) or {})
                    .get("presence_state"))].append(tag)
            # `--allow_unverified_hazard` is a hand-inspection door, not a way
            # to ship a number: nothing proves that hazard file is this model's.
            if summary.get("hazard_verified") is False:
                problems.append(
                    f"{tag}: its hazard was merged WITHOUT a manifest "
                    "(--allow_unverified_hazard); nothing proves the file is "
                    "this model's.  Re-run the analysis with --hazard_manifest "
                    "or delete the summary")
        else:
            no_hazard_tags.append(tag)
    for field, values in seen.items():
        if len(values) > 1:
            detail = "; ".join(f"{value or '<empty>'}: {len(tags)} model(s) "
                               f"e.g. {tags[0]}"
                               for value, tags in sorted(values.items(),
                                                         key=lambda kv: -len(kv[1])))
            problems.append(f"{field} differs across models -- {detail}")
    if hazard_tags and no_hazard_tags:
        notes.append(
            f"hazard coverage is partial: {len(hazard_tags)} model(s) have a "
            f"behaviour column and {len(no_hazard_tags)} do not "
            f"(e.g. {no_hazard_tags[0]}); gate 7 is absent, not passed, on those"
            " -- read the hazard columns only within the covered set")
    return problems, notes


def write_report(path: Path, analysis_dir: Path, problems: List[str],
                 notes: Sequence[str] = ()) -> None:
    lines: List[str] = ["# E2 仪器与支持条件化报告", ""]
    summaries = [read_json(p) for p in sorted(analysis_dir.glob("*_e2_summary.json"))]
    if not summaries:
        lines.append("（还没有任何模型的分析结果）")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines += ["## 1. 仪器闸门（关于工具，不是关于假设）", "",
              "这几道闸门只要在**校准模型**（未训练模型或 cleanv2）上失败，",
              "后面所有数字都不可解释，E3 不能开始。",
              "闸门 7 只在跑了短续写时出现：`–` 表示**没测**（NO_HAZARD=1），不是通过。",
              "",
              "| 模型 | " + " | ".join(GATE_HEADERS) + " |",
              "|---" * (len(GATE_HEADERS) + 1) + "|"]
    for summary in summaries:
        gates = {gate["gate"]: gate for gate in summary.get("gates", [])}
        cells = "".join(
            f" {'–' if name not in gates else ('✓' if gates[name].get('pass') else '✗')} |"
            for name in INSTRUMENT_GATES)
        lines.append(f"| {summary.get('tag')} |{cells}")

    skipped = [str(s.get("tag")) for s in summaries
               if not any(g["gate"] == INSTRUMENT_GATES[-1]
                          for g in s.get("gates", []))]
    if skipped:
        lines += ["", f"闸门 7 在 {len(skipped)} 个模型上**未测**（无短续写）："
                      f"{', '.join(skipped[:6])}"
                      + (" …" if len(skipped) > 6 else "")]

    lines += ["", "## 2. 支持响应（关于模型，是结果不是故障）", "",
              "闸门 3 = 操纵是否让 StopMargin 动。校准模型上必须通过；",
              "**受训臂不通过而闸门 2 通过，说明支持条件化没了而一般指令跟随还在**",
              "—— 这正是 H3 的形状。", "",
              "证据轴的替换是**全文字符串替换**（中文没有词边界，见 README §6），",
              "所以 pooled ECI 里可能混进普通散文改写、以及中性臂改动量不等的影响。",
              "**ECI 必须连同「等量子集」一起读**：那是 manip/neutral 在每一档改写的",
              "mention 数完全相等的锚点。两者同号同量级，这条效应才扛得住严格匹配；",
              "只报 pooled ECI 并称其为纯证据轴效应是不成立的。", "",
              "| 模型 | ACI-logit (nat) | ECI-logit (nat) | ECI 等量子集 | PC (nat) | 闸门3 |",
              "|---|---|---|---|---|---|"]
    for summary in summaries:
        main = summary.get("sm_primary", {})
        gates = {gate["gate"]: gate for gate in summary.get("gates", [])}

        def cell(axis: str) -> str:
            block = (main.get(axis, {}) or {}).get("slope") or {}
            if block.get("mean") is None:
                return "n/a"
            return (f"{block['mean']:+.3f} "
                    f"[{block.get('ci_low', float('nan')):+.3f}, "
                    f"{block.get('ci_high', float('nan')):+.3f}]")

        evidence = main.get("evidence", {}) or {}
        matched = evidence.get("slope_occurrence_matched") or {}
        matched_text = "未记录" if matched.get("mean") is None else (
            f"{matched['mean']:+.3f} "
            f"[{matched.get('ci_low', float('nan')):+.3f}, "
            f"{matched.get('ci_high', float('nan')):+.3f}] "
            f"(n={evidence.get('n_occurrence_matched')})")
        pc = (main.get("admissibility", {}) or {}).get("pc") or {}
        pc_text = "n/a" if pc.get("mean") is None else \
            f"{pc['mean']:+.3f} [{pc.get('ci_low', float('nan')):+.3f}, ...]"
        mark = "✓" if gates.get(MODEL_GATES[0], {}).get("pass") else "✗"
        lines.append(f"| {summary.get('tag')} | {cell('admissibility')} | "
                     f"{cell('evidence')} | {matched_text} | {pc_text} | {mark} |")

    if any(summary.get("has_hazard") for summary in summaries):
        lines += ["", "## 3. 短续写（模型打的分 vs 模型真做的事）", "",
                  "闸门 7 的两个来源：`other` 率高说明续写不再是可辨认的结构决策，",
                  "那不是「效应小」而是**根本没测到**；`同向` 是 §9.5 要求的",
                  "「打的分和行为一致」。", "",
                  "| 模型 | ACI hazard 斜率 | ECI hazard 斜率 | base 停止率 | other 率 | 非规范率 | 同向 |",
                  "|---|---|---|---|---|---|---|"]
        for summary in summaries:
            if not summary.get("has_hazard"):
                continue
            hazard = summary.get("stop_hazard", {}) or {}

            def mean_of(axis: str, name: str):
                return ((hazard.get(axis, {}) or {}).get(name) or {}).get("mean")

            def worst(name: str) -> str:
                values = [mean_of(axis, name) for axis in ("admissibility", "evidence")]
                values = [value for value in values if value is not None]
                return "n/a" if not values else f"{max(values):.3f}"

            def signed(axis: str, name: str) -> str:
                value = mean_of(axis, name)
                return "n/a" if value is None else f"{value:+.3f}"

            agrees = summary.get("hazard_agrees_with_margin", {}) or {}
            decided = [v for v in agrees.values() if v is not None]
            lines.append(
                f"| {summary.get('tag')} | "
                f"{signed('admissibility', 'slope')} | "
                f"{signed('evidence', 'slope')} | "
                f"{signed('admissibility', 'base_rate')} | "
                f"{worst('other_rate')} | {worst('noncanonical_rate')} | "
                f"{('✓' if all(decided) else '✗') if decided else '–'} |")

    lines += ["", "## 4. 零答案参照（EmptyMargin）", "",
              "答案确实应该是 `[]` 时，模型在 `[` 之后的收尾余量。",
              "它是前面那些 margin 的参照点：模型「知道无话可说」长什么样。", "",
              "| 模型 | EmptyMargin | close 为 top-1 的比例 |", "|---|---|---|"]
    for summary in summaries:
        zero = (summary.get("sm_primary", {}) or {}).get("zero_answer", {}) or {}
        margin = (zero.get("empty_margin") or {}).get("mean")
        top1 = (zero.get("close_is_top1_rate") or {}).get("mean")
        lines.append(f"| {summary.get('tag')} | "
                     f"{'n/a' if margin is None else f'{margin:+.3f}'} | "
                     f"{'n/a' if top1 is None else f'{top1:.3f}'} |")

    if problems:
        lines += ["", "## 5. 可比性问题", ""]
        lines += [f"- {problem}" for problem in problems]
    else:
        lines += ["", "## 5. 可比性", "",
                  "所有模型使用同一套锚点、同一个 prompt、同一个协议版本、",
                  "同一版分析代码、同样的 bootstrap 次数。"]
    if notes:
        lines += ["", "读表提示：", ""]
        lines += [f"- {note}" for note in notes]

    lines += ["", "---", "",
              "读法提醒（§9.5）：只有 raw logit、校准后的 margin 和至少一个 ",
              "rank/sampler 读出**同向**时，才能说「支持条件化下降」。",
              "单看斜率不够。"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 on a comparability PROBLEM (not on a "
                             "note).  run_e2_all.sh always passes it.")
    parser.add_argument("--keep_orphan_rows", action="store_true",
                        help="Keep CSV rows whose summary is gone.  Off by "
                             "default: a row nothing can reproduce is worse "
                             "than a missing one, and an ONLY= run on an old "
                             "RESULT_ROOT is how they appear.")
    args = parser.parse_args()

    root = Path(args.result_root)
    analysis_dir = root / "analysis"
    rows = build_rows(analysis_dir)
    if not rows:
        print(f"no summaries under {analysis_dir}")
        return 1

    target = Path(args.csv) if args.csv else root / "e2_metrics.csv"
    # Decide BEFORE writing.  Writing the table and then returning 1 leaves an
    # `e2_metrics.csv` on disk that the exit code says is not valid -- and the
    # file outlives the exit code, which is what a reader finds three weeks
    # later.  Under --strict an incomparable table is not written at all; the
    # REPORT still is, because that is where the reason lives.
    with FileLock(target):
        rows = build_rows(analysis_dir)
        known = {str(row["tag"]) for row in rows}
        # Rows in the CSV with no summary behind them.  The CSV is REBUILT from
        # the summaries, so these are models whose analysis was deleted or never
        # re-run -- numbers nobody can reproduce.  An `ONLY=` run on an existing
        # RESULT_ROOT is exactly how they turn up, so they are dropped by
        # default and only kept on request.
        orphans = [row for row in read_csv_rows(target)
                   if str(row.get("tag")) not in known]
        carried = orphans if args.keep_orphan_rows else []

        problems, notes = comparability(analysis_dir)
        for tag in (str(row.get("tag")) for row in orphans):
            message = (f"{tag}: had a CSV row with no summary behind it; "
                       + ("KEPT (--keep_orphan_rows) -- nothing can reproduce it"
                          if args.keep_orphan_rows else
                          "dropped.  Re-run its analysis to bring it back"))
            (problems if args.keep_orphan_rows else notes).append(message)

        blocked = bool(problems) and args.strict
        if not blocked:
            write_csv(target, rows + carried, CSV_FIELDS)
    if blocked:
        print(f"NOT written (--strict, {len(problems)} comparability "
              f"problem(s)): {target}")
    else:
        print(f"wrote {len(rows) + len(carried)} row(s) -> {target}")

    report_path = Path(args.report) if args.report else root / "e2_instrument_report.md"
    write_report(report_path, analysis_dir, problems, notes)
    print(f"wrote {report_path}")

    for note in notes:
        print(f"  [note] {note}")
    if problems:
        print("\n---- comparability problems ----")
        for problem in problems:
            print(f"  [!] {problem}")
        if args.strict:
            return 1
    else:
        print("comparability checks passed: one anchor set, one protocol, "
              "one analysis version")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
