# coding: utf-8
"""R0: rebuild reuse, exit and sustained capture from the responses we already have.

Scientific question (H2 / contribution C2): a model that repeats more may be
reaching the reuse state more often, or leaving it less often once there, or
both.  The E1 CSV cannot tell those apart -- ``episodes_per_response`` and
``per_episode_capture_hazard`` multiply to the answer rate, but a product is not
a pair of independent causal channels.  P0d's interleaved competing-risk axis
is what separates them, and this entry runs it over every pair the paper needs,
in one pass, with no new generation and no GPU.

Three things happen here:

1. **Pairing.**  Each E1 tag pair becomes the M0/M1 directory the 01b analysers
   validate on load (`pc.pairing`).  M0 is always the reference arm, so a
   positive M1-M0 difference always reads as "the treatment made it worse".
2. **P0d for all nine pairs, P0c for the two that feed a prefix pool.**  The
   frozen analysers are invoked as-is; nothing about the episode state machine,
   the tail pooling or the Shapley decomposition is reimplemented here.
3. **Two CPU statistics P0d does not produce**, both asked for by the plan:
   whether a captured response still emits anything new afterwards, and the
   per-block table R1 selects its prefixes from.

What the decomposition is not
-----------------------------
The Shapley components are a descriptive standardisation of an observed
difference on a common risk range.  They are not "the fraction of the training
effect mediated by stopping": censoring, state composition and the tail model
all move them, and P0d reports the raw CIF difference next to the decomposed
one precisely so the gap stays visible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.paper import gold as goldlib                                    # noqa: E402
from tcr.paper import pairing, registry                                  # noqa: E402
from tcr.paper.io import (append_jsonl, fmt, fmt_pct, iter_jsonl, log,    # noqa: E402
                   read_csv, write_compact, write_csv, write_json, write_jsonl)
from tcr.paper.layout import Layout                                      # noqa: E402

#: Blocks stored per response in the prefix-candidate table.  Every boundary
#: R1 can select (a natural close, half of the reference list, the block before
#: the first reuse) is well inside this; a runaway response's thousandth block
#: is not a candidate for anything.
MAX_STORED_BLOCKS = 512


# --------------------------------------------------------------- response join

def responses_by_key(path: Path) -> dict[str, dict[str, Any]]:
    """`stable_prompt_id -> record`, with the eight samples kept as written."""
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        out[goldlib.canonical_key(row.get("key"))] = row
    return out


def response_text(record: Mapping[str, Any], seed: int) -> str | None:
    for sample in record.get("responses", []) or []:
        if int(sample.get("seed", -1)) == int(seed):
            return str(sample.get("response", "") or "")
    return None


# ------------------------------------------------------- per-response analysis

def block_table(row: Mapping[str, Any], text: str, record: goldlib.GoldRecord
                ) -> dict[str, Any]:
    """Everything R1's point selection and R0's utility check need, per response.

    Stored as parallel arrays rather than one row per block: 8,848 responses of
    ~66 blocks each is half a million blocks per model, and the selection only
    ever reads a handful of positions out of each response.
    """
    spans = list(row.get("block_index", []) or [])[:MAX_STORED_BLOCKS]
    values = goldlib.blocks_from_spans(text, spans)
    running = goldlib.coverage_prefix(values, record)

    seen: set[str] = set()
    first_occurrence: list[bool] = []
    for span in spans:
        signature = str(span.get("triple_hash"))
        complete = bool(span.get("identity_complete"))
        novel = complete and signature not in seen
        if complete:
            seen.add(signature)
        first_occurrence.append(novel)

    last_new_pair_block = 0
    previous_hits = 0
    for index, state in enumerate(running):
        if state["gold_pairs_hit"] > previous_hits:
            last_new_pair_block = index + 1
        previous_hits = state["gold_pairs_hit"]

    capture = row.get("motif_capture_triple", {}) or {}
    reuse = row.get("first_nonempty_triple_reuse", {}) or {}
    legacy = row.get("legacy_orbit", {}) or {}
    return {
        "sample_id": row.get("sample_id"),
        "stable_prompt_id": row.get("stable_prompt_id"),
        "key": row.get("key"),
        "seed": int(row.get("seed", -1)),
        "n_blocks": int(row.get("n_blocks", 0)),
        "n_blocks_stored": len(spans),
        "gold_blocks": record.n_blocks,
        "gold_pairs": len(record.pairs),
        "gold_triples": len(record.triples),
        "finish_reason": row.get("finish_reason", ""),
        "gen_tokens": int(row.get("gen_tokens", 0)),
        "hit_max": bool(legacy.get("hit_max_tokens")),
        "stable_orbit": bool(legacy.get("exists")),
        "full_json_list_valid": bool(row.get("full_json_list_valid")),
        "semantic_capture": bool(capture.get("exists")),
        "capture_second_copy_start_0based":
            int(capture["second_copy_start_0based"]) if capture.get("exists") else None,
        "capture_confirmed_block_exclusive_0based":
            int(capture["confirmed_at_block_exclusive_0based"]) if capture.get("exists") else None,
        "first_reuse_block_1based":
            int(reuse["block_index_1based"]) if reuse.get("exists") else None,
        "last_new_gold_pair_block_1based": last_new_pair_block,
        "char_end": [int(span["char_end"]) for span in spans],
        "identity_complete": [bool(span.get("identity_complete")) for span in spans],
        "first_occurrence": first_occurrence,
        "gold_pairs_hit": [state["gold_pairs_hit"] for state in running],
        "remaining_gold_pairs": [state["remaining_gold_pairs"] for state in running],
        "remaining_gold_triples": [state["remaining_gold_triples"] for state in running],
    }


def post_capture_utility(row: Mapping[str, Any], text: str,
                         record: goldlib.GoldRecord) -> dict[str, Any] | None:
    """After a capture is confirmed, does anything new still arrive?

    "New" here means a triple this response had not produced before.  A new
    string is not automatically a new correct relation, and a triple that does
    not match the reference is not automatically false, so the reference hits
    are counted separately from the novel strings.
    """
    capture = row.get("motif_capture_triple", {}) or {}
    if not capture.get("exists"):
        return None
    confirmed = int(capture["confirmed_at_block_exclusive_0based"])
    spans = list(row.get("block_index", []) or [])
    values = goldlib.blocks_from_spans(text, spans)

    seen: set[tuple[str, str, str]] = set()
    covered_triples: set[tuple[str, str, str]] = set()
    covered_pairs: set[tuple[str, str]] = set()
    for block in values[:confirmed]:
        identity = goldlib.block_identity(block) if isinstance(block, Mapping) else None
        if identity is None:
            continue
        seen.add(identity)
        if identity in record.triples:
            covered_triples.add(identity)
        if identity[:2] in record.pairs:
            covered_pairs.add(identity[:2])

    novel = new_gold_triples = new_gold_pairs = 0
    for block in values[confirmed:]:
        identity = goldlib.block_identity(block) if isinstance(block, Mapping) else None
        if identity is None or identity in seen:
            continue
        seen.add(identity)
        novel += 1
        if identity in record.triples and identity not in covered_triples:
            covered_triples.add(identity)
            new_gold_triples += 1
        if identity[:2] in record.pairs and identity[:2] not in covered_pairs:
            covered_pairs.add(identity[:2])
            new_gold_pairs += 1
    return {
        "blocks_after_capture": max(len(values) - confirmed, 0),
        "novel_triples_after_capture": novel,
        "new_gold_triples_after_capture": new_gold_triples,
        "new_gold_pairs_after_capture": new_gold_pairs,
        "gold_pairs_covered_at_capture": len(covered_pairs),
        "gold_pairs_total": len(record.pairs),
    }


def evaluated(layout: Layout, tag: str) -> str | None:
    """None when this model's evaluation has finished, else why it has not.

    The test is the SUMMARY and nothing else.  E1 appends to `responses` as it
    generates and resumes from it by key, so a partial responses file exists
    throughout the run and its presence says nothing about completion; `events`
    is replaced into place atomically at the end of the analysis stage and the
    summary is written after that stage returns.  Summary present therefore
    implies events present and responses complete.

    A summary with its data missing is not something to wait for -- the files
    were moved or deleted after the evaluation -- so it is reported separately
    from "still evaluating" and does not silently block a pair forever.
    """
    if not layout.summary(tag).is_file():
        return ("still evaluating" if layout.responses(tag).is_file()
                else "not evaluated")
    gone = [name for name, path in (("responses", layout.responses(tag)),
                                    ("events", layout.events(tag)))
            if not path.is_file()]
    if gone:
        return ("evaluated, but its " + " and ".join(gone)
                + " file is gone; R0 reads both")
    return None


def scan_model(tag: str, layout: Layout, gold: goldlib.GoldSet, *,
               want_blocks: bool, out_dir: Path) -> dict[str, Any]:
    """One pass over a model's events + responses: utility, and optionally the
    per-block prefix-candidate table."""
    events_path, responses_path = layout.events(tag), layout.responses(tag)
    for path in (events_path, responses_path):
        if not path.is_file():
            raise SystemExit(f"{tag}: missing {path}")
    records = responses_by_key(responses_path)

    utility: list[dict[str, Any]] = []
    candidates_path = out_dir / f"R0_prefix_candidates_{tag}.jsonl"
    if want_blocks and candidates_path.exists():
        candidates_path.unlink()
    n_rows = n_missing_text = n_missing_gold = 0
    buffered: list[dict[str, Any]] = []
    for row in iter_jsonl(events_path):
        n_rows += 1
        prompt_id = str(row.get("stable_prompt_id"))
        record_row = records.get(prompt_id)
        text = response_text(record_row, int(row.get("seed", -1))) if record_row else None
        if text is None:
            n_missing_text += 1
            continue
        try:
            record = gold.by_prompt_id(prompt_id)
        except KeyError:
            n_missing_gold += 1
            continue
        entry = post_capture_utility(row, text, record)
        if entry is not None:
            utility.append({"tag": tag, "sample_id": row.get("sample_id"),
                            "stable_prompt_id": prompt_id, **entry})
        if want_blocks:
            buffered.append(block_table(row, text, record))
            if len(buffered) >= 500:
                for item in buffered:
                    append_jsonl(candidates_path, item)
                buffered.clear()
        if n_rows % 2000 == 0:
            log("r0", f"  {tag}: {n_rows} responses")
    for item in buffered:
        append_jsonl(candidates_path, item)

    summary: dict[str, Any] = {
        "tag": tag,
        "n_responses": n_rows,
        "n_missing_response_text": n_missing_text,
        "n_missing_gold": n_missing_gold,
        "n_captured": len(utility),
        "prefix_candidates": str(candidates_path) if want_blocks else None,
    }
    if utility:
        def mean(field: str) -> float:
            return sum(float(row[field]) for row in utility) / len(utility)

        summary.update({
            "captured_share": len(utility) / max(n_rows, 1),
            "mean_blocks_after_capture": mean("blocks_after_capture"),
            "mean_novel_triples_after_capture": mean("novel_triples_after_capture"),
            "mean_new_gold_pairs_after_capture": mean("new_gold_pairs_after_capture"),
            "share_with_any_new_gold_pair": sum(
                1 for row in utility if row["new_gold_pairs_after_capture"] > 0
            ) / len(utility),
            "mean_gold_pair_coverage_at_capture": sum(
                row["gold_pairs_covered_at_capture"] / max(row["gold_pairs_total"], 1)
                for row in utility) / len(utility),
        })
    return {"summary": summary, "utility": utility}


# ------------------------------------------------------------- P0 invocations

def p0_outputs(out_dir: Path, pair: registry.Pair) -> dict[str, bool]:
    """Which analysers have actually left a manifest for this pair."""
    root = out_dir / pair.name
    return {
        "p0d": (root / "p0d_episode_hazard" / "P0D_MANIFEST.json").is_file(),
        "p0c": (root / "p0c_set_completion" / "P0C_MANIFEST.json").is_file()
               if pair.with_p0c else None,
    }


def p0_complete(out_dir: Path, pair: registry.Pair) -> bool:
    return all(value for value in p0_outputs(out_dir, pair).values()
               if value is not None)


def invalidate_p0_outputs(out_dir: Path, pair: registry.Pair) -> list[str]:
    """Remove this pair's manifests before its analysers run again.

    The analysers clear their own output directory, but only after argument
    validation and only if they get that far: an import error, a bad flag or a
    missing input makes them exit while last run's manifest is still on disk.
    A later "did it finish?" check would then read a manifest that describes an
    analysis this run never performed.  Deleting them first makes the manifest
    mean exactly one thing -- this run wrote it.
    """
    removed: list[str] = []
    for name, relative in (("p0d", "p0d_episode_hazard/P0D_MANIFEST.json"),
                           ("p0c", "p0c_set_completion/P0C_MANIFEST.json")):
        path = out_dir / pair.name / relative
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def analysis_failures(out_dir: Path, pairs: Sequence[registry.Pair],
                      exits: Mapping[str, Mapping[str, bool]] | None = None
                      ) -> list[str]:
    """Pairs that ran and did not finish.

    Two independent signals, because either one alone can lie: a manifest can
    be left over from an earlier run (which `invalidate_p0_outputs` prevents,
    and this re-checks), and a process can exit zero having written nothing.
    A pair is finished only when its analysers exited cleanly AND left the
    manifests the next stage will read.
    """
    failures: list[str] = []
    for pair in pairs:
        reasons: list[str] = []
        for name, present in p0_outputs(out_dir, pair).items():
            if present is None:          # this pair does not run that analyser
                continue
            if not present:
                reasons.append(f"{name} produced no manifest")
            elif exits is not None and exits.get(pair.name, {}).get(name) is False:
                reasons.append(f"{name} exited non-zero")
        if reasons:
            failures.append(f"{pair.name} ({'; '.join(reasons)})")
    return failures


def run_script(script: Path, args: Sequence[str], *, label: str) -> bool:
    command = [sys.executable, str(script), *args]
    log("r0", f"{label}: {' '.join(command[1:])}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        log("r0", f"[!] {label} exited {completed.returncode}; "
                  "its own validation message is above")
        return False
    return True


def p0d_metrics(pair_dir: Path) -> dict[str, Any]:
    """The handful of P0d numbers the paper quotes, read from its own CSVs."""
    path = pair_dir / "p0d_episode_hazard" / "P0D_KEY_METRICS.csv"
    if not path.is_file():
        return {}
    # Names as P0d writes them; `mean_episodes_experienced` is the response-scope
    # exposure count (E1's CSV calls the same quantity `episodes_per_response`).
    wanted = {
        "pooled_capture_hazard", "pooled_gap_stop_hazard",
        "pooled_inepisode_stop_hazard", "cif_capture_final", "cif_stop_final",
        "propensity_component", "exposure_component", "decomposition_total",
        "raw_cif_diff", "capture_rate", "stop_rate", "censor_rate",
        "mean_episodes_experienced", "zero_episode_rate", "mean_reuse_blocks",
        "stop_in_episode_share", "recovered_before_capture_mean",
        "orbit_given_captured",
    }
    out: dict[str, Any] = {}
    for row in read_csv(path):
        if row.get("episode_variant") != "lineage" or row.get("capture_kind") != "primary":
            continue
        metric = row.get("metric", "")
        if metric not in wanted:
            continue
        out[metric] = {key: row.get(key) for key in
                       ("m0", "m1", "diff_or_value", "ci_low", "ci_high", "direction")}
    return out


def compact_pair_lines(pair: registry.Pair, metrics: Mapping[str, Any]) -> list[str]:
    def line(metric: str, name: str, pct: bool = True) -> str:
        row = metrics.get(metric)
        if not row:
            return f"{name}=NA"
        render = fmt_pct if pct else fmt
        return (f"{name}: M0={render(row['m0'])} M1={render(row['m1'])} "
                f"d={render(row['diff_or_value'])}"
                f"[{render(row['ci_low'])},{render(row['ci_high'])}] {row['direction']}")

    return [
        f"[{pair.name}] M0={registry.label(pair.m0)} M1={registry.label(pair.m1)}",
        line("mean_episodes_experienced", "episodes/resp", pct=False),
        line("zero_episode_rate", "no-reuse responses"),
        line("capture_rate", "response capture"),
        line("pooled_capture_hazard", "per-episode capture"),
        line("pooled_gap_stop_hazard", "gap stop"),
        line("pooled_inepisode_stop_hazard", "in-episode stop"),
        line("cif_capture_final", "capture CIF"),
        line("propensity_component", "shapley propensity", pct=False),
        line("exposure_component", "shapley exposure", pct=False),
        line("raw_cif_diff", "raw CIF diff", pct=False),
    ]


# ---------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", default="all",
                        help="comma-separated pair names, or 'all'")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--tail-pool-min", type=int, default=50)
    parser.add_argument("--reparse", choices=("auto", "on", "off"), default="auto",
                        help="P0d's response reparse for the last-new-pair distance")
    parser.add_argument("--skip-p0", action="store_true",
                        help="only rebuild the CPU tables; do not call P0c/P0d")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo pairs and per-model scans that already "
                             "finished; without it they are skipped")
    args = parser.parse_args()

    layout = Layout.from_env()
    layout.ensure()
    out_dir = layout.r0
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = goldlib.load_gold(layout.gold)
    log("r0", f"gold: {len(gold)} evaluation records from {layout.gold}")

    wanted = ({name.strip() for name in args.pairs.split(",") if name.strip()}
              if args.pairs != "all" else None)
    pairs = [pair for pair in registry.R0_PAIRS
             if wanted is None or pair.name in wanted]
    if not pairs:
        raise SystemExit(f"no pair matches {args.pairs!r}")

    p0d_script = HERE / "analyze_p0d_episode_hazard.py"
    p0c_script = HERE / "analyze_p0c_set_completion.py"

    # What can actually run right now.  A pair whose arm is still being
    # evaluated is reported and left for a later run rather than turned into an
    # error: five of the nine pairs need no newly trained model at all, and
    # there is no reason for them to wait on the four that do.
    runnable: list[registry.Pair] = []
    waiting: list[dict[str, Any]] = []
    broken: list[dict[str, Any]] = []
    done: list[registry.Pair] = []
    for pair in pairs:
        blocked = {tag: reason for tag, reason in
                   ((tag, evaluated(layout, tag)) for tag in (pair.m0, pair.m1))
                   if reason}
        if blocked:
            row = {"pair": pair.name, "m0": pair.m0, "m1": pair.m1,
                   "blocked_by": blocked}
            (broken if any("gone" in reason for reason in blocked.values())
             else waiting).append(row)
            continue
        # A pair is finished when EVERY analyser it is supposed to run left its
        # manifest behind.  Checking only P0d would let a pair whose P0c failed
        # be skipped forever, and then reported as though P0c had succeeded.
        if p0_complete(out_dir, pair) and not args.overwrite and not args.skip_p0:
            done.append(pair)
            continue
        runnable.append(pair)

    log("r0", f"{len(runnable)} pair(s) to run, {len(done)} already done, "
              f"{len(waiting)} waiting on an unfinished evaluation"
              + (f", {len(broken)} with missing inputs" if broken else ""))
    for row in [*waiting, *broken]:
        detail = "; ".join(f"{tag} {reason}"
                           for tag, reason in row["blocked_by"].items())
        label = "no inputs" if row in broken else "waiting"
        log("r0", f"  {label}: {row['pair']} -- {detail}")
    for pair in done:
        log("r0", f"  done: {pair.name} (pass --overwrite to recompute)")
    if waiting:
        log("r0", "re-run this same command once those evaluations finish; "
                  "finished pairs are skipped, not recomputed")
    if broken:
        log("r0", "[!] the pairs above are not waiting on anything -- their "
                  "evaluation finished and the files it needs are missing")

    # Per-model CPU pass, cached per model so a later run only pays for what it
    # has not scanned yet.
    pool_sources = {tag for source in registry.R1_POOL_SOURCES.values() for tag in source}
    involved: list[str] = []
    for pair in [*runnable, *done]:
        for tag in (pair.m0, pair.m1):
            if tag not in involved:
                involved.append(tag)
    for tag in involved:
        summary_path = out_dir / f"R0_scan_{tag}.json"
        want_blocks = tag in pool_sources
        candidates_ok = (not want_blocks
                         or (out_dir / f"R0_prefix_candidates_{tag}.jsonl").is_file())
        if summary_path.is_file() and candidates_ok and not args.overwrite:
            log("r0", f"scan {tag}: already done, skipping")
            continue
        log("r0", f"scanning {tag}" + (" (+ prefix candidates)" if want_blocks else ""))
        result = scan_model(tag, layout, gold, want_blocks=want_blocks, out_dir=out_dir)
        write_csv(out_dir / f"R0_utility_{tag}.csv", result["utility"])
        write_json(summary_path, result["summary"])

    utility_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("R0_utility_*.csv")):
        utility_rows.extend(read_csv(path))
    for path in sorted(out_dir.glob("R0_scan_*.json")):
        model_summaries.append(json.loads(path.read_text(encoding="utf-8")))
    write_csv(out_dir / "R0_postcapture_utility.csv", utility_rows)
    write_csv(out_dir / "R0_model_scan_summary.csv", model_summaries)
    log("r0", f"post-capture utility: {len(utility_rows)} captured responses "
              f"over {len(model_summaries)} scanned models")

    # Pairing plus the frozen P0 analysers.
    pair_reports: list[dict[str, Any]] = []
    failures: list[str] = []

    exits: dict[str, dict[str, bool]] = {}
    for pair in runnable:
        pair_dir = out_dir / pair.name
        # Stale manifests go before the analysers start, so afterwards "the
        # manifest is there" can only mean "this run put it there".
        stale = invalidate_p0_outputs(out_dir, pair)
        if stale:
            log("r0", f"  {pair.name}: cleared {', '.join(stale)} manifest(s) "
                      "from a previous run")
        manifest = pairing.make_pair(
            m0_tag=pair.m0, m1_tag=pair.m1,
            m0_events=layout.events(pair.m0), m1_events=layout.events(pair.m1),
            m0_responses=layout.responses(pair.m0),
            m1_responses=layout.responses(pair.m1),
            out_dir=pair_dir)
        report: dict[str, Any] = {"pair": pair.name, "m0": pair.m0, "m1": pair.m1,
                                  "purpose": pair.purpose,
                                  "n_prompts": manifest["selection"]["n_prompts"],
                                  "p0d": False, "p0c": False}
        if not args.skip_p0:
            # --overwrite unconditionally: this pair was selected for a rerun,
            # and the analysers refuse to write into a non-empty directory.
            p0d_args = ["--result-dir", str(pair_dir),
                        "--bootstrap", str(args.bootstrap),
                        "--tail-pool-min", str(args.tail_pool_min),
                        "--reparse", args.reparse,
                        "--responses-m0", str(layout.responses(pair.m0)),
                        "--responses-m1", str(layout.responses(pair.m1)),
                        "--max-return-chars", "1900", "--overwrite"]
            report["p0d"] = run_script(p0d_script, p0d_args, label=f"P0d {pair.name}")
            if pair.with_p0c:
                p0c_args = ["--result-dir", str(pair_dir),
                            "--gold-data", str(layout.gold),
                            "--responses-m0", str(layout.responses(pair.m0)),
                            "--responses-m1", str(layout.responses(pair.m1)),
                            "--bootstrap", str(args.bootstrap),
                            "--max-return-chars", "1900", "--overwrite"]
                report["p0c"] = run_script(p0c_script, p0c_args,
                                           label=f"P0c {pair.name}")
        # The subprocess result is kept as `*_exit_ok`; the manifest on disk is
        # what `p0d` / `p0c` mean, because that is what later stages read.
        report["p0d_exit_ok"], report["p0c_exit_ok"] = report["p0d"], report["p0c"]
        exits[pair.name] = {"p0d": bool(report["p0d_exit_ok"]),
                            "p0c": bool(report["p0c_exit_ok"])}
        report.update(p0_outputs(out_dir, pair))
        report["metrics"] = p0d_metrics(pair_dir)
        pair_reports.append(report)
    if not args.skip_p0:
        failures = analysis_failures(out_dir, runnable, exits)
        for entry in failures:
            log("r0", f"[!] {entry}; it will be retried on the next run")

    # Pairs finished on an earlier run keep contributing their numbers.
    for pair in done:
        pair_reports.append({"pair": pair.name, "m0": pair.m0, "m1": pair.m1,
                             "purpose": pair.purpose, "n_prompts": None,
                             **p0_outputs(out_dir, pair),
                             "metrics": p0d_metrics(out_dir / pair.name)})

    write_json(out_dir / "R0_pairs.json", {
        "layout": layout.describe(),
        "bootstrap": args.bootstrap,
        "tail_pool_min": args.tail_pool_min,
        "pairs": pair_reports,
        "waiting_on_unfinished_evaluation": waiting,
        "blocked_by_missing_inputs": broken,
        "models_scanned": model_summaries,
    })

    # -- flat table for Figure 2 and the compact summaries
    flat: list[dict[str, Any]] = []
    for report in pair_reports:
        for metric, values in (report.get("metrics") or {}).items():
            # `values` carries its own m0/m1 (the two arms' levels), so the
            # model identities keep distinct names.
            flat.append({"pair": report["pair"], "m0_tag": report["m0"],
                         "m1_tag": report["m1"], "metric": metric, **values})
    write_csv(out_dir / "R0_pair_metrics.csv", flat)

    main_pairs = [report for report in pair_reports
                  if report["pair"] in {"4b_CK_s42", "4b_CO15_s42", "4b_notrainC_s42"}]
    other_pairs = [report for report in pair_reports if report not in main_pairs]
    for name, group in (("R0_01", main_pairs), ("R0_02", other_pairs)):
        if not group:
            continue
        lines: list[str] = []
        for report in group:
            spec = next(p for p in registry.R0_PAIRS if p.name == report["pair"])
            lines.extend(compact_pair_lines(spec, report.get("metrics") or {}))
            lines.append("")
        write_compact(layout.compact / f"{name}.txt",
                      "[R0 episode competing risk]", lines)
    log("r0", f"done: {len(pair_reports)} pairs -> {out_dir}")
    if failures:
        # Waiting on an evaluation is not a failure; an analyser that ran and
        # did not finish is.  Exiting non-zero stops the driver here instead of
        # letting the prefix pool be built on top of a broken R0.
        log("r0", f"[!] {len(failures)} pair(s) failed their analysis: "
                  + "; ".join(failures))
        log("r0", "the pairs reported as done above are usable; fix these and "
                  "re-run the same command")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
