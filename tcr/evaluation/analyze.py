# coding=utf-8
"""Score ONE finished responses file: repetition + utility + support, on CPU.

This is the half of a task that runs while its GPU has already moved on to the
next checkpoint.  It reads `<tag>_nothink_n8.jsonl`, and writes

    events/<tag>_event_rows.jsonl     one 01b-schema row per response
    summary/<tag>_summary.json        every metric, with provenance
    audit/<tag>_audit_samples.jsonl   a bounded human-inspection sample
    e1_metrics.csv                    one row (optional; the orchestrator
                                      normally appends it instead)

`event_rows.jsonl` is the artefact that matters beyond E1.  It is written in
the frozen 01b schema, so `tcr/events`'s P0b (competing risk),
P0c (set completion) and P0d (episode hazard) analysers run on E1 output
unchanged -- pair two tasks with `scripts/e1_make_pair.py` and point their
`--gold` at the same `eval_supportclean_keep8.jsonl`.

Memory and restart behaviour
----------------------------
Rows are streamed to a `.partial` file and renamed on success, so analysis is
atomic: a killed worker leaves no half file that the orchestrator could mistake
for a finished one, and re-running simply redoes the task.  Only a ~1 KB
projection per response is held in RAM (`aggregate.project_row`), and the
episode pass re-reads the finished file line by line, so a worker stays around
a hundred MB regardless of how much the model looped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

from . import aggregate, protocol, quality as quality_mod
from .io_utils import (
    JsonlWriter, append_csv_row, append_jsonl, ensure_parent, read_jsonl,
    read_json, sanitize_filename, sha256_file, write_json,
)
from .structured import (
    build_event_row, encode_batch, legacy_detector, load_tokenizer, response_text,
)

LOG = logging.getLogger("e1.analyze")

#: Records per tokenizer batch.  Each carries K=8 responses, so 8 records is a
#: 64-text batch -- large enough for the Rust tokenizer to be efficient, small
#: enough that one very long batch cannot spike memory.
BATCH_RECORDS = 8

#: Progress log cadence, in responses.
_REPORT_EVERY = 2000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one E1 responses file (structured repetition + "
                    "legacy loop + relation-extraction utility + support).")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--eval_data", required=True,
                        help="E-Natural jsonl; gold is joined back by `key`.")
    parser.add_argument("--tokenizer_path", required=True,
                        help="The DEPLOYED checkpoint's fast tokenizer; every "
                             "token onset depends on it.")
    parser.add_argument("--output_root", required=True,
                        help="Result root; events/, summary/ and audit/ live "
                             "under it.")
    parser.add_argument("--tag", default=None,
                        help="Defaults to the responses file stem.")
    parser.add_argument("--task_json", default=None,
                        help="Task metadata from the registry (identity "
                             "columns of the CSV).")
    parser.add_argument("--csv", default=None,
                        help="Append the summary row to this CSV as well.")
    parser.add_argument("--audit_samples", type=int, default=40,
                        help="Loop/capture responses kept with full context "
                             "for manual inspection (0 disables).")
    parser.add_argument("--keep_diagnostics", action="store_true",
                        help="Keep the full parser diagnostics (raw text of "
                             "every malformed object) in the event rows.")
    parser.add_argument("--bootstrap", type=int, default=protocol.BOOTSTRAP)
    parser.add_argument("--log_file", default=None)
    return parser.parse_args(argv)


def _iter_records(path: str) -> Iterator[Dict[str, Any]]:
    return read_jsonl(path)


def analyze(*, responses_path: str, eval_data: str, tokenizer_path: str,
            output_root: str, tag: str, task: Dict[str, Any] | None = None,
            audit_samples: int = 40, keep_diagnostics: bool = False,
            n_boot: int = protocol.BOOTSTRAP, tokenizer: Any = None
            ) -> Dict[str, Any]:
    """Run the whole CPU stage and return the summary dict.

    ``tokenizer`` is injectable so the whole path can be exercised without a
    model directory; production always leaves it None and loads the deployed
    checkpoint's own fast tokenizer, which is what every token onset means."""
    started = time.time()
    root = Path(output_root)
    safe = sanitize_filename(tag)
    events_path = root / "events" / f"{safe}_event_rows.jsonl"
    partial_path = events_path.with_name(events_path.name + ".partial")
    audit_path = root / "audit" / f"{safe}_audit_samples.jsonl"
    summary_path = root / "summary" / f"{safe}_summary.json"
    ensure_parent(events_path)
    ensure_parent(audit_path)
    ensure_parent(summary_path)
    partial_path.unlink(missing_ok=True)
    audit_path.unlink(missing_ok=True)

    gold_by_key = {record["key"]: quality_mod.gold_record(record)
                   for record in read_jsonl(eval_data)}
    tokenizer = tokenizer if tokenizer is not None else load_tokenizer(tokenizer_path)
    detector = legacy_detector()

    projections: List[Dict[str, Any]] = []
    quality_samples: List[Dict[str, Any]] = []
    n_skipped = 0
    total_gen_tokens = 0
    audit_written = 0

    batch: List[Dict[str, Any]] = []
    next_report = _REPORT_EVERY

    with JsonlWriter(partial_path, mode="w") as events_out:

        def flush(records: List[Dict[str, Any]]) -> None:
            nonlocal total_gen_tokens, audit_written
            texts: List[str] = []
            meta: List[tuple] = []
            for record in records:
                gold = gold_by_key[record["key"]]
                for sample in record["responses"]:
                    texts.append(response_text(sample))
                    meta.append((record, sample, gold))
            encoded = encode_batch(tokenizer, texts)
            for (record, sample, gold), text, (ids, offsets) in zip(
                    meta, texts, encoded):
                row, snippets = build_event_row(
                    model_tag=tag, key=record["key"], seed=sample.get("seed", -1),
                    text=text, token_ids=ids, offsets=offsets,
                    finish_reason=sample.get("finish_reason", ""),
                    prompt_tokens=record.get("prompt_tokens"),
                    gen_tokens_engine=sample.get("gen_tokens_engine"),
                    config=detector, keep_diagnostics=keep_diagnostics,
                    keep_snippets=False)
                scored = quality_mod.score_sample(sample.get("response", ""), gold)
                row["quality"] = scored
                events_out.write(row)

                projections.append(aggregate.project_row(row))
                quality_samples.append(scored)
                total_gen_tokens += int(row["gen_tokens"])

                if (audit_written < audit_samples
                        and (row["stage_flags"]["capture"]
                             or row["legacy_orbit"]["exists"])):
                    append_jsonl(audit_path, {
                        "tag": tag, "key": record["key"],
                        "seed": sample.get("seed"),
                        "stage_chain": row["stage_chain"],
                        "alignment_type": row["alignment_type"],
                        "n_blocks": row["n_blocks"],
                        "gen_tokens": row["gen_tokens"],
                        "snippets": snippets,
                    })
                    audit_written += 1

        for record in _iter_records(responses_path):
            if not record.get("responses"):
                n_skipped += 1
                continue
            if record["key"] not in gold_by_key:
                # Fail immediately: scoring the rest would only produce a
                # complete, plausible, wrong set of numbers.
                raise ValueError(
                    f"response record key={record['key']!r} has no gold row in "
                    f"{eval_data}; these responses were generated against a "
                    "different evaluation set")
            batch.append(record)
            if len(batch) >= BATCH_RECORDS:
                flush(batch)
                batch = []
                if len(projections) >= next_report:
                    LOG.info("[%s] %d responses scored", tag, len(projections))
                    next_report += _REPORT_EVERY
        if batch:
            flush(batch)

    os.replace(partial_path, events_path)

    events = aggregate.summarize_events(projections)
    pooled = quality_mod.pooled(quality_samples, total_gen_tokens=total_gen_tokens)
    ci = aggregate.bootstrap_rates(projections, quality_samples, n_boot=n_boot)

    from .episodes import episode_summary          # imported late: heavy module
    episodes = episode_summary(read_jsonl(events_path))

    generation: Dict[str, Any] = {}
    sentinel = Path(responses_path).with_suffix(".done.json")
    if sentinel.is_file():
        manifest = read_json(sentinel)
        generation = {
            name: manifest.get(name) for name in (
                "identity", "eval_data_sha256", "prompt_rendered_sha256",
                "chat_prefix_ok", "vllm_version", "n_skipped_too_long",
                "n_eval_records", "prompt_tokens_max", "tokenizer_is_fallback")
        }
        generation["generation_minutes"] = round(
            float(manifest.get("generation_seconds", 0.0)) / 60.0, 2)
    if generation.get("n_skipped_too_long") is None:
        generation["n_skipped_too_long"] = n_skipped

    summary = {
        "tag": tag,
        "task": task or {"tag": tag},
        "mode": protocol.MODE,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol": protocol.describe(),
        "responses_path": str(responses_path),
        "responses_sha256": sha256_file(responses_path),
        "event_rows_path": str(events_path),
        "eval_data": str(eval_data),
        "tokenizer_path": str(tokenizer_path),
        "n_skipped_records": n_skipped,
        "generation": generation,
        "events": events,
        "quality": pooled,
        "episodes": episodes,
        "bootstrap_ci": ci,
        "analysis_minutes": round((time.time() - started) / 60.0, 2),
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(summary_path, summary)
    LOG.info("[%s] scored %d responses in %.1f min -> %s", tag,
             len(projections), summary["analysis_minutes"], summary_path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tag = args.tag or Path(args.responses).stem
    log_file = args.log_file or f"logs/analyze_{sanitize_filename(tag)}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_file, encoding="utf-8")],
        force=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    task = json.loads(Path(args.task_json).read_text(encoding="utf-8")) \
        if args.task_json else None
    summary = analyze(responses_path=args.responses, eval_data=args.eval_data,
                      tokenizer_path=args.tokenizer_path,
                      output_root=args.output_root, tag=tag, task=task,
                      audit_samples=args.audit_samples,
                      keep_diagnostics=args.keep_diagnostics,
                      n_boot=args.bootstrap)
    if args.csv:
        append_csv_row(args.csv, aggregate.summary_to_row(summary),
                       aggregate.CSV_FIELDS)

    events = summary["events"]
    print("=" * 72)
    print(f"{tag}  records={events['n_records']} responses={events['n_responses']}")
    print(f"  SemanticCapture={events['semantic_capture_rate']:.4f}  "
          f"StableOrbit={events['stable_orbit_rate']:.4f}  "
          f"legacyLoop={events['legacy_loop_rate']:.4f}  "
          f"hitMax={events['hit_max_rate']:.4f}")
    print(f"  strictF1={summary['quality']['strict_f1']:.4f}  "
          f"relaxedF1={summary['quality']['relaxed_f1']:.4f}  "
          f"json={summary['quality']['json_valid_rate']:.4f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
