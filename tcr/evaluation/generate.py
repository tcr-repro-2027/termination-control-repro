# coding=utf-8
"""Generate the K=8 E-Natural responses for ONE checkpoint on ONE GPU.

This is the unit of work the orchestrator hands to a GPU slot.  A 4B/8B Qwen3
fits on a single GPU with room for a 32K context, so tensor parallelism buys
nothing here: eight independent single-GPU workers keep all eight cards busy
and let each finish (and free its card) independently, which is the whole
point of the E1 queue.

Guarantees this file is responsible for
---------------------------------------
* the prompt is the FULL training template (`tcr.evaluation.prompts`), never the
  abridged v1.0 copy;
* the chat template resolves to the exact prefix training used
  (`protocol.CHAT_PREFIX_SUFFIX`).  A checkpoint whose `chat_template` drifted
  is a silent confound, so the prefix is asserted once and recorded;
* each of the K samples is its own sequence with its own seed -- never one
  request with `n=8`, whose per-choice seeds cannot be pinned;
* `max_tokens=None`, i.e. every prompt may generate its own remaining context
  (vLLM's default is 16, so None must be passed explicitly);
* a record whose prompt leaves less than `MIN_FREE_TOKENS` of room is written
  as an explicit *skipped* record instead of crashing the run or silently
  shrinking the denominator;
* the file is append-only and resumable by `key`, and a `.done.json` sentinel
  is written only after every record is on disk -- the orchestrator treats the
  sentinel, not the exit code, as proof of completion;
* resume is IDENTITY-CHECKED (`tcr.evaluation.identity`).  A `.run.json` sidecar is
  written before the first token and re-read on every resume, so an edited
  evaluation file, a replaced checkpoint, a bumped protocol or a leftover
  `LIMIT=8` smoke run stops the task instead of quietly contributing rows that
  no longer belong together.

Example::

    python -m tcr.evaluation.generate \
      --eval_data ./cleanv2/eval_supportclean_keep8.jsonl \
      --output_dir ./outputs/eval/responses \
      --model_path ./outputs/checkpoints/qwen3-4b-cleanv2-s42/checkpoint-417 \
      --tag qwen3-4b-cleanv2-s42 --gpu 3
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from . import protocol
from .identity import (
    build_identity, describe_mismatch, identity_mismatches, load_identity,
)
from .io_utils import (
    append_jsonl, load_done_keys, read_jsonl, sanitize_filename, sha256_file,
    write_json,
)
from .prompts import build_extraction_relation_prompt, prompt_fingerprint

EVAL_FIELDS = ("key", "source", "text", "entities_str")

LOG = logging.getLogger("e1.generate")


# --------------------------------------------------------------------- input

def load_eval_records(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    """Load E-Natural and validate what generation depends on."""
    records = list(read_jsonl(path))
    if not records:
        raise ValueError(f"eval data {path!r} is empty")
    for index, record in enumerate(records):
        missing = [field for field in EVAL_FIELDS if field not in record]
        if missing:
            raise ValueError(f"eval record #{index} (key={record.get('key')!r}) "
                             f"is missing {missing}")
    keys = [record["key"] for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"eval data {path!r} has duplicate keys; "
                         "resume-by-key needs them unique")
    return records


def response_path(output_dir: str | os.PathLike[str], tag: str) -> Path:
    return Path(output_dir) / f"{sanitize_filename(tag)}_{protocol.MODE}_n{protocol.K}.jsonl"


def done_path(output_dir: str | os.PathLike[str], tag: str) -> Path:
    return response_path(output_dir, tag).with_suffix(".done.json")


def run_manifest_path(output_dir: str | os.PathLike[str], tag: str) -> Path:
    """Identity sidecar, written BEFORE generation so a killed run is still
    identifiable on resume."""
    return response_path(output_dir, tag).with_suffix(".run.json")


# ------------------------------------------------------------ chat templating

def render_prompt(record: Dict[str, Any]) -> str:
    return build_extraction_relation_prompt(text=record["text"],
                                            entities_str=record["entities_str"])


def apply_chat_template(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=protocol.MODE == "think",
    )


def check_chat_prefix(tokenizer) -> Dict[str, Any]:
    """Assert the nothink chat prefix matches what ms-swift trained under."""
    rendered = apply_chat_template(tokenizer, "__E1_PROBE__")
    ok = rendered.endswith(protocol.CHAT_PREFIX_SUFFIX)
    return {
        "chat_prefix_ok": ok,
        "chat_prefix_expected": protocol.CHAT_PREFIX_SUFFIX,
        "chat_prefix_actual_tail": rendered[-len(protocol.CHAT_PREFIX_SUFFIX) - 40:],
    }


# ------------------------------------------------------------------ main pass

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate K=8 seeded E-Natural responses for one "
                    "checkpoint on one GPU (E1 protocol).")
    parser.add_argument("--eval_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tokenizer_path", default=None,
                        help="Defaults to --model_path.  Set it when a "
                             "checkpoint was saved without tokenizer files.")
    parser.add_argument("--tag", required=True,
                        help="Names every artefact of this task.")
    parser.add_argument("--gpu", default=None,
                        help="A single GPU id; sets CUDA_VISIBLE_DEVICES "
                             "before vLLM is imported.")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--max_model_len", type=int,
                        default=protocol.MAX_MODEL_LEN)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--swap_space", type=int, default=None)
    parser.add_argument("--batch_records", type=int, default=64,
                        help="Eval records per LLM.generate call (x8 "
                             "sequences); also the resume granularity.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Smoke test: only the first N eval records.")
    parser.add_argument("--allow_unverified_resume", action="store_true",
                        help="Continue an existing responses file that carries "
                             "no run manifest.  Only for files written before "
                             "identity checking existed; it disables the one "
                             "guard against mixing two runs in one file.")
    parser.add_argument("--log_file", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    log_file = args.log_file or f"logs/gen_{sanitize_filename(args.tag)}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
        force=True)

    started = time.time()
    output = response_path(args.output_dir, args.tag)
    sentinel = done_path(args.output_dir, args.tag)
    run_manifest = run_manifest_path(args.output_dir, args.tag)
    tokenizer_path = args.tokenizer_path or args.model_path

    eval_sha = sha256_file(args.eval_data)
    identity = build_identity(eval_data_sha256=eval_sha,
                              model_path=args.model_path,
                              tokenizer_path=tokenizer_path,
                              max_model_len=args.max_model_len,
                              limit=args.limit)
    verify_resume(output, run_manifest, sentinel, identity, tag=args.tag,
                  allow_unverified=args.allow_unverified_resume)
    write_json(run_manifest, {"tag": args.tag, "identity": identity,
                              "started": time.strftime("%F %T")})

    records = load_eval_records(args.eval_data)
    if args.limit:
        records = records[:args.limit]
    done = load_done_keys(output)
    todo = [record for record in records if record["key"] not in done]
    LOG.info("tag=%s model=%s gpu=%s records=%d done=%d todo=%d -> %s",
             args.tag, args.model_path, args.gpu, len(records), len(done),
             len(todo), output)

    # Imported only now, so CUDA_VISIBLE_DEVICES is already in place.
    from transformers import AutoTokenizer
    import vllm
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prefix_report = check_chat_prefix(tokenizer)
    if not prefix_report["chat_prefix_ok"]:
        raise RuntimeError(
            "chat template does not end with the training-time nothink prefix "
            f"{protocol.CHAT_PREFIX_SUFFIX!r}; got "
            f"...{prefix_report['chat_prefix_actual_tail']!r}.  Evaluating this "
            "checkpoint would compare a different conditioning than it was "
            "trained under.")

    # Prompt lengths are model-independent (one Qwen3 vocabulary), so the
    # skipped set is identical for every task and the comparison stays balanced.
    chat_prompts: Dict[Any, str] = {
        record["key"]: apply_chat_template(tokenizer, render_prompt(record))
        for record in records}
    prompt_lens: Dict[Any, int] = {}
    keys = [record["key"] for record in records]
    for start in range(0, len(keys), 64):
        window = keys[start:start + 64]
        encoded = tokenizer([chat_prompts[key] for key in window],
                            add_special_tokens=False)["input_ids"]
        prompt_lens.update(zip(window, (len(ids) for ids in encoded)))
    budget = args.max_model_len - protocol.MIN_FREE_TOKENS
    too_long = {key for key, length in prompt_lens.items() if length > budget}
    if too_long:
        LOG.warning("%d/%d records exceed max_model_len-%d and will be recorded "
                    "as skipped", len(too_long), len(records),
                    protocol.MIN_FREE_TOKENS)

    # Records that cannot fit are written first and unconditionally: if they
    # are all that is left, the run must not spend five minutes loading a model
    # it will never call.
    pending = [record for record in todo if record["key"] not in too_long]
    for record in todo:
        if record["key"] in too_long:
            append_jsonl(output, _skipped_record(record, prompt_lens))

    if pending:
        llm_kwargs: Dict[str, Any] = {
            "model": args.model_path,
            "tokenizer": tokenizer_path,
            "tensor_parallel_size": args.tensor_parallel,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": "bfloat16",
        }
        if args.max_num_seqs:
            llm_kwargs["max_num_seqs"] = args.max_num_seqs
        if args.swap_space:
            llm_kwargs["swap_space"] = args.swap_space
        llm = LLM(**llm_kwargs)

        base = protocol.sampling_params()
        per_seed = [SamplingParams(seed=seed, max_tokens=protocol.MAX_TOKENS, **base)
                    for seed in protocol.SEEDS]

        written = 0
        for start in range(0, len(pending), args.batch_records):
            chunk = pending[start:start + args.batch_records]
            prompts: List[str] = []
            params: List[SamplingParams] = []
            for record in chunk:
                for sampling in per_seed:
                    prompts.append(chat_prompts[record["key"]])
                    params.append(sampling)
            # use_tqdm=False: a two-day unattended run must not fill its log
            # with progress bars.  Progress is one line per chunk instead.
            outputs = llm.generate(prompts, params, use_tqdm=False)

            for index, record in enumerate(chunk):
                samples = []
                for offset, seed in enumerate(protocol.SEEDS):
                    completion = outputs[index * protocol.K + offset].outputs[0]
                    answer, reasoning = split_reasoning(completion.text)
                    samples.append({
                        "seed": seed,
                        "response": answer,
                        "reasoning": reasoning,
                        "finish_reason": completion.finish_reason or "",
                        "gen_tokens_engine": len(completion.token_ids),
                    })
                append_jsonl(output, {
                    "key": record["key"],
                    "source": record.get("source", ""),
                    "text": record["text"],
                    "entities_str": record["entities_str"],
                    "prompt_tokens": prompt_lens[record["key"]],
                    "skipped_reason": "",
                    "responses": samples,
                })
                written += 1
            LOG.info("[%s] %d/%d records generated (%.1f min elapsed)",
                     args.tag, written, len(pending), (time.time() - started) / 60.0)

    elapsed = time.time() - started
    manifest = {
        "tag": args.tag,
        "identity": identity,
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "tokenizer_is_fallback": tokenizer_path != args.model_path,
        "gpu": args.gpu,
        "eval_data": str(args.eval_data),
        "eval_data_sha256": eval_sha,
        "n_eval_records": len(records),
        "n_records_written": len(load_done_keys(output)),
        "n_skipped_too_long": len(too_long),
        "skipped_keys": sorted(too_long, key=repr),
        "prompt_tokens_max": max(prompt_lens.values()) if prompt_lens else 0,
        "responses_path": str(output),
        "generation_seconds": round(elapsed, 1),
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        **prefix_report,
        **prompt_fingerprint(),
        **protocol.describe(),
    }
    write_json(sentinel, manifest)
    LOG.info("[%s] generation finished in %.1f min -> %s",
             args.tag, elapsed / 60.0, output)
    return 0


def verify_resume(output: Path, run_manifest: Path, sentinel: Path,
                  identity: Dict[str, Any], *, tag: str,
                  allow_unverified: bool = False) -> None:
    """Refuse to append to, or reuse, a file from a different run.

    Three cases, and they need three different answers:
    * no output yet                -> nothing to verify;
    * output plus a manifest       -> compare, and stop on any mismatch;
    * output but NO manifest       -> unknown provenance.  Stopping is the only
      safe answer: appending would interleave two runs inside one file, and
      that file is what every downstream number is computed from.
    """
    if not output.is_file() or output.stat().st_size == 0:
        return
    for source in (run_manifest, sentinel):
        found = load_identity(source)
        if found is None:
            continue
        problems = identity_mismatches(identity, found)
        if problems:
            raise SystemExit(describe_mismatch(tag, source, problems))
        LOG.info("[%s] resuming a run with matching identity (%s)",
                 tag, source.name)
        return
    if allow_unverified:
        LOG.warning("[%s] resuming %s without a run manifest "
                    "(--allow_unverified_resume)", tag, output.name)
        return
    raise SystemExit(
        f"[{tag}] {output} already has records but no run manifest beside it, "
        "so there is no way to tell which run wrote them.  Delete it (and this "
        "tag's events/, summary/ entries) to regenerate, or pass "
        "--allow_unverified_resume if you are certain it belongs to this run.")


def _skipped_record(record: Dict[str, Any],
                    prompt_lens: Dict[Any, int]) -> Dict[str, Any]:
    return {
        "key": record["key"],
        "source": record.get("source", ""),
        "text": record["text"],
        "entities_str": record["entities_str"],
        "prompt_tokens": prompt_lens[record["key"]],
        "skipped_reason": "prompt_exceeds_context",
        "responses": [],
    }


def split_reasoning(generated: str) -> tuple[str, str]:
    """Split raw generated text into ``(answer, reasoning)``.

    Identical to protocol v1.0's rule.  In nothink the model is already past
    `</think>`, so the whole generation is the answer; the branch is kept so a
    checkpoint that emits a think block anyway is recorded rather than scored
    as if the block were part of the answer.
    """
    if "</think>" in generated:
        head, _, tail = generated.partition("</think>")
        return tail.lstrip("\n"), head.replace("<think>", "", 1).strip()
    stripped = generated.lstrip()
    if stripped.startswith("<think>"):
        return "", stripped[len("<think>"):].strip()
    return generated, ""


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
