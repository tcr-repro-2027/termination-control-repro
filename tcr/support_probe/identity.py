# coding=utf-8
"""What makes two E2 measurements the same measurement.

A stage is skipped when its sentinel exists.  That is only safe if the sentinel
proves the finished run answered the QUESTION THIS RUN IS ASKING -- same
protocol, same anchors, same model, same sampler, same arms, same (absence of a)
limit.  Without that check a debug run silently satisfies the real one, and the
CSV carries a number nobody can reproduce.

Three things are guarded here, not one:

* the SENTINEL, which says a stage finished;
* the PARTIAL readout file, which resume appends to.  A partial file has no
  header of its own, so it gets a sidecar stamp; without it, rebuilding the
  anchors and re-running would append new cells to rows measured against the
  old anchor set and nothing would notice.
* the ANCHOR SET itself, which every model must share: an anchor file whose
  report is missing, stale or describes a different eval set puts the models on
  different scales, and ACI/ECI stop being comparable numbers.

A `--limit` run never writes the final sentinel: it is a smoke test, and a
smoke test must not be able to mark the real work done.

Two entry points, deliberately different:

* :func:`sentinel_says_done` RAISES -- it is called inside the worker that is
  about to redo or reuse the work, where guessing is worse than stopping;
* :func:`sentinel_mismatches` RETURNS the problems -- it is called by the
  orchestrator, which must survey every stage before spending a night of GPU
  time and report all of them at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from . import protocol
from .io_utils import read_json, read_jsonl, sha256_file, write_json
from .prompts import prompt_fingerprint


def run_identity(*, anchors_path: str, model_path: str, tokenizer_path: str,
                 limit: Optional[int] = None,
                 arms: Optional[str] = None,
                 max_model_len: Optional[int] = None,
                 anchors_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Everything that would change the numbers if it changed.

    ``anchors_sha256`` may be supplied by a caller that already hashed the
    anchor file: the orchestrator builds one identity per (model, stage) on
    every scheduling pass, and re-reading a 30 MB anchor set each time is pure
    waste.  The digest is the same value either way."""
    identity: Dict[str, Any] = {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "anchors_sha256": anchors_sha256 or sha256_file(anchors_path),
        "prompt_rendered_sha256": prompt_fingerprint()["prompt_rendered_sha256"],
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "sampler": protocol.sampler_params(),
        "limit": limit,
    }
    if arms is not None:
        identity["arms"] = arms
    if max_model_len is not None:
        # The hazard GENERATES, so vLLM's context ceiling is part of what it
        # measured: a smaller one changes which anchors fit and what the model
        # can emit.  Scoring does not take it (the length gate is the protocol's
        # MAX_CONTEXT_TOKENS), so it is absent there rather than null.
        identity["max_model_len"] = int(max_model_len)
    return identity


def mismatches(expected: Dict[str, Any], found: Dict[str, Any]) -> List[str]:
    return [f"{key}: existing {found.get(key)!r} != current {value!r}"
            for key, value in expected.items() if found.get(key) != value]


def stamp_path(out_path: Path) -> Path:
    """Sidecar for the append-only readout file."""
    return out_path.with_suffix(out_path.suffix + ".identity.json")


def sentinel_mismatches(sentinel: Path, identity: Dict[str, Any]) -> List[str]:
    """Why the sentinel on disk does not describe THIS measurement.

    Empty list means the finished stage answered the question this run is
    asking.  The caller decides what to do about a non-empty one; this function
    never raises, because the orchestrator needs to survey every stage and
    print one complete list rather than die on the first stale file.
    """
    if not sentinel.is_file():
        return ["sentinel missing"]
    try:
        found = read_json(sentinel).get("identity", {})
    except (OSError, ValueError) as exc:
        return [f"sentinel is unreadable: {exc}"]
    if not found:
        return ["sentinel carries no identity block (written before identity "
                "checking existed); delete it and re-measure"]
    return mismatches(identity, found)


def sentinel_says_done(sentinel: Path, identity: Dict[str, Any], tag: str,
                       stage: str) -> bool:
    """True if this stage is genuinely already done for THIS identity.

    Raises when a sentinel exists but was produced by a different measurement:
    silently redoing the work would be safe, silently REUSING it is not, and
    guessing which the user meant is worse than stopping.
    """
    if not sentinel.is_file():
        return False
    problems = sentinel_mismatches(sentinel, identity)
    if problems:
        raise SystemExit(
            f"[{tag}] refusing to reuse {sentinel.name} for {stage}: "
            + "; ".join(problems)
            + f"\n  delete {sentinel} (and the matching readouts) to re-measure")
    return True


def guard_partial(out_path: Path, identity: Dict[str, Any], tag: str) -> None:
    """Refuse to append to a partial file from a different measurement."""
    stamp = stamp_path(out_path)
    if out_path.is_file() and stamp.is_file():
        problems = mismatches(identity, read_json(stamp).get("identity", {}))
        if problems:
            raise SystemExit(
                f"[{tag}] refusing to resume {out_path.name}: "
                + "; ".join(problems)
                + f"\n  delete {out_path} and {stamp.name} to re-measure")
    elif out_path.is_file() and not stamp.is_file():
        raise SystemExit(
            f"[{tag}] {out_path.name} exists without {stamp.name}, so the "
            "measurement it holds cannot be identified; delete it to re-measure")
    write_json(stamp, {"identity": identity, "tag": tag})


# ------------------------------------------------------------- the anchor set

def anchor_set_problems(anchors_path: str | Path, report_path: str | Path, *,
                        n_per_type: int,
                        eval_data: Optional[str | Path] = None,
                        require_token_matched: bool = True,
                        max_length_delta: Optional[int] = 0) -> List[str]:
    """Everything that makes an anchor file unusable, as one list.

    The launcher skips construction when the file exists, and every model in
    the matrix is then measured against whatever that file happens to hold, so
    the checks here are the only thing standing between a wrong anchor set and
    a complete, healthy-looking, incomparable result table:

    * the report must EXIST and its `anchors_sha256` must be present and match.
      A missing key is not a pass -- the builder always writes it, so its
      absence means the report came from somewhere else;
    * the protocol version must be this one;
    * every built type must be full: an under-filled axis only widens a CI, so
      nothing downstream would ever reveal it;
    * `token_matched` must be true.  A character-fallback build matches donor
      lengths in characters, which is a pilot build, not the protocol's
      token-length-matched one;
    * the donor length tolerance must be the one this run means.  An anchor set
      built with `MAX_LENGTH_DELTA=2` is not the §8.3 length-matched one, and
      reusing it under the default 0 would silently relax an invariant the run
      believes it is enforcing;
    * when the caller knows which eval set it asked for, the report's
      `eval_data_sha256` must be that file's.
    """
    problems: List[str] = []
    anchors = Path(anchors_path)
    report_file = Path(report_path)
    if not anchors.is_file():
        return [f"anchor set missing: {anchors} (run with --build first)"]
    if not report_file.is_file():
        return [f"no report beside {anchors.name} ({report_file.name}): the "
                "build did not finish; delete the file and rebuild"]
    try:
        report = read_json(report_file)
    except (OSError, ValueError) as exc:
        return [f"{report_file.name} is unreadable: {exc}"]

    digest = sha256_file(anchors)
    recorded = report.get("anchors_sha256")
    if not recorded:
        problems.append(f"{report_file.name} records no anchors_sha256; it does "
                        "not describe this anchor file")
    elif recorded != digest:
        problems.append(f"{anchors.name} does not match its report "
                        f"({digest[:12]} != {str(recorded)[:12]})")

    version = (report.get("protocol") or {}).get("protocol_version")
    if version != protocol.PROTOCOL_VERSION:
        problems.append(f"anchors were built under protocol {version!r}, this "
                        f"run is {protocol.PROTOCOL_VERSION!r}")

    if require_token_matched and not report.get("token_matched"):
        problems.append("anchors were built WITHOUT a tokenizer, so donor "
                        "entities are matched on characters instead of tokens "
                        "(a pilot build); rebuild with BUILD_TOKENIZER set")

    if max_length_delta is not None:
        allowed = report.get("max_length_delta_allowed")
        if allowed is None:
            problems.append("the report does not say what donor length "
                            "tolerance the build used; it predates the "
                            "length-matching invariant")
        elif int(allowed) != int(max_length_delta):
            problems.append(f"anchors were built with a donor length tolerance "
                            f"of {allowed}, this run means {max_length_delta}")

    if eval_data is not None:
        want = Path(eval_data)
        if not want.is_file():
            problems.append(f"evaluation set missing: {want}")
        elif report.get("eval_data_sha256") != sha256_file(want):
            problems.append(f"anchors were built from a different eval set "
                            f"({report.get('eval_data')}), not {want}")

    counts: Dict[str, int] = {}
    try:
        for row in read_jsonl(anchors):
            name = row.get("anchor_type")
            counts[name] = counts.get(name, 0) + 1
    except (OSError, ValueError) as exc:
        problems.append(f"{anchors.name} does not read as jsonl: {exc}")
        return problems
    short = {name: counts.get(name, 0) for name in protocol.BUILT_ANCHOR_TYPES
             if counts.get(name, 0) < n_per_type}
    if short:
        problems.append(f"under-filled anchor type(s) {short} "
                        f"(wanted {n_per_type} each)")
    unexpected = sorted(set(counts) - set(protocol.BUILT_ANCHOR_TYPES))
    if unexpected:
        problems.append(f"anchor file carries unbuilt type(s): {unexpected}")
    return problems
