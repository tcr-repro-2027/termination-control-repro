# coding=utf-8
"""What makes two runs of the same tag the SAME run.

Resume is keyed on `key`, which answers "did we already generate this record"
but not "did we generate it under the same conditions".  Those are different
questions, and only the second one decides whether the old rows may be reused:
the evaluation file can be edited in place, a checkpoint can be replaced, the
protocol can be bumped, and a `LIMIT=8` smoke run leaves behind a `.done.json`
that would otherwise convince the orchestrator a model is finished.

So every generation writes its identity next to its output -- first as a
`.run.json` sidecar before the first token, then into the final `.done.json`
and from there into the summary -- and every resume, and every decision to
reuse an artefact, compares it.  A mismatch is never repaired silently: it
stops the task with the field that differs, because the alternative is a
complete, plausible, incomparable set of numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from . import protocol
from .io_utils import read_json
from .prompts import rendered_digest

#: Everything that must agree for old rows to be reusable.  Anything that
#: changes what a response IS belongs here; anything that only changes how fast
#: it is produced (batch size, GPU, worker count) deliberately does not.
IDENTITY_FIELDS = (
    "protocol_version",     # the frozen protocol as a whole
    "mode", "k", "seeds",   # what a "response" is
    "max_model_len",        # the context the model was given
    "eval_data_sha256",     # the exact evaluation file
    "prompt_rendered_sha256",  # the exact prompt template
    "model_path",           # the exact checkpoint
    "tokenizer_path",       # onsets depend on it
    "limit",                # a smoke subset is not a run
)


def build_identity(*, eval_data_sha256: str, model_path: str,
                   tokenizer_path: str, max_model_len: int,
                   limit: Optional[int] = None) -> Dict[str, Any]:
    """The identity of the run that is about to produce (or produced) a file."""
    return {
        "protocol_version": protocol.PROTOCOL_VERSION,
        "mode": protocol.MODE,
        "k": protocol.K,
        "seeds": list(protocol.SEEDS),
        "max_model_len": int(max_model_len),
        "eval_data_sha256": eval_data_sha256,
        "prompt_rendered_sha256": rendered_digest(),
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "limit": int(limit) if limit else None,
    }


def identity_mismatches(expected: Mapping[str, Any],
                        found: Mapping[str, Any]) -> List[str]:
    """Human-readable list of the fields that disagree (empty = compatible)."""
    problems: List[str] = []
    for field in IDENTITY_FIELDS:
        want, got = expected.get(field), found.get(field)
        if field not in found:
            problems.append(f"{field}: missing from the existing artefact "
                            f"(expected {want!r})")
        elif want != got:
            problems.append(f"{field}: existing {got!r} != current {want!r}")
    return problems


def load_identity(path: str | Path) -> Optional[Dict[str, Any]]:
    """Read an identity block out of a `.run.json`, `.done.json` or summary.

    Returns ``None`` when the file does not exist or carries no identity, so
    the caller can distinguish "incompatible" from "unknown" -- they need
    different answers.
    """
    target = Path(path)
    if not target.is_file():
        return None
    try:
        document = read_json(target)
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    for holder in (document, document.get("generation") or {}):
        if isinstance(holder, dict) and isinstance(holder.get("identity"), dict):
            return holder["identity"]
    return None


def describe_mismatch(tag: str, source: Path, problems: List[str]) -> str:
    return (
        f"[{tag}] refusing to reuse {source.name}: it was produced by a "
        f"different run.\n  " + "\n  ".join(problems) +
        "\n  Either point --result_root at a fresh directory, or delete this "
        "task's artefacts (responses/, events/, summary/ entries for this tag) "
        "and let it regenerate."
    )
