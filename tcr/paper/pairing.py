# coding: utf-8
"""Turn two E1 tasks into the M0/M1 directory the 01b P0 analysers read.

`experiments/3_evaluate/e1_make_pair.py` already does the relabelling. What it
does not write is the manifest shape P0c and P0d validate on load -- they want
`selection.mode == "all"`, `protocol.raw_onset_moved == false`, a
`selection_manifest.json`, and `inputs.responses_m0/m1` so the response reparse
can find its files.  Without them the only way through is `--allow-sample`,
which those scripts document as a smoke-test escape hatch; writing the real
manifest is both more honest about what this is (a complete evaluation, all
1,106 records, all eight seeds) and keeps every downstream validator armed.

The pairing itself is deliberately conservative: a prompt/seed present in only
one arm is dropped and counted, because a paired hazard over a prompt set that
differs between arms is not a paired hazard.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .gold import canonical_key
from .io import iter_jsonl, log, read_json, write_json

#: The 01b P0 analysers are hard-wired to these two tags.
M0, M1 = "M0", "M1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = (str(row.get("stable_prompt_id")), int(row.get("seed")))
        if key in rows:
            raise SystemExit(
                f"{path} has two rows for prompt={key[0]} seed={key[1]}; the file "
                "mixes more than one run and cannot be paired")
        rows[key] = row
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def _summary_beside(events_path: Path) -> dict[str, Any] | None:
    suffix = "_event_rows.jsonl"
    stem = events_path.name[:-len(suffix)] if events_path.name.endswith(suffix) \
        else events_path.stem
    candidate = events_path.parent.parent / "summary" / f"{stem}_summary.json"
    if not candidate.is_file():
        return None
    try:
        return read_json(candidate)
    except (OSError, ValueError):
        return None


def check_comparable(m0_events: Path, m1_events: Path) -> dict[str, Any]:
    """Both arms must have been scored under one protocol on one eval set.

    Pairing arms that saw different prompts produces a difference that is not a
    difference between the models, so this fails rather than warns.  A missing
    summary is not evidence of compatibility either -- the prompt ids would
    still line up and the pairing would look perfectly healthy.
    """
    left, right = _summary_beside(m0_events), _summary_beside(m1_events)
    missing = [str(path) for path, summary in ((m0_events, left), (m1_events, right))
               if summary is None]
    if missing:
        raise SystemExit(
            "no summary/*.json beside " + ", ".join(missing) + "; the two arms "
            "cannot be shown to share an evaluation set, a prompt and a protocol")
    checks = {
        "protocol_version": ("protocol_version",),
        "eval_data_sha256": ("generation", "eval_data_sha256"),
        "prompt_rendered_sha256": ("generation", "prompt_rendered_sha256"),
    }
    shared: dict[str, Any] = {}
    for label, path in checks.items():
        def dig(document: Mapping[str, Any]) -> Any:
            value: Any = document
            for step in path:
                value = value.get(step) if isinstance(value, Mapping) else None
            return value

        a, b = dig(left), dig(right)
        if a != b:
            raise SystemExit(
                f"the two arms do not share the same {label} ({a!r} vs {b!r}); "
                "pairing them would compare two evaluations, not two models")
        shared[label] = a
    return shared


def make_pair(*, m0_tag: str, m1_tag: str, m0_events: Path, m1_events: Path,
              m0_responses: Path, m1_responses: Path, out_dir: Path,
              target: str = "answer") -> dict[str, Any]:
    """Write `event_rows.jsonl` + the two manifests the P0 analysers require."""
    for path in (m0_events, m1_events, m0_responses, m1_responses):
        if not Path(path).is_file():
            raise SystemExit(f"missing input: {path}")
    shared = check_comparable(Path(m0_events), Path(m1_events))
    left, right = _index(Path(m0_events)), _index(Path(m1_events))
    keys = sorted(set(left) & set(right))
    dropped = len(set(left) ^ set(right))
    if not keys:
        raise SystemExit(f"{m0_tag} and {m1_tag} share no (prompt, seed)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "event_rows.jsonl"
    partial = events_path.with_suffix(events_path.suffix + ".partial")
    prompts: set[str] = set()
    seeds: Counter = Counter()
    import json as _json
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for tag, source in ((M0, left), (M1, right)):
            for key in keys:
                row = dict(source[key])
                row["e1_model_tag"] = row.get("model_tag")
                row["model_tag"] = tag
                row["sample_id"] = f"{tag}:{canonical_key(row.get('key'))}:{row['seed']}"
                roles = set(row.get("selection_roles") or [])
                roles.add("prevalence")
                row["selection_roles"] = sorted(roles)
                handle.write(_json.dumps(row, ensure_ascii=False) + "\n")
                prompts.add(key[0])
                seeds[key[1]] += 1
    partial.replace(events_path)

    run_manifest = {
        "written_by": "tcr/paper/pairing.py",
        "pair": {"M0": m0_tag, "M1": m1_tag},
        "inputs": {
            "events_m0": str(m0_events), "events_m1": str(m1_events),
            "responses_m0": str(m0_responses), "responses_m1": str(m1_responses),
        },
        # Left empty on purpose: the per-response SHA / block / hash checks the
        # analysers already run are the meaningful ones, and freezing a
        # whole-file digest here would only re-check a file this step just read.
        "input_file_sha256": {},
        "selection": {"mode": "all", "n_prompts": len(prompts),
                      "n_rows": 2 * len(keys)},
        "protocol": {"target": target, "raw_onset_moved": False,
                     **shared},
        "n_dropped_unpaired_responses": dropped,
        "output": str(events_path),
    }
    write_json(out_dir / "run_manifest.json", run_manifest)
    write_json(out_dir / "selection_manifest.json", {
        "mode": "all",
        "n_prompts": len(prompts),
        "seeds": sorted(seeds),
        "note": "every evaluation record and every generation seed of both arms",
    })
    log("pair", f"{m0_tag} / {m1_tag}: {len(prompts)} prompts, {2 * len(keys)} rows"
                + (f", {dropped} unpaired dropped" if dropped else ""))
    if dropped:
        log("pair", f"[!] {dropped} response(s) had no counterpart; check that "
                    "both evaluations finished before trusting a paired rate")
    return run_manifest
