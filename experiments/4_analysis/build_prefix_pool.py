# coding: utf-8
"""Build the symmetric natural-prefix pool R1 and R2 score every model on.

Why a pool at all
-----------------
The free-generation contrast tells us that the raw-data model repeats more.  It
cannot tell us whether that model, *in the same state*, chooses to continue
where the cleaned model chooses to stop -- the two models were never in the same
state, because each produced its own text.  Fixing the input AND the generated
prefix is what turns an aggregate difference into a comparable local decision.

Symmetry is the point of the construction: half the prefixes come from the
cleaned model's own trajectories and half from the raw model's, so the pool is
not biased toward the states one model happens to reach.  What symmetry does
NOT buy is on-policy status for everyone: a text prefix comes from exactly one
model's sampling, and replaying it under another model is a controlled
comparison, not that model's own trajectory.  Every anchor therefore records
which model produced it, and R1 reports the two sources separately.

Three kinds of position, per source (§4.2)
------------------------------------------
``natural_stop``      the last complete block of a response that closed the list
                      normally, with no capture, no orbit and no length cap --
                      where the donor model actually chose to end.  It does not
                      mean the reference relations were exhausted.
``early_remaining``   a boundary near half the reference list with at least four
                      reference entity pairs still unemitted, chosen WITHOUT
                      looking at any model's stop score.  This is the position
                      that can tell an intervention apart from a truncation.
``pre_first_reuse``   the complete-block boundary just before the first repeated
                      triple, so the exit decision is measured before any loop
                      exists, not after one has formed.

Anchor geometry is S1's, unchanged: the measurement point is the longest common
token prefix of the close variant and the continue variant, because Qwen's BPE
merges a block's closing ``"}`` with whatever follows it and the real decision
is that merged token, not a character boundary.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _path in (REPO_ROOT, HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tcr.paper import gold as goldlib                                     # noqa: E402
from tcr.paper import registry                                            # noqa: E402
from tcr.paper.io import iter_jsonl, log, write_compact, write_json, write_jsonl  # noqa: E402
from tcr.paper.layout import Layout                                       # noqa: E402
from tcr.prompt_template import build_extraction_relation_prompt  # noqa: E402
from tcr.boundary.anchors import (                                # noqa: E402
    AnchorDraft, build_anchor, donor_from_event_row, extract_close_text,
    extract_opener_texts, modal_text, tokenize_anchor_paths, validate_response,
)
from tcr.boundary.runtime import load_tokenizer                   # noqa: E402

#: Per source model, per role.  256 anchors per scale in total.
QUOTAS = {"natural_stop": 64, "early_remaining": 32, "pre_first_reuse": 32}

#: An early boundary must leave at least this many reference entity pairs
#: unemitted, so "did the intervention delete content that was still owed?" has
#: an observable answer.  It is a lower bound from the reference list, not a
#: claim that the document holds no other true relation.
MIN_REMAINING_GOLD_PAIRS = 4

#: A boundary needs enough of a response behind it for an inter-block opener to
#: exist, and enough ahead for the position to mean anything.
MIN_BOUNDARY_BLOCK = 4
MIN_BLOCKS_NATURAL = 6

SPLIT_SEED = 20260908
SELECT_SEED = 20260909


def responses_by_prompt(path: Path) -> dict[str, dict[str, Any]]:
    return {goldlib.canonical_key(row.get("key")): row for row in iter_jsonl(path)}


def candidates_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in iter_jsonl(path)}


def pick_boundary(candidate: Mapping[str, Any], role: str) -> int | None:
    """The 1-based boundary block for one role, or None if this response has none."""
    n_blocks = int(candidate["n_blocks"])
    stored = int(candidate["n_blocks_stored"])
    complete = candidate["identity_complete"]
    remaining = candidate["remaining_gold_pairs"]

    def usable(block_1based: int) -> bool:
        return (MIN_BOUNDARY_BLOCK <= block_1based <= min(stored, n_blocks)
                and bool(complete[block_1based - 1]))

    if role == "natural_stop":
        if candidate["hit_max"] or candidate["stable_orbit"] or candidate["semantic_capture"]:
            return None
        if not candidate["full_json_list_valid"]:
            return None
        if str(candidate.get("finish_reason", "")) not in ("", "stop"):
            return None
        if n_blocks < MIN_BLOCKS_NATURAL or n_blocks > stored:
            return None
        return n_blocks if usable(n_blocks) else None

    if role == "early_remaining":
        gold_blocks = max(int(candidate["gold_blocks"]), 1)
        target = max(MIN_BOUNDARY_BLOCK, int(round(0.5 * gold_blocks)))
        ceiling = min(stored, n_blocks - 1)
        if ceiling < MIN_BOUNDARY_BLOCK:
            return None
        capture_start = candidate.get("capture_second_copy_start_0based")
        if capture_start is not None:
            ceiling = min(ceiling, int(capture_start))  # stay before any loop
        # Walk outward from the target so the chosen point stays as close to
        # mid-list as the response allows, without ever consulting a stop score.
        for offset in range(0, ceiling + 1):
            for block in (target - offset, target + offset):
                if not (MIN_BOUNDARY_BLOCK <= block <= ceiling):
                    continue
                if not usable(block):
                    continue
                if int(remaining[block - 1]) >= MIN_REMAINING_GOLD_PAIRS:
                    return block
        return None

    if role == "pre_first_reuse":
        first_reuse = candidate.get("first_reuse_block_1based")
        if not first_reuse:
            return None
        block = int(first_reuse) - 1
        return block if usable(block) else None

    raise ValueError(f"unknown role {role!r}")


def draft_for(donor, boundary: int, candidate: Mapping[str, Any]) -> AnchorDraft:
    return AnchorDraft(
        anchor_type="pool",
        donor=donor,
        boundary_block_1based=boundary,
        gold_blocks=max(int(candidate["gold_blocks"]), 1),
        last_new_relaxed_block_1based=int(candidate["last_new_gold_pair_block_1based"]) or None,
    )


def build_scale(scale: str, layout: Layout, *, out_dir: Path, quotas: Mapping[str, int],
                tokenizer_path: Path) -> dict[str, Any]:
    sources = registry.R1_POOL_SOURCES[scale]
    log("pool", f"{scale}: sources {sources}")
    tokenizer = load_tokenizer(tokenizer_path)

    events: dict[str, dict[str, Any]] = {}
    candidates: dict[str, dict[str, Any]] = {}
    responses: dict[str, dict[str, Any]] = {}
    for tag in sources:
        path = layout.r0 / f"R0_prefix_candidates_{tag}.jsonl"
        if not path.is_file():
            raise SystemExit(
                f"{path} is missing; run run_r0.py first -- the pool selects its "
                "positions from the per-block table R0 writes")
        candidates[tag] = candidates_by_sample(path)
        events[tag] = {str(row["sample_id"]): row
                       for row in iter_jsonl(layout.events(tag))}
        responses[tag] = responses_by_prompt(layout.responses(tag))
        log("pool", f"  {tag}: {len(candidates[tag])} candidate responses")

    diagnostics: Counter = Counter()
    global_openers: list[str] = []
    close_texts: list[str] = []
    built: list[Any] = []
    used_responses: set[str] = set()
    used_prompt_role: set[tuple[str, str, str]] = set()

    def prepare(tag: str, sample_id: str):
        row = events[tag].get(sample_id)
        if row is None:
            diagnostics["missing_event_row"] += 1
            return None
        donor = donor_from_event_row(row)
        record = responses[tag].get(donor.prompt_id)
        if record is None:
            diagnostics["missing_response_record"] += 1
            return None
        text = None
        finish_reason = ""
        for sample in record.get("responses", []) or []:
            if int(sample.get("seed", -1)) == donor.seed:
                text = str(sample.get("response", "") or "")
                finish_reason = str(sample.get("finish_reason", "") or "")
                break
        if not text:
            diagnostics["missing_response_text"] += 1
            return None
        try:
            validate_response(donor, text)
        except ValueError:
            diagnostics["response_sha_mismatch"] += 1
            return None
        prompt_text = build_extraction_relation_prompt(
            text=str(record.get("text", "")),
            entities_str=str(record.get("entities_str", "")))
        return donor, text, prompt_text, finish_reason

    def make(tag: str, role: str, sample_id: str, boundary: int, close_text: str,
             close_source: str):
        prepared = prepare(tag, sample_id)
        if prepared is None:
            return None
        donor, text, prompt_text, finish_reason = prepared
        own_openers = extract_opener_texts(donor, text, before_block_1based=boundary)
        opener = modal_text(own_openers)
        opener_source = "own_modal"
        if opener is None:
            opener = modal_text(global_openers)
            opener_source = "global_modal"
        if opener is None:
            diagnostics[f"{role}_no_opener"] += 1
            return None
        char_end = donor.blocks[boundary - 1].char_end
        payload, reason = tokenize_anchor_paths(
            tokenizer, text=text, char_end=char_end,
            close_text=close_text, opener_text=opener)
        if payload is None:
            diagnostics[f"{role}_{reason}"] += 1
            return None
        candidate = candidates[tag][sample_id]
        anchor = build_anchor(
            draft_for(donor, boundary, candidate), payload,
            anchor_type=role, boundary_block_1based=boundary,
            prompt_text=prompt_text, finish_reason=finish_reason,
            close_source=close_source, opener_source=opener_source)
        record = anchor.to_dict()
        record["role"] = role
        record["scale"] = scale
        record["source_tag"] = tag
        record["remaining_gold_pairs"] = int(candidate["remaining_gold_pairs"][boundary - 1])
        record["remaining_gold_triples"] = int(candidate["remaining_gold_triples"][boundary - 1])
        record["gold_pairs_total"] = int(candidate["gold_pairs"])
        record["gold_pairs_hit"] = int(candidate["gold_pairs_hit"][boundary - 1])
        record["donor_n_blocks"] = int(candidate["n_blocks"])
        record["donor_hit_max"] = bool(candidate["hit_max"])
        record["donor_semantic_capture"] = bool(candidate["semantic_capture"])
        return record, own_openers

    def eligible(tag: str, role: str) -> list[tuple[str, int]]:
        """(sample_id, boundary) for every response of `tag` usable in `role`."""
        found: list[tuple[str, int]] = []
        for sample_id, candidate in candidates[tag].items():
            boundary = pick_boundary(candidate, role)
            if boundary is not None:
                found.append((sample_id, boundary))
        return found

    # Pass 1: natural stops, whose real close tails also define the modal close
    # text every other role has to borrow (an early boundary has no close tail
    # of its own -- the donor did not stop there).
    rng = random.Random(SELECT_SEED)
    order: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for tag in sources:
        for role in quotas:
            found = eligible(tag, role)
            # Prefer prompts that are not yet used anywhere, then a stable
            # shuffle, so the two sources cover different records where they can.
            rng.shuffle(found)
            order[(tag, role)] = found
            diagnostics[f"eligible_{role}_{tag}"] = len(found)

    anchors: list[dict[str, Any]] = []
    for tag in sources:
        taken = 0
        for sample_id, boundary in order[(tag, "natural_stop")]:
            if taken >= quotas["natural_stop"]:
                break
            prompt_id = candidates[tag][sample_id]["stable_prompt_id"]
            if (tag, "natural_stop", prompt_id) in used_prompt_role:
                continue
            prepared = prepare(tag, sample_id)
            if prepared is None:
                continue
            donor, text, _prompt, _finish = prepared
            close_text = extract_close_text(text, donor.blocks[boundary - 1].char_end)
            if close_text is None:
                diagnostics["natural_stop_atypical_close_tail"] += 1
                continue
            made = make(tag, "natural_stop", sample_id, boundary, close_text, "own")
            if made is None:
                continue
            record, own_openers = made
            global_openers.extend(own_openers)
            close_texts.append(close_text)
            anchors.append(record)
            used_responses.add(sample_id)
            used_prompt_role.add((tag, "natural_stop", prompt_id))
            taken += 1

    close_modal = modal_text(close_texts)
    if close_modal is None:
        raise SystemExit(
            f"{scale}: no natural-stop anchor could be built, so there is no "
            f"close tail for the other roles. diagnostics={dict(diagnostics)}")

    # Pass 2: the two roles that need a borrowed close tail.
    for tag in sources:
        for role in ("early_remaining", "pre_first_reuse"):
            taken = 0
            for sample_id, boundary in order[(tag, role)]:
                if taken >= quotas[role]:
                    break
                if sample_id in used_responses:
                    continue
                prompt_id = candidates[tag][sample_id]["stable_prompt_id"]
                if (tag, role, prompt_id) in used_prompt_role:
                    continue
                made = make(tag, role, sample_id, boundary, close_modal, "modal")
                if made is None:
                    continue
                record, _own = made
                anchors.append(record)
                used_responses.add(sample_id)
                used_prompt_role.add((tag, role, prompt_id))
                taken += 1

    # Dev / test split by PROMPT: every anchor of one record lands on the same
    # side, or a direction fitted on the dev half would have already seen the
    # test half's record.
    prompts = sorted({str(anchor["prompt_id"]) for anchor in anchors})
    split_rng = random.Random(SPLIT_SEED)
    split_rng.shuffle(prompts)
    dev = set(prompts[: len(prompts) // 2])
    for anchor in anchors:
        anchor["split"] = "dev" if str(anchor["prompt_id"]) in dev else "test"

    anchors.sort(key=lambda row: str(row["anchor_id"]))
    pool_path = out_dir / f"prefix_pool_{scale}.jsonl"
    write_jsonl(pool_path, anchors)

    counts = Counter((anchor["source_tag"], anchor["role"], anchor["split"])
                     for anchor in anchors)
    manifest = {
        "scale": scale,
        "sources": list(sources),
        "quotas": dict(quotas),
        "n_anchors": len(anchors),
        "n_prompts": len(prompts),
        "n_dev_prompts": len(dev),
        "modal_close_tail_text": close_modal,
        "global_opener_text": modal_text(global_openers),
        "by_source_role_split": {f"{a}|{b}|{c}": n for (a, b, c), n in sorted(counts.items())},
        "by_role": dict(Counter(anchor["role"] for anchor in anchors)),
        "by_split": dict(Counter(anchor["split"] for anchor in anchors)),
        "boundary_aligned": sum(bool(anchor["boundary_aligned"]) for anchor in anchors),
        "gap_text_counts": dict(Counter(anchor["gap_text"] for anchor in anchors)),
        "tokenizer": str(tokenizer_path),
        "diagnostics": dict(diagnostics),
        "output": str(pool_path),
    }
    write_json(out_dir / f"prefix_pool_{scale}_manifest.json", manifest)
    log("pool", f"{scale}: {len(anchors)} anchors over {len(prompts)} prompts "
                f"({manifest['by_role']}, {manifest['by_split']})")
    for (tag, role, split), count in sorted(counts.items()):
        log("pool", f"    {tag:28s} {role:16s} {split:4s} {count}")
    under = {role: quota * len(sources) - manifest["by_role"].get(role, 0)
             for role, quota in quotas.items()}
    for role, gap in under.items():
        if gap > 0:
            log("pool", f"  [!] {role}: {gap} short of the quota; the actual "
                        "count is what gets reported, not the target")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scales", default="4B,8B")
    parser.add_argument("--tokenizer", default=None,
                        help="default: the base model of each scale")
    parser.add_argument("--natural-stop", type=int, default=QUOTAS["natural_stop"])
    parser.add_argument("--early-remaining", type=int, default=QUOTAS["early_remaining"])
    parser.add_argument("--pre-first-reuse", type=int, default=QUOTAS["pre_first_reuse"])
    args = parser.parse_args()

    layout = Layout.from_env()
    layout.ensure()
    quotas = {"natural_stop": args.natural_stop,
              "early_remaining": args.early_remaining,
              "pre_first_reuse": args.pre_first_reuse}
    manifests = []
    for scale in [value.strip() for value in args.scales.split(",") if value.strip()]:
        if scale not in registry.R1_POOL_SOURCES:
            raise SystemExit(f"no pool sources declared for scale {scale!r}")
        tokenizer_path = Path(args.tokenizer) if args.tokenizer else layout.base_model(scale)
        manifests.append(build_scale(scale, layout, out_dir=layout.r1,
                                     quotas=quotas, tokenizer_path=tokenizer_path))

    lines = []
    for manifest in manifests:
        lines.append(f"[{manifest['scale']}] anchors={manifest['n_anchors']} "
                     f"prompts={manifest['n_prompts']} dev_prompts={manifest['n_dev_prompts']}")
        lines.append("  roles=" + ", ".join(f"{k}:{v}" for k, v in
                                            sorted(manifest["by_role"].items())))
        lines.append("  per source/role/split=" + ", ".join(
            f"{k}:{v}" for k, v in sorted(manifest["by_source_role_split"].items())))
    write_compact(layout.compact / "R1_pool.txt", "[R1 prefix pool]", lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
