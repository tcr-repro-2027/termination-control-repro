# coding: utf-8
"""Independent, manifest-to-file verification for the rebuilt OBR arm."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcr.data.controlled.common import iter_jsonl, norm, position_bucket, read_jsonl, support_state, parse_entities  # noqa: E402


def main() -> int:
    # optional: --repo-root <dir containing datasets/>
    repo = (Path(sys.argv[sys.argv.index("--repo-root") + 1]).resolve()
            if "--repo-root" in sys.argv else REPO_ROOT)
    datasets = repo / "datasets"
    base_path = datasets / "cleanv2" / "train_supportclean_keep8.jsonl"
    variant_path = datasets / "controlled" / "train_obr.jsonl"
    rowmap_path = datasets / "cleanv2" / "train_supportclean_keep8_rowmap.jsonl"
    pairs_path = datasets / "controlled" / "obr_pair_manifest.jsonl"
    rec_ids = [str(row["rec_id"]) for row in read_jsonl(rowmap_path)]
    pairs_by_line: dict[int, list[dict]] = defaultdict(list)
    for pair in iter_jsonl(pairs_path):
        pairs_by_line[int(pair["line"])].append(pair)

    errors: list[str] = []
    changed_blocks = conflicts = 0
    source_degree_mismatch = target_degree_mismatch = 0
    source_exact = target_exact = both_exact = 0
    for line, (base, variant) in enumerate(zip(iter_jsonl(base_path), iter_jsonl(variant_path))):
        if line >= len(rec_ids):
            errors.append("variant has more records than rowmap")
            break
        pairs = pairs_by_line.get(line, [])
        if len(base["output"]) != len(variant["output"]):
            errors.append(f"line {line}: block count changed")
        changed = {index for index, (before, after) in enumerate(zip(base["output"], variant["output"]))
                   if before != after}
        expected = {int(pair["block_index"]) for pair in pairs}
        if changed != expected:
            errors.append(f"line {line}: changed={len(changed)} expected={len(expected)}")
        changed_blocks += len(changed)
        degrees = Counter()
        for block in base["output"]:
            degrees[norm(block["source"])] += 1
            degrees[norm(block["target"])] += 1
        entity_set = set(parse_entities(variant["entities_str"]))
        state = support_state(variant["text"], entity_set, variant["output"])
        for index, pair in ((int(pair["block_index"]), pair) for pair in pairs):
            if str(pair["rec_id"]) != rec_ids[line]:
                errors.append(f"line {line}: rec_id mismatch")
            if position_bucket(index, len(base["output"])) != str(pair["drop_bucket"]):
                errors.append(f"line {line}: cross-bucket pair")
            block = variant["output"][index]
            if any(block[name] != pair[name] for name in ("source", "target")):
                errors.append(f"line {line}: output block does not equal manifest")
            host_source = degrees[norm(base["output"][index]["source"])]
            host_target = degrees[norm(base["output"][index]["target"])]
            source_degree_mismatch += host_source != int(pair["host_source_degree"])
            target_degree_mismatch += host_target != int(pair["host_target_degree"])
            source_exact += int(pair["source_degree"]) == host_source
            target_exact += int(pair["target_degree"]) == host_target
            both_exact += (int(pair["source_degree"]) == host_source and
                           int(pair["target_degree"]) == host_target)
            admissible, evidenced = state[index]
            if admissible and evidenced:
                errors.append(f"line {line}: OBR block remains support-valid")
            else:
                conflicts += 1
    if len(rec_ids) != line + 1:
        errors.append(f"record count {line + 1} != {len(rec_ids)}")
    pair_total = sum(len(values) for values in pairs_by_line.values())
    if changed_blocks != pair_total:
        errors.append(f"changed blocks {changed_blocks} != manifest pairs {pair_total}")
    if conflicts != pair_total:
        errors.append(f"conflicts {conflicts} != manifest pairs {pair_total}")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "records": len(rec_ids), "pairs": pair_total,
        "changed_blocks": changed_blocks, "support_conflicts": conflicts,
        "same_record_and_position_bucket": not any("mismatch" in error or "bucket" in error for error in errors),
        "host_degree_metadata_mismatch": {"source": source_degree_mismatch, "target": target_degree_mismatch},
        "recomputed_exact_degree_ratio": {
            "source": round(source_exact / max(pair_total, 1), 6),
            "target": round(target_exact / max(pair_total, 1), 6),
            "both": round(both_exact / max(pair_total, 1), 6),
        },
        "errors": errors[:100], "error_count": len(errors),
    }
    (datasets / "build_reports").mkdir(parents=True, exist_ok=True)
    (datasets / "build_reports" / "obr_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
