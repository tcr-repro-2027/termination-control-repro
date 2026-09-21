# coding: utf-8
"""Build same-record OBR at the observed filter->clean conflict dose.

The target dose is fixed by the observed problem-4 rate. The construction
keeps every reinsertion in its own cleanv2 record and in the original
early/middle/late position bucket. It matches real A/AE and position profile
exactly, then minimizes degree, relative-position and token-length error.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcr.data.controlled.common import TokenCounter, compact_json, parse_entities, read_jsonl, stable_hash, write_jsonl  # noqa: E402
from tcr.data.controlled.obr import apply_obr, composition, describe_host, pair_record  # noqa: E402


def largest_remainder(counts: Mapping[tuple[str, str], int], target: int) -> dict[tuple[str, str], int]:
    total = max(sum(counts.values()), 1)
    exact = {key: value * target / total for key, value in counts.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = target - sum(result.values())
    for key in sorted(counts, key=lambda key: (-(exact[key] - result[key]), key)):
        if remaining <= 0:
            break
        result[key] += 1
        remaining -= 1
    return result


def category_flow(groups: Mapping[tuple[str, str], dict[str, Any]], bucket: str,
                  target_a: int, target_ae: int) -> dict[tuple[tuple[str, str], str], int]:
    """Allocate a bucket's A/AE demand without overusing record hosts."""
    selected = [(key, value) for key, value in groups.items() if key[1] == bucket]
    source, node_a, node_ae = 0, 1, 2
    nodes = {key: index + 3 for index, (key, _) in enumerate(selected)}
    sink = 3 + len(selected)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add(start: int, end: int, capacity: int) -> tuple[int, int, int]:
        graph[start].append([end, capacity, len(graph[end])])
        graph[end].append([start, 0, len(graph[start]) - 1])
        return start, len(graph[start]) - 1, capacity

    add(source, node_a, target_a)
    add(source, node_ae, target_ae)
    refs: dict[tuple[tuple[str, str], str], tuple[int, int, int]] = {}
    for key, value in selected:
        node = nodes[key]
        refs[(key, "A")] = add(node_a, node, len(value["A"]))
        refs[(key, "AE")] = add(node_ae, node, len(value["AE"]))
        add(node, sink, min(int(value["host_capacity"]), len(value["A"]) + len(value["AE"])))

    flow = 0
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = [source]
        for node in queue:
            for end, capacity, _ in graph[node]:
                if capacity > 0 and level[end] < 0:
                    level[end] = level[node] + 1
                    queue.append(end)
        if level[sink] < 0:
            break
        cursor = [0] * len(graph)

        def send(node: int, amount: int) -> int:
            if node == sink:
                return amount
            while cursor[node] < len(graph[node]):
                index = cursor[node]
                end, capacity, reverse = graph[node][index]
                if capacity > 0 and level[end] == level[node] + 1:
                    pushed = send(end, min(amount, capacity))
                    if pushed:
                        graph[node][index][1] -= pushed
                        graph[end][reverse][1] += pushed
                        return pushed
                cursor[node] += 1
            return 0

        while True:
            pushed = send(source, 10 ** 9)
            if not pushed:
                break
            flow += pushed
    if flow != target_a + target_ae:
        return {}
    return {key: capacity - graph[start][index][1]
            for key, (start, index, capacity) in refs.items()}


def profile(rows: list[Mapping[str, Any]], key: str) -> dict[str, float]:
    return composition(rows, key)


def distribution_delta(observed: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(observed.get(key, 0.0)) - float(reference.get(key, 0.0)), 4)
            for key in sorted(set(observed) | set(reference))}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def tokenizer_dir(explicit, repo_root):
    """--tokenizer, else $MODEL_ROOT/Qwen3-4B, else <repo>/models/Qwen3/Qwen3-4B."""
    import os
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("MODEL_ROOT", str(repo_root / "models" / "Qwen3"))) / "Qwen3-4B"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--data-out", default=None)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--tokenizer", default=None,
                        help="default: $MODEL_ROOT/Qwen3-4B, else <repo>/models/Qwen3/Qwen3-4B")
    parser.add_argument("--obr-dose", type=float, default=0.243,
                        help="default: 24.3 percent of all clean target blocks")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    datasets = repo_root / "datasets"
    stages = datasets / "stages"
    data_out = Path(args.data_out) if args.data_out else datasets / "controlled"
    report_out = Path(args.report_out) if args.report_out else datasets / "build_reports"
    data_out.mkdir(parents=True, exist_ok=True)
    report_out.mkdir(parents=True, exist_ok=True)
    clean_path = datasets / "cleanv2" / "train_supportclean_keep8.jsonl"
    rowmap_path = datasets / "cleanv2" / "train_supportclean_keep8_rowmap.jsonl"
    diff_path = datasets / "build_reports" / "filter_clean_problem4_diff.jsonl"
    filter_rowmap_path = stages / "filter" / "train_filter_rowmap.jsonl"

    records = read_jsonl(clean_path)
    rec_ids = [str(row["rec_id"]) for row in read_jsonl(rowmap_path)]
    if len(records) != len(rec_ids):
        raise ValueError("cleanv2 dataset and rowmap have different lengths")
    print(f"[obr] loaded {len(records)} clean records", flush=True)
    line_by_rec = {rec_id: line for line, rec_id in enumerate(rec_ids)}
    all_real_pool: list[dict[str, Any]] = []
    drops_by_rec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(diff_path):
        if row.get("split", "train") != "train":
            continue
        all_real_pool.append(row)
        if str(row["rec_id"]) in line_by_rec:
            drops_by_rec[str(row["rec_id"])].append(row)
    matchable_pool = [row for values in drops_by_rec.values() for row in values]
    total_blocks = sum(len(record["output"]) for record in records)
    target = round(total_blocks * args.obr_dose)
    if target != 142162 and abs(args.obr_dose - 0.243) < 1e-12:
        raise AssertionError(f"unexpected 24.3% target {target}")
    counter = TokenCounter(tokenizer_dir(args.tokenizer, repo_root))
    print(f"[obr] target={target}; computing same-record position capacity", flush=True)

    # Build only the hard feasibility graph: same record and position bucket.
    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"A": [], "AE": [], "host_capacity": 0})
    position_capacity = Counter()
    for rec_id, drops in drops_by_rec.items():
        line = line_by_rec[rec_id]
        record = records[line]
        host_counts = Counter(
            "early" if index / max(len(record["output"]) - 1, 1) < 0.25
            else "middle" if index / max(len(record["output"]) - 1, 1) < 0.75
            else "late"
            for index in range(len(record["output"])))
        for row in drops:
            key = (rec_id, str(row["position_bucket"]))
            groups[key][str(row["joint_type"])].append(row)
        for bucket, host_count in host_counts.items():
            key = (rec_id, bucket)
            if key in groups:
                groups[key]["host_capacity"] = host_count
    print(f"[obr] allocating profile strata across {len(groups)} record-position groups", flush=True)
    for (rec_id, bucket), value in groups.items():
        position_capacity[bucket] += min(value["host_capacity"], len(value["A"]) + len(value["AE"]))

    real_category = Counter((str(row["joint_type"]), str(row["position_bucket"])) for row in all_real_pool)
    category_target = largest_remainder(real_category, target)
    allocation: dict[tuple[tuple[str, str], str], int] = {}
    for bucket in ("early", "middle", "late"):
        bucket_alloc = category_flow(groups, bucket,
                                     category_target.get(("A", bucket), 0),
                                     category_target.get(("AE", bucket), 0))
        if not bucket_alloc:
            raise RuntimeError(f"cannot satisfy same-record {bucket} quota at 24.3%")
        allocation.update(bucket_alloc)

    selected_by_rec: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ((rec_id, bucket), joint_type), amount in allocation.items():
        if not amount:
            continue
        choices = sorted(groups[(rec_id, bucket)][joint_type], key=lambda row: (
            stable_hash(rec_id, bucket, joint_type, row["filter_block_index"])))
        selected_by_rec[rec_id].extend(choices[:amount])
    selected_total = sum(len(values) for values in selected_by_rec.values())
    if selected_total != target:
        raise AssertionError(f"allocated {selected_total}, expected {target}")
    print(f"[obr] selected {selected_total}; pairing hosts", flush=True)

    pairs_by_line: dict[int, list] = {}
    for rec_id, selected in selected_by_rec.items():
        line = line_by_rec[rec_id]
        pairs = pair_record(line, rec_id, records[line], selected, len(selected), counter)
        if len(pairs) != len(selected):
            raise AssertionError(f"{rec_id}: paired {len(pairs)} of {len(selected)} selected drops")
        pairs_by_line[line] = pairs
    pairs = [pair for values in pairs_by_line.values() for pair in values]
    if len(pairs) != target:
        raise AssertionError(f"realized {len(pairs)}, expected {target}")
    print(f"[obr] paired {len(pairs)}; materializing pair metadata", flush=True)

    selected_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    drop_lookup = {(rec_id, int(row["filter_block_index"])): row
                   for rec_id, values in drops_by_rec.items() for row in values}
    for pair in pairs:
        drop = drop_lookup[(pair.rec_id, pair.filter_block_index)]
        if pair.host_bucket != str(drop["position_bucket"]):
            raise AssertionError("cross-position OBR pair")
        row = {
            "condition": "obr", "rec_id": pair.rec_id, "line": pair.line,
            "block_index": pair.host_block_index,
            "filter_block_index": pair.filter_block_index,
            "joint_type": pair.joint_type, "host_bucket": pair.host_bucket,
            "drop_bucket": pair.drop_bucket, "token_delta": pair.token_delta,
            "degree_distance": pair.degree_distance,
            "position_distance": pair.position_distance,
            "token_balancing_move": pair.token_balancing_move,
            "source": pair.block["source"], "target": pair.block["target"],
            "source_only": bool(drop["source_only"]),
            "target_only": bool(drop["target_only"]),
            "both_endpoints": bool(drop["both_endpoints"]),
            "source_nested": bool(drop["source_nested"]),
            "target_nested": bool(drop["target_nested"]),
            "source_degree": int(drop["source_degree"]),
            "target_degree": int(drop["target_degree"]),
            "host_source_degree": pair.host_source_degree,
            "host_target_degree": pair.host_target_degree,
            "drop_relative_position": float(drop["relative_position"]),
            "host_relative_position": pair.host_relative_position,
        }
        pair_rows.append(row)
        selected_rows.append(dict(drop))

    signed_delta = absolute_delta = clean_tokens_total = 0
    record_deltas: list[float] = []
    def output_rows():
        nonlocal signed_delta, absolute_delta, clean_tokens_total
        for line, record in enumerate(records):
            before = counter.length(compact_json(record["output"]), cache=False)
            clean_tokens_total += before
            pair_list = pairs_by_line.get(line)
            if not pair_list:
                yield record
                continue
            payload = dict(record)
            payload["output"] = apply_obr(record, pair_list)
            after = counter.length(compact_json(payload["output"]), cache=False)
            signed_delta += after - before
            absolute_delta += abs(after - before)
            record_deltas.append(abs(after - before) / max(before, 1))
            yield payload
    write_jsonl(data_out / "train_obr.jsonl", output_rows())
    print("[obr] wrote train_obr.jsonl; updating manifests", flush=True)

    # The shared manifest must replace only OBR rows; ISC rows remain frozen.
    shared_manifest_path = report_out / "pair_manifest.jsonl"
    shared_rows = [row for row in read_jsonl(shared_manifest_path)
                   if str(row.get("condition")) != "obr"] if shared_manifest_path.exists() else []
    write_jsonl(shared_manifest_path, [*shared_rows, *pair_rows])
    write_jsonl(report_out / "obr_pair_manifest.jsonl", pair_rows)

    real_profiles = {key: profile(all_real_pool, key) for key in (
        "joint_type", "position_bucket", "source_only", "target_only",
        "both_endpoints", "source_nested", "target_nested")}
    selected_profiles = {key: profile(selected_rows, key) for key in real_profiles}
    exact_source = sum(row["source_degree"] == row["host_source_degree"] for row in pair_rows)
    exact_target = sum(row["target_degree"] == row["host_target_degree"] for row in pair_rows)
    exact_both = sum(row["source_degree"] == row["host_source_degree"] and
                     row["target_degree"] == row["host_target_degree"] for row in pair_rows)
    degree_distances = [float(row["degree_distance"]) for row in pair_rows]
    position_distances = [float(row["position_distance"]) for row in pair_rows]
    filter_blocks = sum(int(row["blocks"]) for row in read_jsonl(filter_rowmap_path))
    report = {
        "version": "e0-obr-rebuild-2.0",
        "construction": {
            "same_cleanv2_record": True,
            "hard_position_bucket": True,
            "degree_policy": "maximize exact source-target degree pairs; nearest log-degree residual",
            "position_policy": "same bucket, then nearest relative position",
            "token_policy": "nearest block token length after degree and position",
        },
        "clean": {"records": len(records), "blocks": total_blocks},
        "real_profile": {
            "drops": len(all_real_pool), "filter_blocks": filter_blocks,
            "edge_conflict_rate": round(len(all_real_pool) / max(filter_blocks, 1), 6),
            **real_profiles,
        },
        "matchable_source": {
            "same_record_drops": len(matchable_pool),
            "unavailable_no_cleanv2_record": len(all_real_pool) - len(matchable_pool),
        },
        "dose": {
            "requested_rho": args.obr_dose, "requested_pairs": target,
            "realized_pairs": len(pairs), "realized_rho": round(len(pairs) / total_blocks, 6),
            "records_touched": len(pairs_by_line),
            "same_record_ratio": 1.0,
            "same_position_bucket_ratio": 1.0,
            "same_record_position_capacity": dict(sorted(position_capacity.items())),
        },
        "selected_profile": selected_profiles,
        "profile_deviation_selected_minus_real": {
            key: distribution_delta(selected_profiles[key], real_profiles[key]) for key in real_profiles},
        "degree_matching": {
            "exact_source_ratio": round(exact_source / len(pair_rows), 6),
            "exact_target_ratio": round(exact_target / len(pair_rows), 6),
            "exact_both_ratio": round(exact_both / len(pair_rows), 6),
            "log_distance_mean": round(statistics.fmean(degree_distances), 6),
            "log_distance_p50": round(percentile(degree_distances, 0.5), 6),
            "log_distance_p95": round(percentile(degree_distances, 0.95), 6),
        },
        "position_matching": {
            "relative_distance_mean": round(statistics.fmean(position_distances), 6),
            "relative_distance_p50": round(percentile(position_distances, 0.5), 6),
            "relative_distance_p95": round(percentile(position_distances, 0.95), 6),
        },
        "token_budget": {
            "signed_delta_ratio": round(signed_delta / max(clean_tokens_total, 1), 6),
            "absolute_delta_ratio": round(absolute_delta / max(clean_tokens_total, 1), 6),
            "max_record_delta_ratio": round(max(record_deltas, default=0.0), 6),
            "records_over_1pct": sum(value > 0.01 for value in record_deltas),
            "same_bucket_token_balancing_moves": sum(
                bool(row["token_balancing_move"]) for row in pair_rows),
        },
    }
    (report_out / "obr_rebuild_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    scs_manifest_path = report_out / "scs_manifest.json"
    scs_manifest = json.loads(scs_manifest_path.read_text(encoding="utf-8")) if scs_manifest_path.exists() else {}
    scs_manifest["obr"] = {
        "pairs": len(pairs), "realized_rho": report["dose"]["realized_rho"],
        "records_touched": len(pairs_by_line), "same_record_ratio": 1.0,
        "same_position_bucket_ratio": 1.0, "obr_dose_used": args.obr_dose,
        "obr_target_blocks": target, "real_pool_size": len(all_real_pool),
        "same_record_source_pool_size": len(matchable_pool),
        "real_filter_edge_conflict_rate": report["real_profile"]["edge_conflict_rate"],
        "real_joint_type": real_profiles["joint_type"],
        "selected_joint_type": selected_profiles["joint_type"],
        "real_position": real_profiles["position_bucket"],
        "selected_position": selected_profiles["position_bucket"],
        "degree_matching": report["degree_matching"],
        "position_matching": report["position_matching"],
        "aggregate_signed_token_delta_ratio": report["token_budget"]["signed_delta_ratio"],
        "aggregate_absolute_token_delta_ratio": report["token_budget"]["absolute_delta_ratio"],
        "max_record_token_delta_ratio": report["token_budget"]["max_record_delta_ratio"],
        "records_over_1pct": report["token_budget"]["records_over_1pct"],
    }
    scs_manifest_path.write_text(json.dumps(scs_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "rho": report["dose"]["realized_rho"],
                      "degree": report["degree_matching"], "position": report["position_matching"],
                      "token_budget": report["token_budget"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
