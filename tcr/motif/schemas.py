"""Typed artifact schemas used across stages."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

from .constants import SCHEMA_VERSION


def as_record(value: Any) -> dict[str, Any]:
    return dataclasses.asdict(value)


@dataclass(frozen=True)
class ParsedBlock:
    block_index: int
    char_start: int
    char_end: int
    token_start: int | None
    token_end: int | None
    raw_text: str
    canonical_signature: tuple[str, str, str, str]
    triple_signature: tuple[str, str, str]
    field_value_char_spans: dict[str, tuple[int, int]]
    parse_status: str = "strict_valid"
    separator_before: str = ""
    sequence_segment: int = 0


@dataclass(frozen=True)
class ParseDiagnostic:
    char_start: int
    char_end: int
    raw_text: str
    parse_status: str
    reason: str


@dataclass(frozen=True)
class BlockCycleRun:
    block_onset: int
    block_period: int
    block_num_repeats: int
    block_end: int
    coverage_blocks: int
    cycle_signatures: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class RawTokenLoop:
    token_onset: int
    token_period: int
    token_num_repeats: int
    loop_token_legacy: bool
    token_span_start: int | None = None
    token_span_end: int | None = None
    loop_content: str | None = None


@dataclass
class LoopAuditRow:
    stable_id: str
    model_tag: str
    split: str
    token_onset: int | None
    token_period: int | None
    token_num_repeats: int | None
    loop_token_legacy: bool
    block_spans: list[dict[str, Any]]
    block_runs: list[dict[str, Any]]
    block_period: int | None
    block_num_repeats: int | None
    loop_block_cycle_strict: bool
    char_run: dict[str, Any] | None
    loop_char_strict: bool
    loop_any_strict: bool
    alignment_type: str
    alignment_evidence: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    protocol_hash: str
    schema_version: str = SCHEMA_VERSION


@dataclass
class MotifProbe:
    probe_id: str
    stable_prompt_id: str
    split: str
    m: int
    k_1based: int
    k_0based: int
    anchor_region: str
    motif_block_indices_1based: list[int]
    c1_response: str
    c2_response: str
    c3_response: str
    repeat_surface: str
    recovery_surface: str
    motif_surface: str
    nonrecent_motif_surface: str | None
    serialization_mode: str
    lengths: dict[str, int]
    token_data: dict[str, Any]
    source_hashes: dict[str, str]
    protocol_hash: str
    invalid_reasons: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True)
class CandidateBranch:
    probe_id: str
    context_name: Literal["C1", "C2", "C3", "CONTROL"]
    candidate_name: Literal["R", "N"]
    full_text: str
    input_ids: tuple[int, ...]
    target_indices: tuple[int, ...]
    predictor_indices: tuple[int, ...]
    field_token_indices: dict[str, tuple[int, ...]]


@dataclass
class LoopGainRow:
    probe_id: str
    stable_prompt_id: str
    model_tag: str
    m: int
    anchor_region: str
    s_r_c1: float
    s_n_c1: float
    s_r_c2: float
    s_n_c2: float
    margin_c1: float
    margin_c2: float
    loop_gain: float
    repeat_attraction: float
    recovery_deficit: float
    loop_gain_patch: float | None
    mediated_loop_gain: float | None
    protocol_hash: str
    schema_version: str = SCHEMA_VERSION


@dataclass
class MediatorManifest:
    name: str
    source: str
    features_by_layer: dict[int, list[int]]
    effective_rank_by_layer: dict[int, int]
    subspace_hash_by_layer: dict[int, str]
    donor_model_hash: str
    protocol_hash: str
    validation_gate_hash: str | None = None
    schema_version: str = SCHEMA_VERSION


@dataclass
class DRSRow:
    index: int | None
    stable_id: str
    lg_drs: float | None
    lg_drs_per_block: float | None
    lg_drs_per_supervised_token: float | None
    lg_drs_boundary_fraction: float | None
    loss_plus: float | None
    loss_minus: float | None
    sft_loss: float | None
    epsilon: float
    supervised_tokens: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    n_blocks: int | None
    length_bucket: str | None
    block_count_bucket: str | None
    prompt_cluster: str | None
    surface_rep4: float | None
    gzip_ratio: float | None
    score_valid: bool
    invalid_reason: str | None
    status: str
    protocol_hash: str
    model_hash: str
    direction_hash: str
    schema_version: str = SCHEMA_VERSION


@dataclass
class GateResult:
    gate_name: str
    status: str
    blocking: bool
    scientific_validity: str
    criteria: dict[str, Any]
    observed: dict[str, Any]
    failed_criteria: list[str]
    blocked_inputs: list[str]
    upstream_gate_hashes: dict[str, str]
    protocol_hash: str
    artifact_hashes: dict[str, str]
    timestamp: str
    schema_version: str = SCHEMA_VERSION
