"""Project-wide frozen names and invariants."""

from __future__ import annotations

SCHEMA_VERSION = "tcr.motif-v1.1"
MAX_LENGTH = 32768
IDENTITY_FIELDS = ("source", "target", "relation")
RELATION_FIELDS = (*IDENTITY_FIELDS, "description")

GATE_STATUSES = {
    "PASS",
    "FAIL",
    "BLOCKED_INPUT",
    "NOT_RUN",
    "NON_SCIENTIFIC",
}

ALIGNMENT_TYPES = {
    "single_block_aligned",
    "multi_block_aligned",
    "phase_rotated_structured",
    "partial_structured",
    "non_structured",
    "unresolved",
}

STAGES = (
    "preflight",
    "p0_detect",
    "p0_freeze_m",
    "p1_probes",
    "p2_behavior",
    "p3_fcore",
    "p3_discover",
    "c5_external",
    "p4_loss",
    "p4_adam",
    "p4_risk",
    "p4_fdcal",
    "n3_one_step",
    "p4_score",
    "gate_i",
    "gate_t",
    "report",
    "all",
)

GPU_STAGES = {
    "p2_behavior",
    "p3_fcore",
    "p3_discover",
    "c5_external",
    "p4_loss",
    "p4_adam",
    "p4_risk",
    "p4_fdcal",
    "n3_one_step",
    "p4_score",
    "gate_i",
    "gate_t",
}

STAGE_GATE = {
    "p0_detect": "P0",
    "p1_probes": "P1",
    "p2_behavior": "P2",
    "p3_fcore": "P3",
    "p3_discover": "P3",
    "c5_external": "C5",
    "p4_loss": "N2",
    "p4_fdcal": "N0",
    "n3_one_step": "N3",
    "gate_i": "I",
    "gate_t": "T",
}

# Stages that must not run unless all listed machine-readable gates are PASS.
UPSTREAM_GATES = {
    "p0_freeze_m": ("P0",),
    "p1_probes": ("P0",),
    "p2_behavior": ("P1",),
    "p3_fcore": ("P2",),
    "p3_discover": ("P2",),
    "p4_loss": ("P2", "P3"),
    "p4_adam": ("P2", "P3", "N2"),
    "p4_risk": ("P2", "P3", "N2"),
    "p4_fdcal": ("P2", "P3", "N2"),
    "n3_one_step": ("P2", "P3", "N0", "N1", "N2"),
    "p4_score": ("P2", "P3", "N0", "N1", "N2", "N3"),
    "gate_i": ("P2", "P3", "N0", "N1", "N2", "N3"),
    "gate_t": ("P2", "P3", "N0", "N1", "N2", "N3", "I"),
}
