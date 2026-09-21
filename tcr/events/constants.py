"""Frozen scientific names and schema invariants."""

SCHEMA_VERSION = "relation-events-v0.1"
RELATION_FIELDS = ("source", "target", "relation", "description")
IDENTITY_FIELDS = RELATION_FIELDS[:3]

ALIGNMENT_TYPES = (
    "single_block_aligned",
    "multi_block_aligned",
    "phase_rotated_structured",
    "partial_structured",
    "non_structured",
    "unresolved",
    "no_legacy_orbit",
)
