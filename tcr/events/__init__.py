"""Task-native event detection for long relation-extraction responses."""

from .block_parser import ParseResult, ParsedBlock, parse_relation_blocks
from .events import find_first_reuse
from .motif_capture import MotifRun, find_motif_runs, primary_motif_event

__all__ = [
    "MotifRun",
    "ParseResult",
    "ParsedBlock",
    "find_first_reuse",
    "find_motif_runs",
    "parse_relation_blocks",
    "primary_motif_event",
]

__version__ = "0.1.0"
