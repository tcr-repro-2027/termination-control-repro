# coding: utf-8
"""Shared helpers for the mechanism-paper closeout entries.

Nothing here re-derives a definition that already exists elsewhere.  The
repetition events, the stop-margin geometry, the residual hooks and the motif
surfaces all stay in their own modules; this package holds the model registry,
the paths, the paired statistics and the plotting style that every closeout
entry needs and none of them owns.
"""

from __future__ import annotations

__all__ = ["registry", "layout", "io", "stats", "gold", "figures", "pairing"]
