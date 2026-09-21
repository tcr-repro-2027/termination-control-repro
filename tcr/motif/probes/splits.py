"""File-order-independent stable hash splits."""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Any


SPLITS = (
    ("detector_calib", 0.10),
    ("construct", 0.30),
    ("direction", 0.20),
    ("value", 0.20),
    ("external", 0.20),
)


def normalize_prompt(prompt: str) -> str:
    return unicodedata.normalize("NFC", prompt.replace("\r\n", "\n").replace("\r", "\n").strip())


def stable_sample_id(row: dict[str, Any], prompt: str | None = None) -> str:
    if row.get("index") is not None:
        return f"index:{int(row['index'])}"
    if row.get("key") is not None:
        try:
            return f"index:{int(row['key']) - 1}"
        except (TypeError, ValueError):
            return f"key:{row['key']}"
    prompt = prompt if prompt is not None else str(row.get("prompt", row.get("input", "")))
    if not prompt:
        raise ValueError("row lacks index/key and a prompt for fallback hashing")
    digest = hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()
    return f"prompt_sha256:{digest}"


def assign_split(stable_id: str, *, salt: str = "tcr.motif-v1.1") -> str:
    digest = hashlib.sha256(f"{salt}|{stable_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest, "big") / (1 << 256)
    cumulative = 0.0
    for name, fraction in SPLITS:
        cumulative += fraction
        if value < cumulative:
            return name
    return SPLITS[-1][0]


def construct_subsplit(stable_id: str, discovery_fraction: float = 0.5) -> str:
    value = int.from_bytes(hashlib.sha256(f"tcr.motif-feature|{stable_id}".encode()).digest(), "big") / (1 << 256)
    return "discovery" if value < discovery_fraction else "validation"
