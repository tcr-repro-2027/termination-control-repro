# coding=utf-8
"""Evaluation prompt and reproducibility fingerprints.

Training and evaluation import the complete builder from
``tcr.prompt_template``. ``verify_against()`` can compare its rendered output
with a supplied training module before evaluation starts.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Dict

from ..prompt_template import (
    PROMPT_TEMPLATE_IS_TRUNCATED,
    build_extraction_relation_prompt,
)

__all__ = [
    "PROMPT_TEMPLATE_IS_TRUNCATED",
    "build_extraction_relation_prompt",
    "prompt_fingerprint",
    "rendered_digest",
    "verify_against",
]

_PROBE_TEXT = "__E1_PROBE_TEXT__"
_PROBE_ENTITIES = "__E1_PROBE_ENTITIES__"


def rendered_digest(builder=build_extraction_relation_prompt) -> str:
    """sha256 of the template rendered with fixed probe fields.

    Documentation edits do not change the rendered prompt and therefore do
    not affect this digest."""
    rendered = builder(text=_PROBE_TEXT, entities_str=_PROBE_ENTITIES)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def prompt_fingerprint() -> Dict[str, Any]:
    """Digest and shape of the shared template, for the run manifest."""
    rendered = build_extraction_relation_prompt(text=_PROBE_TEXT,
                                                entities_str=_PROBE_ENTITIES)
    return {
        "prompt_source": "tcr/prompt_template.py "
                         "(the single template used for training and evaluation)",
        "prompt_is_truncated": bool(PROMPT_TEMPLATE_IS_TRUNCATED),
        "prompt_rendered_sha256": rendered_digest(),
        "prompt_rendered_chars": len(rendered),
        "prompt_file_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1]
             / "prompt_template.py").read_bytes()
        ).hexdigest(),
    }


def verify_against(training_module_path: str | Path) -> Dict[str, Any]:
    """Compare the shared template with a supplied training module.

    Returns a report dict; ``report["match"] is True`` only when both render
    identically.  A missing training module is reported as ``"available":
    False``; in that case no comparison is available.
    """
    path = Path(training_module_path)
    report: Dict[str, Any] = {
        "training_module": str(path),
        "available": path.is_file(),
        "vendored_rendered_sha256": rendered_digest(),
    }
    if not report["available"]:
        report["match"] = None
        return report

    spec = importlib.util.spec_from_file_location("_e1_training_prompt", path)
    if spec is None or spec.loader is None:      # pragma: no cover - unreachable
        raise ImportError(f"cannot load prompt module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    training_digest = rendered_digest(module.build_extraction_relation_prompt)
    report["training_rendered_sha256"] = training_digest
    report["training_is_truncated"] = bool(
        getattr(module, "PROMPT_TEMPLATE_IS_TRUNCATED", True)
    )
    report["match"] = (training_digest == report["vendored_rendered_sha256"]
                       and not report["training_is_truncated"])
    return report
