# coding=utf-8
"""Input-support prompts using the shared training template.

The builder is imported from ``tcr.prompt_template`` so support-probe margins
and natural-generation measurements use the same base instructions.
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
    "PROMPT_TEMPLATE_IS_TRUNCATED", "build_extraction_relation_prompt",
    "build_user_message", "prompt_fingerprint", "rendered_digest",
    "verify_against",
]

_PROBE_TEXT = "__E2_PROBE_TEXT__"
_PROBE_ENTITIES = "__E2_PROBE_ENTITIES__"


def build_user_message(text: str, entities_str: str, instruction: str = "") -> str:
    """The full user turn: the frozen template, plus any control instruction.

    The instruction is APPENDED, after the template's own `# Output` line, so
    the frozen part is byte-identical across every variant of an anchor and the
    only difference between a control and its neutral counterpart is the
    appended block itself."""
    return build_extraction_relation_prompt(text=text,
                                            entities_str=entities_str) + instruction


def rendered_digest(builder=build_extraction_relation_prompt) -> str:
    rendered = builder(text=_PROBE_TEXT, entities_str=_PROBE_ENTITIES)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def prompt_fingerprint() -> Dict[str, Any]:
    rendered = build_extraction_relation_prompt(text=_PROBE_TEXT,
                                                entities_str=_PROBE_ENTITIES)
    return {
        "prompt_source": "tcr/prompt_template.py "
                         "(the single template used for training and evaluation)",
        "prompt_is_truncated": bool(PROMPT_TEMPLATE_IS_TRUNCATED),
        "prompt_rendered_sha256": rendered_digest(),
        "prompt_rendered_chars": len(rendered),
    }


def verify_against(training_module_path: str | Path) -> Dict[str, Any]:
    """Compare the shared template with a supplied training module."""
    path = Path(training_module_path)
    report: Dict[str, Any] = {"training_module": str(path),
                              "available": path.is_file(),
                              "vendored_rendered_sha256": rendered_digest()}
    if not report["available"]:
        report["match"] = None
        return report
    spec = importlib.util.spec_from_file_location("_e2_training_prompt", path)
    if spec is None or spec.loader is None:            # pragma: no cover
        raise ImportError(f"cannot load prompt module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = rendered_digest(module.build_extraction_relation_prompt)
    report["training_rendered_sha256"] = digest
    report["training_is_truncated"] = bool(
        getattr(module, "PROMPT_TEMPLATE_IS_TRUNCATED", True))
    report["match"] = (digest == report["vendored_rendered_sha256"]
                       and not report["training_is_truncated"])
    return report
