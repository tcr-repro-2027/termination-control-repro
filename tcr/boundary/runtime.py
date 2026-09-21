"""Model/tokenizer loading and frozen chat-prefix construction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from .constants import CHAT_ASSISTANT_NOTHINK, CHAT_USER_PREFIX


def enforce_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_tokenizer(model_path: str | Path):
    enforce_offline_environment()
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"local tokenizer/model directory does not exist: {path}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("S1 requires a fast tokenizer with offset mappings")
    return tokenizer


def frozen_nothink_prefix_text(prompt: str) -> str:
    return CHAT_USER_PREFIX + prompt + CHAT_ASSISTANT_NOTHINK


def audit_chat_template(tokenizer, probe_prompt: str = "S1-template-probe") -> None:
    """The frozen nothink prefix must equal apply_chat_template exactly."""
    frozen_ids = list(tokenizer.encode(frozen_nothink_prefix_text(probe_prompt), add_special_tokens=False))
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": probe_prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        padding=False,
        truncation=False,
        return_tensors=None,
        return_dict=True,
    )
    template_ids = templated["input_ids"] if isinstance(templated, dict) else templated
    if hasattr(template_ids, "input_ids"):
        template_ids = template_ids.input_ids
    template_ids = list(template_ids)
    if frozen_ids != template_ids:
        first = next(
            (i for i, (a, b) in enumerate(zip(frozen_ids, template_ids)) if a != b),
            min(len(frozen_ids), len(template_ids)),
        )
        raise RuntimeError(
            "frozen nothink prefix differs from tokenizer.apply_chat_template "
            f"(first divergence at token {first}); refusing to measure"
        )


def build_chat_ids(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.encode(frozen_nothink_prefix_text(prompt), add_special_tokens=False))


def stop_token_ids(model_path: str | Path, tokenizer) -> list[int]:
    """Stop-token id set exactly as the deployed engine sees it."""
    ids: set[int] = set()
    try:
        from transformers import GenerationConfig

        config = GenerationConfig.from_pretrained(str(model_path), local_files_only=True)
        eos = config.eos_token_id
        if isinstance(eos, int):
            ids.add(int(eos))
        elif isinstance(eos, Sequence):
            ids.update(int(value) for value in eos)
    except Exception:
        pass
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    if not ids:
        raise RuntimeError(f"no stop token ids resolvable for {model_path}")
    return sorted(ids)


def load_model(model_path: str | Path, *, device: str, attn_implementation: str | None = None):
    enforce_offline_environment()
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {path}")
    import torch
    from transformers import AutoModelForCausalLM

    backends = [attn_implementation] if attn_implementation else ["flash_attention_2", "sdpa"]
    last_error: Exception | None = None
    for backend in backends:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(path),
                dtype=torch.bfloat16,
                local_files_only=True,
                attn_implementation=backend,
            )
            model.to(device)
            model.eval()
            model.config.use_cache = True
            return model, backend
        except (ImportError, ValueError, OSError) as exc:  # flash-attn missing etc.
            last_error = exc
    raise RuntimeError(f"could not load model with backends {backends}: {last_error}")
