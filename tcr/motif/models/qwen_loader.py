"""Offline Qwen loading and model-layout validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


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
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for model loading") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), backend="tokenizers", local_files_only=True)
    if not tokenizer.is_fast:
        raise RuntimeError("LG-DRS requires a fast tokenizer with offset mappings")
    probe = tokenizer("offset check", add_special_tokens=False, return_offsets_mapping=True)
    if "offset_mapping" not in probe:
        raise RuntimeError("tokenizer did not return offset mappings")
    return tokenizer


def _torch_dtype(name: str):
    import torch

    mapping = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float32": torch.float32, "fp32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[name]


def load_causal_lm(
    model_path: str | Path,
    *,
    dtype: str = "bfloat16",
    attention_backend: str = "flash_attention_2",
    device: str | None = None,
    max_length: int = 32768,
):
    enforce_offline_environment()
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(f"local model directory does not exist: {path}")
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required for model loading") from exc
    kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(dtype),
        "local_files_only": True,
    }
    if attention_backend:
        kwargs["attn_implementation"] = attention_backend
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    config_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if config_limit and config_limit < max_length:
        raise RuntimeError(f"model max_position_embeddings={config_limit} is below required {max_length}")
    validate_decoder_layout(model)
    if device is not None:
        model.to(device)
    model.config.use_cache = False
    return model


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def base_model(model):
    model = unwrap_model(model)
    if getattr(model, "is_lgdrs_sparse_wrapper", False):
        model = model.causal_lm
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError(f"expected a causal LM exposing model.model, got {type(model).__name__}")
    return inner


def decoder_layers(model):
    layers = getattr(base_model(model), "layers", None)
    if layers is None:
        raise RuntimeError("expected decoder blocks at model.model.layers")
    return layers


def validate_decoder_layout(model) -> None:
    layers = decoder_layers(model)
    if len(layers) == 0:
        raise RuntimeError("model has no decoder layers")
    outer = unwrap_model(model)
    if not hasattr(outer, "lm_head"):
        raise RuntimeError("model has no lm_head")


def causal_lm(model):
    outer = unwrap_model(model)
    return outer.causal_lm if getattr(outer, "is_lgdrs_sparse_wrapper", False) else outer


def wrap_fsdp(model, *, local_rank: int, use_orig_params: bool = True, cpu_offload: bool = False):
    """Full-shard the model while preserving original parameter names."""
    import torch
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel, ShardingStrategy

    torch.cuda.set_device(local_rank)
    return FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=use_orig_params,
        device_id=torch.device("cuda", local_rank),
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=True,
        limit_all_gathers=True,
    )


def enable_activation_checkpointing(model) -> None:
    outer = causal_lm(model)
    if hasattr(outer, "gradient_checkpointing_enable"):
        outer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        raise RuntimeError("model does not support activation checkpointing")
