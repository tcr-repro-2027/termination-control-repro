"""Residual-stream steering with deployment-exact resampling readouts.

The steering hook adds a fixed vector to the residual stream at ONE layer for
every position at or after the anchor boundary (the last prefix position on
the first forward, and every subsequently generated position).  The clamp
MECHANICS follow the 07b/08e convention (resid_post, zero-based layers); the
steered content — d_stop — is S2's own fresh identification.

Per (anchor, condition) one pass yields, from the SAME sampling run:

* boundary first-token metrics (policy/raw margin of the S1 decision tokens,
  close-token top-k/p survival) read off the step-0 logits;
* the resampled stop hazard under the deployed nothink sampler
  (identical order: additive presence penalty on output tokens, temperature,
  top-k, top-p; parameters asserted against the frozen protocol).
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import ALPHA_GRID, K_RESAMPLE, MAX_NEW_TOKENS, N_RANDOM_DIRECTIONS, RANDOM_SEED, SAMPLE_BATCH


class SteeringHook:
    """Adds ``vector`` to resid_post of one decoder block, boundary-onward."""

    def __init__(self, model, layer: int, device):
        self._block = model.model.layers[layer]
        self._vector = None
        self._boundary_index: int | None = None
        self._device = device
        self._handle = self._block.register_forward_hook(self._hook)

    def configure(self, vector, boundary_index: int) -> None:
        self._vector = vector
        self._boundary_index = int(boundary_index)

    def disable(self) -> None:
        self._vector = None

    def _hook(self, _module, _inputs, output):
        if self._vector is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        if seq_len > 1:
            start = self._boundary_index if self._boundary_index is not None else seq_len - 1
            start = max(0, min(start, seq_len - 1))
            hidden[:, start:, :] += self._vector.to(hidden.dtype)
        else:
            hidden[:, :, :] += self._vector.to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    def remove(self) -> None:
        self._handle.remove()


def build_conditions(
    d_raw: np.ndarray,
    *,
    model_tag: str,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    n_random: int = N_RANDOM_DIRECTIONS,
    random_seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    """Steering conditions for one model.

    M1 conditions SUBTRACT the damage direction (restore toward M0); M0
    conditions ADD it (induce damage).  Random-matched controls (R-b) use
    fixed-seed Gaussian directions scaled to the same norm per alpha.
    """
    if model_tag not in {"M0", "M1"}:
        raise ValueError(f"steering supports M0/M1, got {model_tag!r}")
    sign = -1.0 if model_tag == "M1" else 1.0
    d = np.asarray(d_raw, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    if norm <= 0:
        raise ValueError("d_raw has zero norm")
    rng = np.random.default_rng(random_seed)
    randoms = []
    for index in range(n_random):
        vec = rng.normal(size=d.shape[0])
        vec = vec / np.linalg.norm(vec) * norm
        randoms.append(vec)
    conditions: list[dict[str, Any]] = [
        {"condition": "baseline", "family": "baseline", "alpha": 0.0, "vector": None}
    ]
    for alpha in alpha_grid:
        conditions.append(
            {
                "condition": f"dstop_a{alpha:g}",
                "family": "dstop",
                "alpha": float(alpha),
                "vector": sign * alpha * d,
            }
        )
        for index, vec in enumerate(randoms):
            conditions.append(
                {
                    "condition": f"rand{index}_a{alpha:g}",
                    "family": "random",
                    "alpha": float(alpha),
                    "vector": sign * alpha * vec,
                }
            )
    return conditions


def steered_measure(
    model,
    tokenizer,
    anchor: Mapping[str, Any],
    *,
    chat_ids: Sequence[int],
    stop_ids: Sequence[int],
    hook: SteeringHook,
    vector,
    device: str,
    k_resample: int = K_RESAMPLE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sample_batch: int = SAMPLE_BATCH,
    condition_name: str = "baseline",
) -> dict[str, Any]:
    """One (anchor, condition) measurement; hazard + boundary metrics."""
    import torch

    from tcr.boundary.constants import NOTHINK_SAMPLING
    from tcr.boundary.policy import (
        classify_continuation,
        policy_logprobs_np,
        raw_logprobs_np,
        topk_topp_keep_mask_np,
    )
    from tcr.boundary.measure import _model_forward, _warpers

    prefix_response_ids = [int(v) for v in anchor["prefix_response_ids"]]
    prefix = list(chat_ids) + list(prefix_response_ids)
    boundary_index = len(prefix) - 1
    presence_base = {int(v) for v in prefix_response_ids}
    close_first = int(anchor["close_tail_ids"][0])
    cont_first = int(anchor["continue_ids"][0])
    stop_set = {int(v) for v in stop_ids}

    if vector is not None:
        hook.configure(torch.tensor(vector, dtype=torch.float32, device=device), boundary_index)
    else:
        hook.disable()

    top_k_warper, top_p_warper = _warpers()
    presence_penalty = float(NOTHINK_SAMPLING["presence_penalty"])
    temperature = float(NOTHINK_SAMPLING["temperature"])
    vocab = model.get_output_embeddings().weight.shape[0]

    digest = hashlib.sha256(f"{anchor['anchor_id']}::{condition_name}".encode()).hexdigest()
    base_seed = int(digest[:8], 16)

    texts: list[str] = []
    ended_flags: list[bool] = []
    boundary_logits_np: np.ndarray | None = None
    chunk_start = 0
    while chunk_start < k_resample:
        batch = min(sample_batch, k_resample - chunk_start)
        input_ids = torch.tensor([prefix] * batch, dtype=torch.long, device=device)
        presence = torch.zeros((batch, vocab), dtype=torch.bool, device=device)
        response_ids = torch.tensor(sorted(presence_base), dtype=torch.long, device=device)
        presence[:, response_ids] = True
        generator = torch.Generator(device=device)
        generator.manual_seed(base_seed * 1000003 + chunk_start)

        output = _model_forward(model, keep_last=1, input_ids=input_ids, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        if boundary_logits_np is None:
            boundary_logits_np = logits[0].float().cpu().numpy()
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(batch)]
        ended = [False] * batch

        for _step in range(max_new_tokens):
            scores = logits.float() - presence_penalty * presence.float()
            scores = scores / temperature
            dummy = torch.zeros((batch, 1), dtype=torch.long, device=device)
            scores = top_k_warper(dummy, scores)
            scores = top_p_warper(dummy, scores)
            probs = torch.softmax(scores, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
            for row in range(batch):
                if finished[row]:
                    continue
                token = int(next_ids[row].item())
                if token in stop_set:
                    ended[row] = True
                    finished[row] = True
                    continue
                generated[row].append(token)
                presence[row, token] = True
            if bool(finished.all()):
                break
            output = _model_forward(
                model, keep_last=1, input_ids=next_ids.unsqueeze(1), past_key_values=past, use_cache=True
            )
            past = output.past_key_values
            logits = output.logits[:, -1, :]

        for row in range(batch):
            texts.append(tokenizer.decode(generated[row], skip_special_tokens=True))
            ended_flags.append(ended[row])
        del past, output, logits, presence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        chunk_start += batch

    labels = [
        classify_continuation(text, ended_with_stop_token=flag)
        for text, flag in zip(texts, ended_flags)
    ]
    policy_lp = policy_logprobs_np(boundary_logits_np, presence_ids=presence_base)
    raw_lp = raw_logprobs_np(boundary_logits_np)
    keep_mask = topk_topp_keep_mask_np(np.asarray(policy_lp))
    n_stop = sum(label == "stop" for label in labels)
    return {
        "anchor_id": anchor["anchor_id"],
        "anchor_type": anchor["anchor_type"],
        "prompt_id": anchor["prompt_id"],
        "condition": condition_name,
        "stop_hazard": n_stop / k_resample,
        "n_stop": n_stop,
        "n_continue": sum(label == "continue" for label in labels),
        "n_other": sum(label == "other" for label in labels),
        "margin_first_policy": float(policy_lp[close_first] - policy_lp[cont_first]),
        "margin_first_raw": float(raw_lp[close_first] - raw_lp[cont_first]),
        "close_first_survives_sampler": bool(keep_mask[close_first]),
        "continuation_heads": [text[:60] for text in texts[:4]],
    }


def generate_steered(
    model,
    tokenizer,
    anchor: Mapping[str, Any],
    *,
    chat_ids: Sequence[int],
    stop_ids: Sequence[int],
    hook: SteeringHook,
    vector,
    device: str,
    n_samples: int,
    max_new_tokens: int,
    seed_tag: str,
) -> list[dict[str, Any]]:
    """Long-form steered continuations for the quality check."""
    import torch

    from tcr.boundary.constants import NOTHINK_SAMPLING
    from tcr.boundary.measure import _model_forward, _warpers

    prefix_response_ids = [int(v) for v in anchor["prefix_response_ids"]]
    prefix = list(chat_ids) + list(prefix_response_ids)
    boundary_index = len(prefix) - 1
    stop_set = {int(v) for v in stop_ids}
    if vector is not None:
        hook.configure(torch.tensor(vector, dtype=torch.float32, device=device), boundary_index)
    else:
        hook.disable()
    top_k_warper, top_p_warper = _warpers()
    presence_penalty = float(NOTHINK_SAMPLING["presence_penalty"])
    temperature = float(NOTHINK_SAMPLING["temperature"])
    vocab = model.get_output_embeddings().weight.shape[0]
    digest = hashlib.sha256(f"{anchor['anchor_id']}::{seed_tag}".encode()).hexdigest()

    input_ids = torch.tensor([prefix] * n_samples, dtype=torch.long, device=device)
    presence = torch.zeros((n_samples, vocab), dtype=torch.bool, device=device)
    response_ids = torch.tensor(sorted({int(v) for v in prefix_response_ids}), dtype=torch.long, device=device)
    presence[:, response_ids] = True
    generator = torch.Generator(device=device)
    generator.manual_seed(int(digest[:8], 16))

    output = _model_forward(model, keep_last=1, input_ids=input_ids, use_cache=True)
    past = output.past_key_values
    logits = output.logits[:, -1, :]
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(n_samples)]
    ended = [False] * n_samples
    for _step in range(max_new_tokens):
        scores = logits.float() - presence_penalty * presence.float()
        scores = scores / temperature
        dummy = torch.zeros((n_samples, 1), dtype=torch.long, device=device)
        scores = top_k_warper(dummy, scores)
        scores = top_p_warper(dummy, scores)
        probs = torch.softmax(scores, dim=-1)
        next_ids = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        for row in range(n_samples):
            if finished[row]:
                continue
            token = int(next_ids[row].item())
            if token in stop_set:
                ended[row] = True
                finished[row] = True
                continue
            generated[row].append(token)
            presence[row, token] = True
        if bool(finished.all()):
            break
        output = _model_forward(
            model, keep_last=1, input_ids=next_ids.unsqueeze(1), past_key_values=past, use_cache=True
        )
        past = output.past_key_values
        logits = output.logits[:, -1, :]
    del past, output, logits, presence
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [
        {
            "sample_index": row,
            "text": tokenizer.decode(generated[row], skip_special_tokens=True),
            "ended_with_stop_token": ended[row],
            "n_tokens": len(generated[row]),
        }
        for row in range(n_samples)
    ]
