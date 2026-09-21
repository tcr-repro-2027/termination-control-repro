# coding: utf-8
"""Deployment-exact resampling at a block boundary, with or without a pulse.

One sampler serves R1 and R2 so an intervention is never compared against a
baseline that was measured a slightly different way.  The order is the deployed
nothink order and nothing else: additive presence penalty on the tokens already
in the OUTPUT, then temperature, then top-k, then top-p, then sample.  The
per-draw seed is derived from the anchor id, so re-running reproduces the same
sixteen continuations.

Two optional modifications, both applied at the FIRST step only:

``hook`` + ``vector``   S2's residual-stream hook, configured before the prefill
                        and disabled immediately after it.  That is the "single
                        pulse" the plan specifies: the injected vector enters
                        the one forward that predicts the close/continue token,
                        and its consequences then propagate through the cache on
                        their own rather than being re-applied at every token.
``close_bias``          a fixed addition to the close token's processed logit.
                        The crude "just stop more" control an internal direction
                        has to beat on something other than the stop rate.

Continuations are classified by their first decisive character.  Anything that
is neither a legal close nor a legal continue keeps its own name: a sequence
that ended without closing the list is a malformed termination, not a stop.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from tcr.boundary.constants import NOTHINK_SAMPLING       # noqa: E402
from tcr.boundary.measure import (_forward_tail_logits,   # noqa: E402
                                   _model_forward, _warpers)
from tcr.boundary.policy import (classify_continuation,   # noqa: E402
                                  logsumexp_np, policy_logprobs_np,
                                  raw_logprobs_np, topk_topp_keep_mask_np)

LABELS = ("close", "continue", "eos_without_close", "unresolved")


def anchor_seed(anchor_id: str, salt: str = "") -> int:
    digest = hashlib.sha256(f"{anchor_id}::{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def refine_label(text: str, *, ended_with_stop_token: bool) -> str:
    """S1's three-way classification with ``other`` split into its two causes."""
    label = classify_continuation(text, ended_with_stop_token=ended_with_stop_token)
    if label == "stop":
        return "close"
    if label == "continue":
        return "continue"
    return "eos_without_close" if ended_with_stop_token else "unresolved"


def sampler_probability(policy_logprobs: np.ndarray, keep: np.ndarray,
                        token_id: int) -> float:
    """Probability of a token AFTER top-k/top-p truncation and renormalising."""
    if not bool(keep[int(token_id)]):
        return 0.0
    kept = np.asarray(policy_logprobs)[keep]
    peak = float(np.max(kept))
    total = float(np.sum(np.exp(kept - peak)))
    return float(np.exp(float(policy_logprobs[int(token_id)]) - peak) / total)


def boundary_metrics(logits: np.ndarray, *, presence_ids: Sequence[int],
                     close_first: int, continue_first: int,
                     close_bias: float = 0.0) -> dict[str, Any]:
    """Score-space and reachability readouts from the boundary step's logits.

    ``close_bias`` is folded in here as well, so a biased condition is scored
    on the distribution the sampler actually drew from rather than on the
    untouched one.
    """
    policy_lp = policy_logprobs_np(logits, presence_ids=presence_ids)
    raw_lp = raw_logprobs_np(logits)
    if close_bias:
        adjusted = np.asarray(policy_lp, dtype=np.float64).copy()
        adjusted[int(close_first)] += float(close_bias)
        adjusted = adjusted - float(np.max(adjusted))
        adjusted = adjusted - float(np.log(np.sum(np.exp(adjusted))))
        policy_lp = adjusted
    keep = topk_topp_keep_mask_np(np.asarray(policy_lp))
    return {
        "margin_first_policy": float(policy_lp[close_first] - policy_lp[continue_first]),
        "margin_first_raw": float(raw_lp[close_first] - raw_lp[continue_first]),
        "close_prob_pre_filter": float(np.exp(policy_lp[close_first])),
        "continue_prob_pre_filter": float(np.exp(policy_lp[continue_first])),
        "close_sampler_prob": sampler_probability(policy_lp, keep, close_first),
        "continue_sampler_prob": sampler_probability(policy_lp, keep, continue_first),
        "close_reachable": bool(keep[int(close_first)]),
        "continue_reachable": bool(keep[int(continue_first)]),
        "n_tokens_kept_by_sampler": int(keep.sum()),
    }


def close_branch_readout(model, *, chat_ids: Sequence[int],
                         prefix_response_ids: Sequence[int],
                         close_ids: Sequence[int], stop_ids: Sequence[int],
                         device: str) -> dict[str, Any]:
    """One teacher-forced forward over prefix + close branch.

    It answers two different questions at once, which is why it is one forward:
    the boundary step's logits (what the model would do when free to choose) and
    the step after the whole close branch is written (given the ending is
    already being produced, how likely is EOS next).  The second separates
    CHOOSING to close from being ABLE to finish -- a model that rarely picks the
    close token but executes a supplied close correctly has a different problem
    from one that cannot produce the ending at all.
    """
    prefix = list(chat_ids) + list(prefix_response_ids)
    steps = _forward_tail_logits(model, prefix + list(close_ids), device,
                                 keep_last=len(close_ids) + 1)
    presence = {int(v) for v in prefix_response_ids}
    presence.update(int(v) for v in close_ids)
    final = policy_logprobs_np(steps[len(close_ids)], presence_ids=presence)
    eos_logprob = logsumexp_np([float(final[int(v)]) for v in stop_ids])
    return {
        "_boundary_logits": steps[0],
        "eos_after_forced_close_prob": float(np.exp(eos_logprob)),
        "eos_after_forced_close_logprob": float(eos_logprob),
    }


def _prepare_hook(hook, vector, boundary_index: int, device: str):
    import torch

    if hook is None:
        return
    if vector is None:
        hook.disable()
        return
    hook.configure(torch.tensor(np.asarray(vector, dtype=np.float32),
                                dtype=torch.float32, device=device),
                   boundary_index)


def sample_next_event(model, tokenizer, *, chat_ids: Sequence[int],
                      prefix_response_ids: Sequence[int],
                      stop_ids: Sequence[int], device: str,
                      k_resample: int = 16, max_new_tokens: int = 32,
                      sample_batch: int = 8, base_seed: int = 0,
                      hook=None, vector=None, close_bias: float = 0.0,
                      close_token_id: int | None = None,
                      keep_heads: int = 4) -> dict[str, Any]:
    """K short continuations; returns the four-way split and the boundary logits."""
    import torch

    top_k_warper, top_p_warper = _warpers()
    presence_penalty = float(NOTHINK_SAMPLING["presence_penalty"])
    temperature = float(NOTHINK_SAMPLING["temperature"])
    stop_set = {int(v) for v in stop_ids}
    prefix = list(chat_ids) + list(prefix_response_ids)
    boundary_index = len(prefix) - 1
    vocab = model.get_output_embeddings().weight.shape[0]

    texts: list[str] = []
    ended_flags: list[bool] = []
    boundary_logits: np.ndarray | None = None
    chunk_start = 0
    while chunk_start < k_resample:
        batch = min(sample_batch, k_resample - chunk_start)
        input_ids = torch.tensor([prefix] * batch, dtype=torch.long, device=device)
        presence = torch.zeros((batch, vocab), dtype=torch.bool, device=device)
        response_ids = torch.tensor(sorted({int(v) for v in prefix_response_ids}),
                                    dtype=torch.long, device=device)
        presence[:, response_ids] = True
        generator = torch.Generator(device=device)
        generator.manual_seed(int(base_seed) * 1000003 + chunk_start)

        _prepare_hook(hook, vector, boundary_index, device)
        output = _model_forward(model, keep_last=1, input_ids=input_ids, use_cache=True)
        if hook is not None:
            hook.disable()          # single pulse: the boundary forward only
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        if boundary_logits is None:
            boundary_logits = logits[0].float().cpu().numpy()
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(batch)]
        ended = [False] * batch

        for step in range(max_new_tokens):
            scores = logits.float() - presence_penalty * presence.float()
            scores = scores / temperature
            if step == 0 and close_bias and close_token_id is not None:
                scores[:, int(close_token_id)] += float(close_bias)
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
            output = _model_forward(model, keep_last=1, input_ids=next_ids.unsqueeze(1),
                                    past_key_values=past, use_cache=True)
            past = output.past_key_values
            logits = output.logits[:, -1, :]

        for row in range(batch):
            texts.append(tokenizer.decode(generated[row], skip_special_tokens=True))
            ended_flags.append(ended[row])
        del past, output, logits, presence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        chunk_start += batch

    labels = [refine_label(text, ended_with_stop_token=flag)
              for text, flag in zip(texts, ended_flags)]
    counts = Counter(labels)
    result: dict[str, Any] = {"k_resample": k_resample}
    for label in LABELS:
        result[f"n_{label}"] = counts.get(label, 0)
        result[f"{label}_rate"] = counts.get(label, 0) / k_resample
    result["next_event_close_rate"] = result["close_rate"]
    result["other_rate"] = result["eos_without_close_rate"] + result["unresolved_rate"]
    result["continuation_heads"] = [text[:60] for text in texts[:keep_heads]]
    result["_boundary_logits"] = boundary_logits
    return result


def generate_long(model, tokenizer, *, chat_ids: Sequence[int],
                  prefix_response_ids: Sequence[int], stop_ids: Sequence[int],
                  device: str, n_samples: int, max_new_tokens: int,
                  base_seed: int, hook=None, vector=None,
                  close_bias: float = 0.0, close_token_id: int | None = None
                  ) -> list[dict[str, Any]]:
    """Full continuations to EOS or the remaining context, one pulse at the start.

    ``hit_max`` here means the continuation used its whole remaining budget
    without emitting a stop token -- the same terminal state the free-generation
    evaluation records, so the two are comparable.
    """
    import torch

    top_k_warper, top_p_warper = _warpers()
    presence_penalty = float(NOTHINK_SAMPLING["presence_penalty"])
    temperature = float(NOTHINK_SAMPLING["temperature"])
    stop_set = {int(v) for v in stop_ids}
    prefix = list(chat_ids) + list(prefix_response_ids)
    boundary_index = len(prefix) - 1
    vocab = model.get_output_embeddings().weight.shape[0]

    input_ids = torch.tensor([prefix] * n_samples, dtype=torch.long, device=device)
    presence = torch.zeros((n_samples, vocab), dtype=torch.bool, device=device)
    response_ids = torch.tensor(sorted({int(v) for v in prefix_response_ids}),
                                dtype=torch.long, device=device)
    presence[:, response_ids] = True
    generator = torch.Generator(device=device)
    generator.manual_seed(int(base_seed))

    _prepare_hook(hook, vector, boundary_index, device)
    output = _model_forward(model, keep_last=1, input_ids=input_ids, use_cache=True)
    if hook is not None:
        hook.disable()
    past = output.past_key_values
    logits = output.logits[:, -1, :]
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in range(n_samples)]
    ended = [False] * n_samples

    for step in range(max_new_tokens):
        scores = logits.float() - presence_penalty * presence.float()
        scores = scores / temperature
        if step == 0 and close_bias and close_token_id is not None:
            scores[:, int(close_token_id)] += float(close_bias)
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
        output = _model_forward(model, keep_last=1, input_ids=next_ids.unsqueeze(1),
                                past_key_values=past, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1, :]

    del past, output, logits, presence
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [
        {
            "sample_index": row,
            "text": tokenizer.decode(generated[row], skip_special_tokens=True),
            "ended_with_stop_token": bool(ended[row]),
            "n_tokens": len(generated[row]),
            "hit_max": not ended[row],
        }
        for row in range(n_samples)
    ]
