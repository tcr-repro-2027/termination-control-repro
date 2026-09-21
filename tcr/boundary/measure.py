"""Per-anchor GPU measurement: teacher-forced StopMargin + on-policy stop hazard.

Margins
-------
Two full forwards per anchor (prefix+close, prefix+continue).  The boundary
next-token distribution is read from position ``len(prefix)-1`` and is
asserted identical across the two forwards.  Path scores are sums of
log-probabilities along the path tokens, in BOTH spaces:

* policy: additive presence penalty on output-so-far tokens, then /T, then
  log-softmax (pre top-k/p) — numpy reference in :mod:`.policy`;
* raw: plain log-softmax.

The close path ends with one extra stop-token step scored as
``log sum_prob(stop ids)``.

Hazard
------
K short continuations resampled with EXACTLY the deployed nothink sampler
(additive presence penalty on output tokens, T=0.7, top-k=20 then top-p=0.8
via the HF warpers, min_p=0, repetition_penalty=1).  Each continuation is
classified stop / continue / other by its first non-whitespace character.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    K_RESAMPLE,
    MAX_NEW_TOKENS,
    NOTHINK_SAMPLING,
    SAMPLE_BATCH,
)
from .policy import (
    logsumexp_np,
    policy_logprobs_np,
    raw_logprobs_np,
    topk_topp_keep_mask_np,
    classify_continuation,
)


def _model_forward(model, *, keep_last: int, **kwargs):
    """Forward pass keeping only the last ``keep_last`` positions' logits.

    Materializing full-sequence logits at 32K-class prefixes costs gigabytes
    (seq x 151k vocab); ``logits_to_keep`` (transformers 5.x) /
    ``num_logits_to_keep`` (4.x) restricts the LM head to the needed tail.
    Falls back to a plain forward plus slicing when neither kwarg exists.
    """
    import torch

    with torch.no_grad():
        try:
            output = model(**kwargs, logits_to_keep=keep_last)
        except TypeError:
            try:
                output = model(**kwargs, num_logits_to_keep=keep_last)
            except TypeError:
                output = model(**kwargs)
    if output.logits.shape[1] > keep_last:
        output.logits = output.logits[:, -keep_last:, :]
    return output


def _forward_tail_logits(model, input_ids: Sequence[int], device: str, *, keep_last: int):
    """Float32 CPU numpy logits for the last ``keep_last`` positions."""
    import torch

    tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    output = _model_forward(model, keep_last=keep_last, input_ids=tensor, use_cache=False)
    return output.logits[0].float().cpu().numpy()


def _path_scores(
    step_logits: "np.ndarray",
    path_ids: Sequence[int],
    *,
    presence_base: set[int],
) -> dict[str, Any]:
    """Sum path log-probabilities in policy and raw space.

    ``step_logits[i]`` is the raw next-token logit vector BEFORE consuming
    ``path_ids[i]``; the presence set grows as path tokens are consumed.
    """
    presence = set(presence_base)
    policy_sum = 0.0
    raw_sum = 0.0
    first_policy = math.nan
    first_raw = math.nan
    for index, token_id in enumerate(path_ids):
        logits = step_logits[index]
        policy_lp = policy_logprobs_np(logits, presence_ids=presence)[int(token_id)]
        raw_lp = raw_logprobs_np(logits)[int(token_id)]
        policy_sum += float(policy_lp)
        raw_sum += float(raw_lp)
        if index == 0:
            first_policy = float(policy_lp)
            first_raw = float(raw_lp)
        presence.add(int(token_id))
    return {
        "policy_sum": policy_sum,
        "raw_sum": raw_sum,
        "first_policy": first_policy,
        "first_raw": first_raw,
        "presence_after": presence,
    }


def measure_margins(
    model,
    *,
    chat_ids: Sequence[int],
    prefix_response_ids: Sequence[int],
    close_ids: Sequence[int],
    continue_ids: Sequence[int],
    stop_ids: Sequence[int],
    device: str,
) -> dict[str, Any]:
    prefix = list(chat_ids) + list(prefix_response_ids)
    presence_base = {int(v) for v in prefix_response_ids}

    # Tail logits windows: rows cover positions [boundary .. end-of-path].
    close_steps = _forward_tail_logits(
        model, prefix + list(close_ids), device, keep_last=len(close_ids) + 1
    )
    cont_steps = _forward_tail_logits(
        model, prefix + list(continue_ids), device, keep_last=len(continue_ids) + 1
    )
    boundary_a = close_steps[0]
    boundary_b = cont_steps[0]
    drift = float(np.max(np.abs(boundary_a - boundary_b)))
    if drift > 5e-2:
        raise RuntimeError(f"boundary logits differ between path forwards: {drift}")
    cont_steps = cont_steps[: len(continue_ids)]

    close_score = _path_scores(close_steps, close_ids, presence_base=presence_base)
    cont_score = _path_scores(cont_steps, continue_ids, presence_base=presence_base)

    # Final close step: probability of ANY stop token after the close tail.
    final_logits = close_steps[len(close_ids)]
    policy_lp = policy_logprobs_np(final_logits, presence_ids=close_score["presence_after"])
    raw_lp = raw_logprobs_np(final_logits)
    stop_policy = logsumexp_np([float(policy_lp[int(v)]) for v in stop_ids])
    stop_raw = logsumexp_np([float(raw_lp[int(v)]) for v in stop_ids])
    close_policy_total = close_score["policy_sum"] + stop_policy
    close_raw_total = close_score["raw_sum"] + stop_raw

    boundary_policy_lp = policy_logprobs_np(boundary_a, presence_ids=presence_base)
    keep_mask = topk_topp_keep_mask_np(
        np.asarray(boundary_policy_lp)  # monotone transform of policy logits
    )
    close_first = int(close_ids[0])
    cont_first = int(continue_ids[0])
    return {
        "prefix_tokens_total": len(prefix),
        "margin_policy": close_policy_total - cont_score["policy_sum"],
        "margin_raw": close_raw_total - cont_score["raw_sum"],
        "margin_first_policy": close_score["first_policy"] - cont_score["first_policy"],
        "margin_first_raw": close_score["first_raw"] - cont_score["first_raw"],
        "close_path_policy_logprob": close_policy_total,
        "continue_path_policy_logprob": cont_score["policy_sum"],
        "close_path_raw_logprob": close_raw_total,
        "continue_path_raw_logprob": cont_score["raw_sum"],
        "close_first_token_policy_logprob": close_score["first_policy"],
        "continue_first_token_policy_logprob": cont_score["first_policy"],
        "close_first_survives_sampler": bool(keep_mask[close_first]),
        "continue_first_survives_sampler": bool(keep_mask[cont_first]),
        "boundary_policy_top_id": int(np.argmax(boundary_policy_lp)),
    }


def _warpers():
    from transformers import TopKLogitsWarper, TopPLogitsWarper

    return (
        TopKLogitsWarper(int(NOTHINK_SAMPLING["top_k"])),
        TopPLogitsWarper(float(NOTHINK_SAMPLING["top_p"])),
    )


def sample_stop_hazard(
    model,
    tokenizer,
    *,
    chat_ids: Sequence[int],
    prefix_response_ids: Sequence[int],
    stop_ids: Sequence[int],
    device: str,
    k_resample: int = K_RESAMPLE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sample_batch: int = SAMPLE_BATCH,
    base_seed: int = 0,
) -> dict[str, Any]:
    import torch

    top_k_warper, top_p_warper = _warpers()
    presence_penalty = float(NOTHINK_SAMPLING["presence_penalty"])
    temperature = float(NOTHINK_SAMPLING["temperature"])
    stop_set = {int(v) for v in stop_ids}
    prefix = list(chat_ids) + list(prefix_response_ids)
    vocab = model.get_output_embeddings().weight.shape[0]

    texts: list[str] = []
    ended_flags: list[bool] = []
    chunk_start = 0
    while chunk_start < k_resample:
        batch = min(sample_batch, k_resample - chunk_start)
        input_ids = torch.tensor([prefix] * batch, dtype=torch.long, device=device)
        presence = torch.zeros((batch, vocab), dtype=torch.bool, device=device)
        response_ids = torch.tensor(sorted({int(v) for v in prefix_response_ids}), dtype=torch.long, device=device)
        presence[:, response_ids] = True

        generator = torch.Generator(device=device)
        generator.manual_seed(int(base_seed) * 1000003 + chunk_start)

        output = _model_forward(model, keep_last=1, input_ids=input_ids, use_cache=True)
        past = output.past_key_values
        logits = output.logits[:, -1, :]
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        generated: list[list[int]] = [[] for _ in range(batch)]
        ended = [False] * batch

        for _step in range(max_new_tokens):
            # Deployed nothink order: additive presence penalty on output-so-far
            # tokens, then temperature, then top-k, then top-p, then sample.
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
                model,
                keep_last=1,
                input_ids=next_ids.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
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
    n_stop = sum(label == "stop" for label in labels)
    n_continue = sum(label == "continue" for label in labels)
    n_other = sum(label == "other" for label in labels)
    return {
        "k_resample": k_resample,
        "stop_hazard": n_stop / k_resample,
        "n_stop": n_stop,
        "n_continue": n_continue,
        "n_other": n_other,
        "continuation_heads": [text[:80] for text in texts],
    }


def measure_anchor(
    model,
    tokenizer,
    anchor: Mapping[str, Any],
    *,
    chat_ids: Sequence[int],
    stop_ids: Sequence[int],
    device: str,
    with_hazard: bool,
    k_resample: int = K_RESAMPLE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    sample_batch: int = SAMPLE_BATCH,
) -> dict[str, Any]:
    prefix_response_ids = [int(v) for v in anchor["prefix_response_ids"]]
    if len(prefix_response_ids) != int(anchor["prefix_token_count"]):
        raise RuntimeError(
            f"anchor {anchor['anchor_id']}: stored prefix ids "
            f"{len(prefix_response_ids)} != prefix_token_count {anchor['prefix_token_count']}"
        )
    # Stored ids are the frozen ground truth (BPE need not retokenize a cut
    # text identically); the re-encode comparison is a soft diagnostic only.
    reencoded = list(tokenizer.encode(anchor["prefix_response_text"], add_special_tokens=False))
    result: dict[str, Any] = {
        "anchor_id": anchor["anchor_id"],
        "anchor_type": anchor["anchor_type"],
        "prompt_id": anchor["prompt_id"],
        "boundary_block_1based": anchor["boundary_block_1based"],
        "progress_boundary": anchor["progress_boundary"],
        "prefix_retokenize_match": reencoded == prefix_response_ids,
    }
    result.update(
        measure_margins(
            model,
            chat_ids=chat_ids,
            prefix_response_ids=prefix_response_ids,
            close_ids=[int(v) for v in anchor["close_tail_ids"]],
            continue_ids=[int(v) for v in anchor["continue_ids"]],
            stop_ids=stop_ids,
            device=device,
        )
    )
    if with_hazard:
        import hashlib

        digest = hashlib.sha256(str(anchor["anchor_id"]).encode("utf-8")).hexdigest()
        result.update(
            sample_stop_hazard(
                model,
                tokenizer,
                chat_ids=chat_ids,
                prefix_response_ids=prefix_response_ids,
                stop_ids=stop_ids,
                device=device,
                k_resample=k_resample,
                max_new_tokens=max_new_tokens,
                sample_batch=sample_batch,
                base_seed=int(digest[:8], 16),
            )
        )
    return result
