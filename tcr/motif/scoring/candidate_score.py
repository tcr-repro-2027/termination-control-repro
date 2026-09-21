"""Identity-token score without materializing sequence-by-vocabulary logits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..constants import IDENTITY_FIELDS
from ..models.qwen_loader import base_model, causal_lm, unwrap_model


@dataclass
class CandidateScore:
    mean_log_probability: Any
    field_scores: dict[str, Any]
    target_log_probabilities: Any
    target_indices: tuple[int, ...]


def target_log_probs_from_hidden(model, hidden, target_ids, *, chunk_positions: int = 64):
    import torch

    outer = causal_lm(model)
    if hidden.ndim != 2:
        raise ValueError("hidden must be [positions, d_model]")
    if len(hidden) != len(target_ids):
        raise ValueError("hidden/target length mismatch")
    values = []
    for start in range(0, len(target_ids), chunk_positions):
        h = hidden[start : start + chunk_positions]
        target = target_ids[start : start + chunk_positions]
        logits = outer.lm_head(h).float()
        selected = logits.gather(-1, target[:, None]).squeeze(-1)
        values.append(selected - torch.logsumexp(logits, dim=-1))
        del logits
    return torch.cat(values) if values else hidden.new_empty((0,), dtype=torch.float32)


def build_sparse_forward_wrapper(causal_model, *, chunk_positions: int = 64):
    """Wrap a CausalLM so root FSDP can run a no-full-logits forward."""
    import torch

    class SparseForwardWrapper(torch.nn.Module):
        is_lgdrs_sparse_wrapper = True

        def __init__(self, model, chunk):
            super().__init__()
            self.causal_lm = model
            self.chunk_positions = int(chunk)

        def forward(self, input_ids, predictor_indices, target_indices):
            if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                raise ValueError("LG-DRS sparse wrapper requires batch size 1")
            outputs = self.causal_lm.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
            hidden_all = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            hidden = hidden_all[0].index_select(0, predictor_indices)
            targets = input_ids[0].index_select(0, target_indices)
            values = []
            for start in range(0, len(targets), self.chunk_positions):
                logits = self.causal_lm.lm_head(hidden[start : start + self.chunk_positions]).float()
                selected = logits.gather(-1, targets[start : start + self.chunk_positions, None]).squeeze(-1)
                values.append(selected - torch.logsumexp(logits, dim=-1))
                del logits
            return torch.cat(values)

    return SparseForwardWrapper(causal_model, chunk_positions)


def score_branch(
    model,
    branch: dict[str, Any],
    *,
    device=None,
    chunk_positions: int = 64,
    hook_context=None,
) -> CandidateScore:
    import torch

    ids = torch.tensor(branch["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    predictors = tuple(int(value) for value in branch["predictor_indices"])
    targets = tuple(int(value) for value in branch["target_indices"])
    if len(predictors) != len(targets) or not targets:
        raise ValueError("candidate branch has an invalid identity mask")
    predictor_tensor = torch.tensor(predictors, dtype=torch.long, device=ids.device)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=ids.device)
    context = hook_context if hook_context is not None else _NullContext()
    with context:
        unwrapped = unwrap_model(model)
        if getattr(unwrapped, "is_lgdrs_sparse_wrapper", False):
            log_probs = model(ids, predictor_tensor, target_tensor)
        else:
            outputs = base_model(model)(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            hidden_all = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            selected_hidden = hidden_all[0].index_select(0, predictor_tensor)
            target_ids = ids[0].index_select(0, target_tensor)
            log_probs = target_log_probs_from_hidden(
                model, selected_hidden, target_ids, chunk_positions=chunk_positions
            )
    position_to_local = {position: local for local, position in enumerate(targets)}
    field_scores = {}
    for field in IDENTITY_FIELDS:
        positions = tuple(int(value) for value in branch["field_token_indices"][field])
        local = torch.tensor([position_to_local[p] for p in positions], device=log_probs.device)
        field_scores[field] = log_probs.index_select(0, local).mean()
    return CandidateScore(log_probs.mean(), field_scores, log_probs, targets)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
