"""Boundary/per-token residual-stream capture with forward hooks (torch).

Hook convention (validated in 02): SAE layer n (0-based) reads the OUTPUT of
``model.model.layers[n]`` (resid_post).  ``output_hidden_states`` is never
used — the final RMSNorm would contaminate the last layer's stream.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class ResidualCapture:
    """Forward hooks over all decoder blocks capturing selected positions."""

    def __init__(self, model, layers: Sequence[int]):
        self.layers = list(layers)
        self._storage: dict[int, Any] = {}
        self._handles = []
        self._positions: slice | None = None
        blocks = model.model.layers
        for layer in self.layers:
            self._handles.append(
                blocks[layer].register_forward_hook(self._make_hook(layer))
            )

    def _make_hook(self, layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            positions = self._positions if self._positions is not None else slice(-1, None)
            self._storage[layer] = hidden[:, positions, :].detach().to("cpu")
            return output

        return hook

    def set_positions(self, positions: slice | None) -> None:
        self._positions = positions

    def take(self) -> dict[int, Any]:
        grabbed, self._storage = self._storage, {}
        return grabbed

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _forward_min_logits(model, input_ids, **kwargs):
    import torch

    with torch.no_grad():
        try:
            return model(input_ids=input_ids, logits_to_keep=1, **kwargs)
        except TypeError:
            try:
                return model(input_ids=input_ids, num_logits_to_keep=1, **kwargs)
            except TypeError:
                return model(input_ids=input_ids, **kwargs)


def capture_boundary(
    model,
    *,
    chat_ids: Sequence[int],
    prefix_response_ids: Sequence[int],
    layers: Sequence[int],
    device: str,
) -> np.ndarray:
    """(n_layers, d_model) float16 residual state at the boundary position."""
    import torch

    capture = ResidualCapture(model, layers)
    capture.set_positions(slice(-1, None))
    try:
        ids = list(chat_ids) + list(prefix_response_ids)
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        _forward_min_logits(model, tensor, use_cache=False)
        grabbed = capture.take()
    finally:
        capture.remove()
    rows = [grabbed[layer][0, 0, :].float().numpy() for layer in layers]
    return np.stack(rows).astype(np.float16)


def capture_layer_tokens(
    model,
    *,
    chat_ids: Sequence[int],
    prefix_response_ids: Sequence[int],
    layer: int,
    device: str,
) -> np.ndarray:
    """(n_response_tokens, d_model) float16 stream at one layer (response part)."""
    import torch

    capture = ResidualCapture(model, [layer])
    capture.set_positions(slice(len(chat_ids), None))
    try:
        ids = list(chat_ids) + list(prefix_response_ids)
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        _forward_min_logits(model, tensor, use_cache=False)
        grabbed = capture.take()
    finally:
        capture.remove()
    return grabbed[layer][0].float().numpy().astype(np.float16)
