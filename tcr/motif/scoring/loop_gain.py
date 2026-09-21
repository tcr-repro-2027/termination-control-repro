"""Frozen repeat-vs-recover Loop Gain and its exact decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LoopGainValues:
    margin_c1: Any
    margin_c2: Any
    loop_gain: Any
    repeat_attraction: Any
    recovery_deficit: Any


def compute_loop_gain(scores: Mapping[str, Any]) -> LoopGainValues:
    required = {"R_C1", "N_C1", "R_C2", "N_C2"}
    missing = sorted(required - set(scores))
    if missing:
        raise KeyError(f"Loop Gain scores missing {missing}")
    margin_c1 = scores["R_C1"] - scores["N_C1"]
    margin_c2 = scores["R_C2"] - scores["N_C2"]
    attraction = scores["R_C2"] - scores["R_C1"]
    deficit = scores["N_C1"] - scores["N_C2"]
    gain = margin_c2 - margin_c1
    return LoopGainValues(margin_c1, margin_c2, gain, attraction, deficit)


def mediated_loop_gain(raw_gain, patched_gain):
    return raw_gain - patched_gain


def assert_decomposition(values: LoopGainValues, *, atol: float = 1e-6) -> None:
    difference = values.loop_gain - (values.repeat_attraction + values.recovery_deficit)
    try:
        magnitude = float(abs(difference.detach().cpu()))
    except AttributeError:
        magnitude = abs(float(difference))
    if magnitude > atol:
        raise AssertionError(f"Loop Gain decomposition failed: error={magnitude}")


def feedback_curve(scores: Mapping[str, Any]) -> dict[str, Any]:
    required = {"R_C1", "N_C1", "R_C2", "N_C2", "R_C3", "N_C3"}
    missing = sorted(required - set(scores))
    if missing:
        raise KeyError(f"feedback curve scores missing {missing}")
    margins = {
        "M_C1": scores["R_C1"] - scores["N_C1"],
        "M_C2": scores["R_C2"] - scores["N_C2"],
        "M_C3": scores["R_C3"] - scores["N_C3"],
    }
    margins["gain_first"] = margins["M_C2"] - margins["M_C1"]
    margins["gain_second"] = margins["M_C3"] - margins["M_C2"]
    return margins
