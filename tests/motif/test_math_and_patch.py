from __future__ import annotations

import numpy as np
import pytest

from tcr.motif.scoring.loop_gain import assert_decomposition, compute_loop_gain, mediated_loop_gain


def test_loop_gain_and_exact_decomposition():
    values = compute_loop_gain({"R_C1": -2.0, "N_C1": -1.0, "R_C2": -0.5, "N_C2": -1.5})
    assert values.margin_c1 == -1.0
    assert values.margin_c2 == 1.0
    assert values.loop_gain == 2.0
    assert values.repeat_attraction == 1.5
    assert values.recovery_deficit == 0.5
    assert_decomposition(values)
    assert mediated_loop_gain(values.loop_gain, 0.25) == 1.75
