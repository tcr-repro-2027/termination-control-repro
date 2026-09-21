from __future__ import annotations

import math

import numpy as np

from tcr.boundary.measure import _path_scores
from tcr.boundary.policy import policy_logprobs_np, raw_logprobs_np


def test_path_scores_presence_grows_along_path():
    rng = np.random.default_rng(5)
    vocab = 6
    step_logits = rng.normal(size=(2, vocab))
    path = [3, 3]  # token 3 twice: second step must see it in the presence set
    presence_base = {1}

    result = _path_scores(step_logits, path, presence_base=presence_base)

    lp0 = policy_logprobs_np(step_logits[0], presence_ids={1})[3]
    lp1 = policy_logprobs_np(step_logits[1], presence_ids={1, 3})[3]
    raw0 = raw_logprobs_np(step_logits[0])[3]
    raw1 = raw_logprobs_np(step_logits[1])[3]

    assert math.isclose(result["policy_sum"], float(lp0 + lp1), abs_tol=1e-12)
    assert math.isclose(result["raw_sum"], float(raw0 + raw1), abs_tol=1e-12)
    assert math.isclose(result["first_policy"], float(lp0), abs_tol=1e-12)
    assert result["presence_after"] == {1, 3}
    # base set must not be mutated
    assert presence_base == {1}


def test_path_scores_policy_penalizes_seen_token():
    # Identical logits; token 2 is in the base presence set, token 0 is not.
    logits = np.zeros((1, 4))
    seen = _path_scores(logits, [2], presence_base={2})
    fresh = _path_scores(logits, [0], presence_base={2})
    assert seen["policy_sum"] < fresh["policy_sum"]
    assert math.isclose(seen["raw_sum"], fresh["raw_sum"], abs_tol=1e-12)
