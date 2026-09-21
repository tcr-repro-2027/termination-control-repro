from __future__ import annotations

import math

import numpy as np
import pytest

from tcr.boundary.constants import NOTHINK_SAMPLING
from tcr.boundary.policy import (
    assert_frozen_sampling,
    classify_continuation,
    log_softmax_np,
    logsumexp_np,
    policy_logits_np,
    policy_logprobs_np,
    raw_logprobs_np,
    topk_topp_keep_mask_np,
)


def test_assert_frozen_sampling():
    assert_frozen_sampling(NOTHINK_SAMPLING)
    broken = dict(NOTHINK_SAMPLING)
    broken["temperature"] = 0.6
    with pytest.raises(ValueError, match="temperature"):
        assert_frozen_sampling(broken)
    broken = dict(NOTHINK_SAMPLING)
    broken["presence_penalty"] = 0.0
    with pytest.raises(ValueError, match="presence_penalty"):
        assert_frozen_sampling(broken)


def test_policy_logits_presence_and_temperature():
    raw = np.array([2.0, 1.0, 0.0, -1.0])
    out = policy_logits_np(raw, presence_ids=[0, 2, 2])
    # presence -1.5 on ids {0,2}, then /0.7
    expected = np.array([(2.0 - 1.5), 1.0, (0.0 - 1.5), -1.0]) / 0.7
    assert np.allclose(out, expected)
    # raw input must not be mutated
    assert np.allclose(raw, [2.0, 1.0, 0.0, -1.0])


def test_log_softmax_and_logprobs():
    raw = np.array([1.0, 2.0, 3.0])
    lp = log_softmax_np(raw)
    assert math.isclose(float(np.exp(lp).sum()), 1.0, abs_tol=1e-12)
    assert float(np.argmax(lp)) == 2
    # policy space reorders when presence penalizes the raw argmax
    policy = policy_logprobs_np(np.array([3.0, 2.9, 0.0]), presence_ids=[0])
    assert int(np.argmax(policy)) == 1
    raw_lp = raw_logprobs_np(np.array([3.0, 2.9, 0.0]))
    assert int(np.argmax(raw_lp)) == 0


def test_topk_topp_mask_topk_cap():
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    mask = topk_topp_keep_mask_np(logits, top_k=2, top_p=1.0)
    assert mask.tolist() == [True, True, False, False, False]


def test_topk_topp_mask_includes_crossing_token():
    # probs ~ [0.6, 0.3, 0.1]: cumulative before = [0, .6, .9]
    # top_p=0.8 keeps ids 0 and 1 (1 crosses the boundary), drops 2.
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    mask = topk_topp_keep_mask_np(logits, top_k=0, top_p=0.8)
    assert mask.tolist() == [True, True, False]
    # top_p=0.5: id 0 alone crosses.
    mask = topk_topp_keep_mask_np(logits, top_k=0, top_p=0.5)
    assert mask.tolist() == [True, False, False]


def test_topk_topp_never_empty():
    logits = np.array([0.0, -50.0, -50.0])
    mask = topk_topp_keep_mask_np(logits, top_k=1, top_p=0.0001)
    assert mask.tolist() == [True, False, False]


def test_logsumexp():
    assert math.isclose(logsumexp_np([math.log(0.25), math.log(0.75)]), 0.0, abs_tol=1e-12)


def test_classify_continuation():
    assert classify_continuation("]\n", ended_with_stop_token=True) == "stop"
    assert classify_continuation("\n]", ended_with_stop_token=False) == "stop"
    assert classify_continuation(', {"source": "x"}', ended_with_stop_token=False) == "continue"
    assert classify_continuation('\n{"source"', ended_with_stop_token=False) == "continue"
    assert classify_continuation("", ended_with_stop_token=True) == "stop"
    assert classify_continuation("", ended_with_stop_token=False) == "other"
    assert classify_continuation("好的，我继续", ended_with_stop_token=False) == "other"
    # Straddling-boundary re-emissions: the block tail precedes the decision.
    assert classify_continuation('"}]', ended_with_stop_token=True) == "stop"
    assert classify_continuation('"}]', ended_with_stop_token=False) == "stop"
    assert classify_continuation('"},\n  {"source', ended_with_stop_token=False) == "continue"
    assert classify_continuation('。"}\n]', ended_with_stop_token=True) == "stop"
    # Stop token right after finishing the block, without the list close.
    assert classify_continuation('"}', ended_with_stop_token=True) == "stop"
    assert classify_continuation('"}', ended_with_stop_token=False) == "other"


def test_parity_with_hf_warpers():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import TopKLogitsWarper, TopPLogitsWarper

    rng = np.random.default_rng(0)
    for _ in range(20):
        logits = rng.normal(size=200).astype(np.float64) * 3
        mask_np = topk_topp_keep_mask_np(logits)
        scores = torch.tensor(logits).unsqueeze(0)
        dummy = torch.zeros((1, 1), dtype=torch.long)
        scores = TopKLogitsWarper(int(NOTHINK_SAMPLING["top_k"]))(dummy, scores)
        scores = TopPLogitsWarper(float(NOTHINK_SAMPLING["top_p"]))(dummy, scores)
        mask_hf = torch.isfinite(scores[0]).numpy()
        assert mask_np.tolist() == mask_hf.tolist()
