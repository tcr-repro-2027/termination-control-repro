# coding=utf-8
"""The decision point and the sampler pipeline.

Both are places where a wrong answer looks completely normal: a margin measured
one token too early is still a number, and a mis-ordered sampler still returns a
probability.  So both are pinned against hand-computed cases here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tcr.support_probe import protocol
from tcr.support_probe.boundary import (
    DecisionPointFinder, common_prefix_length, find_decision_point,
    sampler_readouts, _top_k_top_p_mask,
)


class MergingTokenizer:
    """A BPE-like fake that merges `}` with whatever follows it.

    This is the case the whole `find_decision_point` design exists for: if
    `},` and `}]` are single tokens, cutting the context after `}` would put
    the model in a state training never produced."""

    VOCAB = {"}": 10, "},": 11, "}]": 12, "]": 13, ",": 14, " {": 15,
             "PROMPT": 20, "A": 21, "B": 22, '{"': 16, "[": 17, "[]": 18,
             '[{"': 19}

    def __call__(self, text: str):
        ids, index = [], 0
        while index < len(text):
            for size in (3, 2, 1):
                piece = text[index:index + size]
                if piece in self.VOCAB:
                    ids.append(self.VOCAB[piece])
                    index += size
                    break
            else:
                ids.append(100 + ord(text[index]) % 50)
                index += 1
        return ids


@pytest.fixture
def encode():
    return MergingTokenizer()


def test_common_prefix_length():
    assert common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_length([1, 2], [1, 2]) == 2
    assert common_prefix_length([], [1]) == 0


def test_decision_point_survives_a_merged_boundary(encode):
    """`}` merges with the next character, so the deciding token is `}]` vs
    `},` -- not `]` vs `,`.  The comparison must still be well defined."""
    point = find_decision_point(encode, "PROMPTA}", "]", ", {")

    assert point.close_token == encode.VOCAB["}]"]
    assert point.continue_token == encode.VOCAB["},"]
    # the shared context stops BEFORE the merged token, so the forward pass
    # runs on a real tokenisation of a real string
    assert point.context_ids == encode("PROMPTA")
    assert point.position == len(point.context_ids)


def test_decision_point_for_the_empty_list(encode):
    point = find_decision_point(encode, "PROMPT[", "]", '{"')
    assert point.close_token == encode.VOCAB["[]"]
    assert point.continue_token == encode.VOCAB['[{"']


def test_generation_prompt_defines_the_presence_set(encode):
    point = find_decision_point(encode, "PROMPTA}", "]", ", {",
                                generation_prompt="PROMPT")
    assert point.generated_start == len(encode("PROMPT"))
    # only the assistant's own tokens are penalised, never the prompt's
    assert point.presence_ids == encode("A")


def test_a_generation_prompt_that_is_not_a_token_prefix_is_rejected(encode):
    """If the chat template does not tokenise as a prefix, "generated tokens"
    is undefined and the presence penalty would be applied to the wrong set."""
    with pytest.raises(ValueError, match="token prefix"):
        find_decision_point(encode, "PROMPTA}", "]", ", {",
                            generation_prompt="XPROMPT")


def test_continuations_must_diverge(encode):
    with pytest.raises(ValueError, match="diverge"):
        find_decision_point(encode, "PROMPT", "A", "A")


# --------------------------------------------------------------- the sampler

def test_raw_margin_is_the_plain_logit_difference():
    logits = np.zeros(8)
    logits[1] = 2.0          # close
    logits[2] = 0.5          # continue
    out = sampler_readouts(logits, close_token=1, continue_token=2,
                           presence_ids=[], temperature=1.0,
                           presence_penalty=0.0, top_k=0, top_p=1.0)
    assert out["sm_raw"] == pytest.approx(1.5)
    assert out["sm_primary"] == pytest.approx(1.5)


def test_presence_penalty_hits_only_generated_tokens():
    """At a block boundary `,` has occurred many times and `]` has not, so the
    penalty shifts the margin by a full penalty in the closing direction.  That
    is not a rounding detail -- it is most of the absolute margin."""
    logits = np.zeros(8)
    logits[1], logits[2] = 2.0, 0.5
    out = sampler_readouts(logits, close_token=1, continue_token=2,
                           presence_ids=[2], temperature=1.0,
                           presence_penalty=1.5, top_k=0, top_p=1.0)
    assert out["sm_raw"] == pytest.approx(1.5)
    assert out["sm_primary"] == pytest.approx(1.5 + 1.5)


def test_temperature_scales_the_primary_margin():
    logits = np.zeros(8)
    logits[1], logits[2] = 2.0, 0.5
    out = sampler_readouts(logits, close_token=1, continue_token=2,
                           presence_ids=[], temperature=0.7,
                           presence_penalty=0.0, top_k=0, top_p=1.0)
    assert out["sm_primary"] == pytest.approx(1.5 / 0.7)


def test_rank_is_taken_after_the_penalty():
    """Rank is what the sampler orders on, so the penalty must be applied
    before it -- otherwise a heavily penalised token looks competitive."""
    logits = np.array([0.0, 5.0, 4.0, 3.0])
    unpenalised = sampler_readouts(logits, close_token=1, continue_token=2,
                                   presence_ids=[], temperature=1.0,
                                   presence_penalty=1.5, top_k=0, top_p=1.0)
    assert unpenalised["close_rank"] == 1

    penalised = sampler_readouts(logits, close_token=1, continue_token=2,
                                 presence_ids=[1], temperature=1.0,
                                 presence_penalty=1.5, top_k=0, top_p=1.0)
    assert penalised["close_rank"] == 2      # 5.0 - 1.5 now sits below 4.0


def test_top_k_filters_by_rank():
    logits = np.arange(10, dtype=float)       # token 9 best, token 0 worst
    out = sampler_readouts(logits, close_token=0, continue_token=9,
                           presence_ids=[], temperature=1.0,
                           presence_penalty=0.0, top_k=3, top_p=1.0)
    assert out["close_in_top_k"] is False
    assert out["close_sampler_prob"] == pytest.approx(0.0)
    assert out["continue_sampler_prob"] > 0
    assert out["n_kept_by_sampler"] == 3


def test_top_p_keeps_the_smallest_set_reaching_p():
    """vLLM masks a token when the mass strictly before it already reached p,
    so the token that crosses the threshold is kept."""
    probs = np.array([0.6, 0.3, 0.07, 0.03])
    logits = np.log(probs)
    out = sampler_readouts(logits, close_token=0, continue_token=2,
                           presence_ids=[], temperature=1.0,
                           presence_penalty=0.0, top_k=0, top_p=0.8)
    # 0.6 then 0.9 crosses 0.8 -> the first two survive, the rest do not
    assert out["n_kept_by_sampler"] == 2
    assert out["close_survives_top_p"] is True
    assert out["continue_sampler_prob"] == pytest.approx(0.0)
    assert out["close_sampler_prob"] == pytest.approx(0.6 / 0.9)


def test_logprobs_are_a_proper_distribution():
    logits = np.array([0.0, 1.0, 2.0])
    out = sampler_readouts(logits, close_token=2, continue_token=0,
                           presence_ids=[], temperature=1.0,
                           presence_penalty=0.0, top_k=0, top_p=1.0)
    total = sum(math.exp(v - max(logits)) for v in logits)
    assert out["close_logprob"] == pytest.approx(2.0 - max(logits) - math.log(total))


def test_protocol_defaults_are_the_e_natural_sampler():
    """A margin measured under a different sampler is not the margin that
    governed the generations E1 scored."""
    assert protocol.TEMPERATURE == 0.7
    assert protocol.TOP_P == 0.8 and protocol.TOP_K == 20
    assert protocol.PRESENCE_PENALTY == 1.5
    assert protocol.SAMPLER_ORDER == ("presence", "temperature", "top_k", "top_p")


# ------------------------------------------------- sampler ORDER, not just ops

def _logits(probs):
    """Logits whose plain softmax is exactly `probs`."""
    return np.log(np.asarray(probs, dtype=np.float64))


def test_top_p_runs_on_the_top_k_renormalised_distribution():
    """The order is load-bearing, and this is a case where it changes the set.

    probs = [.5, .3, .1, .06, .04], top_k=4, top_p=0.8.

    * top-p on the FULL vocabulary keeps ranks 0..2 (mass before rank 2 is
      0.8), intersecting with top-k leaves THREE tokens;
    * vLLM drops rank 4 first, renormalises the remaining 0.96 mass, and then
      needs only ranks 0..1 to hold 0.8 of it -- TWO tokens.

    Reporting three would overstate what the sampler can draw and could mark a
    close token as surviving top-p when the real sampler had dropped it.
    """
    logits = _logits([0.5, 0.3, 0.1, 0.06, 0.04])
    mask = _top_k_top_p_mask(logits, top_k=4, top_p=0.8)
    assert mask.sum() == 2
    assert list(np.flatnonzero(mask)) == [0, 1]


def test_top_k_alone_is_a_value_threshold_so_ties_all_survive():
    """vLLM compares against the k-th largest VALUE, so a tie at the boundary
    keeps every tied token rather than cutting arbitrarily at k."""
    logits = _logits([0.4, 0.2, 0.2, 0.2])
    mask = _top_k_top_p_mask(logits, top_k=2, top_p=1.0)
    assert mask.sum() == 4                       # three-way tie at rank 1


def test_the_argmax_always_survives_top_p():
    """A peaked distribution must not produce an empty candidate set."""
    logits = _logits([0.999, 0.0005, 0.0005])
    mask = _top_k_top_p_mask(logits, top_k=20, top_p=0.1)
    assert mask.sum() >= 1
    assert mask[0]


def test_disabled_filters_keep_everything():
    logits = _logits([0.4, 0.3, 0.2, 0.1])
    assert _top_k_top_p_mask(logits, top_k=0, top_p=1.0).all()
    assert _top_k_top_p_mask(logits, top_k=99, top_p=1.0).all()


def test_readouts_report_the_filtered_set_the_sampler_would_use():
    logits = _logits([0.5, 0.3, 0.1, 0.06, 0.04])
    out = sampler_readouts(logits, close_token=0, continue_token=1,
                           presence_ids=[], temperature=1.0, top_p=0.8,
                           top_k=4, presence_penalty=0.0)
    assert out["n_kept_by_sampler"] == 2
    assert out["close_survives_top_p"] is True
    # probabilities are renormalised over the survivors, so they sum to 1
    assert out["close_sampler_prob"] + out["continue_sampler_prob"] == pytest.approx(1.0)


# ----------------------------------------------- what the fast path verifies

class CountingEncoder(MergingTokenizer):
    """Records the decode side so a finder can be driven without a tokenizer."""

    def decode(self, ids):
        reverse = {value: key for key, value in self.VOCAB.items()}
        return "".join(reverse.get(i, chr((i - 100) % 50)) for i in ids)


def test_the_verification_budget_counts_prefixes_not_cells():
    """An anchor has eight variants sharing ONE prefix, so a per-cell budget of
    eight verified a single prefix and trusted every later one -- and it is
    prefix length and its final tokens, not the prompt, that decide whether the
    tail window is wide enough."""
    tok = CountingEncoder()
    finder = DecisionPointFinder(tok, tok.decode, verify_first=2)

    for variant in range(8):                     # one anchor, eight prompts
        finder.find(prompt_text=f"PROMPT{variant}A}}"[:-2] + "",
                    prefix_text="A}", close_continuation="]",
                    continue_continuation=", {")
    assert finder.verified == 1                  # one DISTINCT prefix so far

    finder.find(prompt_text="PROMPT", prefix_text="B}",
                close_continuation="]", continue_continuation=", {")
    assert finder.verified == 2                  # a second prefix, also checked

    finder.find(prompt_text="PROMPT", prefix_text="A}B}",
                close_continuation="]", continue_continuation=", {")
    assert finder.verified == 2                  # budget spent
    assert not finder.fell_back


def test_the_manifest_reports_the_budget_it_actually_used():
    tok = CountingEncoder()
    finder = DecisionPointFinder(tok, tok.decode, verify_first=3)
    finder.find(prompt_text="PROMPT", prefix_text="A}",
                close_continuation="]", continue_continuation=", {")
    manifest = finder.manifest()
    assert manifest["decision_point_verified_prefixes"] == 1
    assert manifest["decision_point_verify_budget"] == 3
    assert manifest["decision_point_fell_back"] is False


# ------------------------- the two presence states the two stages sample in

def test_the_scorer_reports_both_presence_states():
    """`score` treats the assistant prefix as GENERATED (what E1's generation
    does, so the presence penalty applies to it).  The hazard cannot: vLLM is
    handed the prefix as prompt, and prompt tokens do not reach the presence
    penalty.  Both states are therefore reported from the same logits, so the
    hazard can be read against the one it was actually sampled in."""
    import numpy as np
    from tcr.support_probe.boundary import sampler_readouts
    from tcr.support_probe import protocol
    logits = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
    # continue token (1) occurs in the prefix, close token (0) does not --
    # the shape measured on 254 of the 384 real anchors
    out = sampler_readouts(logits, close_token=0, continue_token=1,
                           presence_ids=[1, 1, 2])
    assert out["continue_presence_penalised"] is True
    assert out["close_presence_penalised"] is False
    # the penalty pushes the margin toward closing by penalty/temperature
    shift = protocol.PRESENCE_PENALTY / protocol.TEMPERATURE
    assert out["sm_primary"] - out["sm_unpenalised"] == pytest.approx(shift)
    # and the unpenalised margin is exactly the raw one under temperature
    assert out["sm_unpenalised"] == pytest.approx(
        out["sm_raw"] / protocol.TEMPERATURE)


def test_the_states_agree_when_nothing_has_been_generated():
    """At a zero-answer anchor the presence set is empty (`[]` is one token, so
    the decision falls on the first generated token), and the two stages are
    then in the same state by construction."""
    import numpy as np
    from tcr.support_probe.boundary import sampler_readouts
    out = sampler_readouts(np.array([2.0, 1.0, 0.5]), close_token=0,
                           continue_token=1, presence_ids=[])
    assert out["sm_primary"] == pytest.approx(out["sm_unpenalised"])
    assert out["close_sampler_prob"] == pytest.approx(
        out["close_sampler_prob_unpenalised"])
    assert out["n_presence_ids"] == 0


def test_the_penalty_cancels_from_a_delta_but_not_from_a_level():
    """Why ACI/ECI are untouched by the state and the hazard's rate is not: the
    prefix is identical across an anchor's variants, so the penalty is the same
    constant in both arms."""
    import numpy as np
    from tcr.support_probe.boundary import sampler_readouts
    presence = [1, 1, 2]
    manip = sampler_readouts(np.array([2.5, 1.0, 0.5]), close_token=0,
                             continue_token=1, presence_ids=presence)
    neutral = sampler_readouts(np.array([2.0, 1.0, 0.5]), close_token=0,
                               continue_token=1, presence_ids=presence)
    delta_primary = manip["sm_primary"] - neutral["sm_primary"]
    delta_unpenalised = manip["sm_unpenalised"] - neutral["sm_unpenalised"]
    assert delta_primary == pytest.approx(delta_unpenalised)
    assert manip["sm_primary"] != pytest.approx(manip["sm_unpenalised"])
