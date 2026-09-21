from __future__ import annotations

from conftest import char_offsets, relation, surface
from tcr.events.audit import audit_response


def test_response_audit_orders_online_events_not_retrospective_onset():
    motif = [relation(0), relation(1)]
    text = surface(motif * 5)
    offsets = char_offsets(text)
    # Character tokenizer for the synthetic test.  Raw onset is deliberately
    # inside the first block, while confirmation is much later.
    starts = []
    cursor = 0
    for _ in range(10):
        cursor = text.find('{', cursor)
        starts.append(cursor)
        cursor += 1
    period = starts[2] - starts[0]
    onset = starts[0] + 2
    legacy = {
        "loop": 1,
        "onset": onset,
        "onsetChar": onset,
        "period": period,
        "numRepeats": 3,
        "hit_max_tokens": 1,
        "gen_len": len(text),
    }
    row = audit_response(
        model_tag="M1",
        key=1,
        seed=0,
        text=text,
        token_ids=[ord(char) for char in text],
        offsets=offsets,
        legacy_sample=legacy,
        selection_roles=["prevalence"],
        capture_min_repeats=3,
        max_motif_period_blocks=None,
        legacy_min_repeats=3,
    )
    assert row["first_triple_reuse"]["exists"]
    assert row["motif_capture_triple"]["exists"]
    assert row["legacy_orbit"]["exists"]
    assert row["alignment_evidence"]["raw_onset_moved"] is False
    assert row["temporal_alignment"]["online_confirmation_order_valid"] is True
    assert row["legacy_orbit"]["raw_onset_token"] < row["first_triple_reuse"]["confirmed_at_token_exclusive"]
