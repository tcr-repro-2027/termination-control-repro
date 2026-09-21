from __future__ import annotations

import pytest

from tcr.events.legacy import adapt_legacy_sample


def test_legacy_periodicity_is_reverified_exactly():
    ids = [9, 8] + [1, 2] * 4
    sample = {
        "loop": 1,
        "onset": 2,
        "period": 2,
        "numRepeats": 4,
        "hit_max_tokens": 1,
    }
    orbit = adapt_legacy_sample(
        sample, min_repeats=3, n_tokens=len(ids), token_ids=ids
    )
    assert orbit.periodicity_reverified is True
    assert orbit.orbit_confirmed_at_token_exclusive == 8

    broken = ids.copy()
    broken[7] = 99
    with pytest.raises(ValueError, match="periodicity failed exact replay"):
        adapt_legacy_sample(
            sample, min_repeats=3, n_tokens=len(broken), token_ids=broken
        )
