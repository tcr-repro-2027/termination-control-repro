# coding=utf-8
"""Unit tests for the tcr.token_orbit package (no transformers needed).

Run from the package project dir:   python tests/test_loop_detect.py
"""

import os
import sys

# Make the package importable without installation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tcr.token_orbit import (
    DetectorConfig,
    PeriodConfig,
    build_text,
    detect_loop,
    find_periodic_run,
    gzip_ratio,
    rep_n,
    score_response,
)


# --------------------------------------------------------------------------- #
# Fake char-level tokenizer: one token per character, so token index == char
# index.  This lets us assert the position wiring exactly.
# --------------------------------------------------------------------------- #
def fake_encode(text):
    ids = [ord(c) for c in text]
    offsets = [(i, i + 1) for i in range(len(text))]
    return ids, offsets


# --------------------------------------------------------------------------- #
# Periodicity kernel
# --------------------------------------------------------------------------- #
def test_periodic_run_onset_period_repeats():
    prefix = [1, 2, 3]
    unit = [7, 8]
    line = prefix + unit * 12 + [9]
    run = find_periodic_run(line, p_min=1, p_max=10, k=10)
    assert run is not None
    assert run.onset == len(prefix)
    assert run.period == 2
    assert run.num_repeats == 12
    assert run.unit == unit
    print("PASS test_periodic_run_onset_period_repeats")


def test_scattered_repeats_not_flagged():
    # The motif [7, 8] appears 12 times but never >= k times CONSECUTIVELY.
    line = []
    for i in range(12):
        line += [7, 8, 100 + i]
    assert find_periodic_run(line, p_min=1, p_max=10, k=10) is None
    print("PASS test_scattered_repeats_not_flagged")


def test_below_k_repeats_not_flagged():
    line = [1, 2, 3] + [7, 8] * 5 + [4, 5, 6]
    assert find_periodic_run(line, p_min=1, p_max=10, k=10) is None
    print("PASS test_below_k_repeats_not_flagged")


def test_smaller_period_wins_ties():
    # [7, 7, 7, ...] is periodic with p=1 and p=2 from the same onset; the
    # minimal unit (p=1) must be reported.
    line = [7] * 30
    run = find_periodic_run(line, p_min=1, p_max=10, k=10)
    assert run is not None and run.period == 1 and run.onset == 0
    print("PASS test_smaller_period_wins_ties")


# --------------------------------------------------------------------------- #
# detect_loop: threshold gate + positions + content
# --------------------------------------------------------------------------- #
def test_detect_loop_positions_and_content():
    prefix = "ANSWER="
    text = prefix + "loop " * 50
    config = DetectorConfig(threshold=50, period=PeriodConfig(), before_tokens=100)
    entry = detect_loop(text, fake_encode, config)
    assert entry is not None, "expected a loop"
    assert entry["onset"] == len(prefix)
    assert entry["onsetChar"] == len(prefix)      # char == token index here
    assert entry["period"] == 5
    assert entry["numRepeats"] == 50
    assert entry["loopContent"] == "loop "
    # 100 tokens before the loop, clamped to the start -> the whole prefix.
    assert entry["contextBefore"] == prefix
    print("PASS test_detect_loop_positions_and_content")


def test_detect_loop_then_different_ending():
    # Repeats many times, then ends with a DIFFERENT, non-repeating sentence;
    # the whole-sequence scan must still flag it.
    text = "ANSWER=" + "loop " * 50 + "and a totally different closing sentence."
    config = DetectorConfig(threshold=50, period=PeriodConfig(), before_tokens=100)
    entry = detect_loop(text, fake_encode, config)
    assert entry is not None and entry["onset"] == len("ANSWER=")
    print("PASS test_detect_loop_then_different_ending")


def test_detect_loop_threshold_gate():
    text = "loop " * 50                       # 250 tokens
    config = DetectorConfig(threshold=1000)   # under the gate -> not scanned
    assert detect_loop(text, fake_encode, config) is None
    print("PASS test_detect_loop_threshold_gate")


# --------------------------------------------------------------------------- #
# score_response: the unified four-metric record
# --------------------------------------------------------------------------- #
def test_score_response_looping():
    text = "X=" + "ab" * 200
    config = DetectorConfig(threshold=100, period=PeriodConfig())
    rec = score_response(text, "length", fake_encode, config)
    assert rec["loop"] == 1
    assert rec["hit_max_tokens"] == 1
    assert rec["gen_len"] == len(text)
    assert rec["onset"] == 2 and rec["period"] == 2     # "ab" repeats from index 2
    assert rec["loopContent"] == "ab"
    assert rec["numRepeats"] == 200
    assert 0.0 <= rec["gzip_ratio"] <= 1.0
    assert rec["rep4"] > 0.9
    print("PASS test_score_response_looping")


def test_score_response_clean():
    text = "a perfectly normal short answer with no repetition at all."
    config = DetectorConfig(threshold=10, period=PeriodConfig())
    rec = score_response(text, "stop", fake_encode, config)
    assert rec["loop"] == 0
    assert "onset" not in rec
    assert rec["hit_max_tokens"] == 0
    assert rec["rep4"] < 0.2
    print("PASS test_score_response_clean")


def test_severity_metrics_ordering():
    repetitive = "spam " * 200
    diverse = "".join(chr(ord("a") + (i * 7) % 26) for i in range(1000))
    assert rep_n(fake_encode(repetitive)[0]) > rep_n(fake_encode(diverse)[0])
    assert gzip_ratio(repetitive) < gzip_ratio(diverse)
    print("PASS test_severity_metrics_ordering")


def test_empty_text():
    config = DetectorConfig(threshold=10)
    rec = score_response("", "stop", fake_encode, config)
    assert rec == {"loop": 0, "hit_max_tokens": 0, "rep4": 0.0,
                   "gzip_ratio": 1.0, "gen_len": 0}
    assert detect_loop("", fake_encode, config) is None
    print("PASS test_empty_text")


# --------------------------------------------------------------------------- #
# build_text: detection target selection
# --------------------------------------------------------------------------- #
def test_build_text_targets():
    assert build_text("A", "R", "answer") == "A"
    assert build_text("A", "R", "reasoning") == "R"
    assert build_text("A", "R", "full") == "R\nA"
    assert build_text("A", "", "full") == "A"
    assert build_text("", "R", "full") == "R"
    try:
        build_text("A", "R", "bogus")
        raise AssertionError("expected ValueError for unknown target")
    except ValueError:
        pass
    print("PASS test_build_text_targets")


if __name__ == "__main__":
    test_periodic_run_onset_period_repeats()
    test_scattered_repeats_not_flagged()
    test_below_k_repeats_not_flagged()
    test_smaller_period_wins_ties()
    test_detect_loop_positions_and_content()
    test_detect_loop_then_different_ending()
    test_detect_loop_threshold_gate()
    test_score_response_looping()
    test_score_response_clean()
    test_severity_metrics_ordering()
    test_empty_text()
    test_build_text_targets()
    print("\nAll tcr.token_orbit tests passed.")
