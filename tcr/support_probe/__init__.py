# coding=utf-8
"""E2 -- support-conditioning instrument: anchors, StopMargin, calibration.

    tcr.support_probe.build          eval set        -> anchors.jsonl (A/B/D + neutral + PC)
    tcr.support_probe.score          anchors + model -> teacher-forced StopMargin readouts
    e2.hazard         anchors + model -> short-continuation stop hazard
    tcr.support_probe.analyze        readouts        -> ACI/ECI slopes, PC, nSCI, gate report
    tcr.support_probe.orchestrator   the whole matrix on 8 GPUs, unattended

The frozen protocol is `tcr.support_probe.protocol`; nothing else decides a sampler value, an
anchor count or a manipulation level.
"""

__all__ = ["analyze", "anchors", "boundary", "build", "hazard", "io_utils",
           "orchestrator", "prompts", "protocol", "score"]
