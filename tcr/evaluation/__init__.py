# coding=utf-8
"""E1 -- unified re-evaluation of every trained model on one frozen protocol.

Stages, each a standalone CLI so any one of them can be re-run by hand:

    tcr.evaluation.generate       one checkpoint on one GPU -> responses jsonl
    tcr.evaluation.analyze        one responses jsonl -> event rows + summary + CSV row
    tcr.evaluation.orchestrator   the whole matrix, 8 GPUs, unattended

The frozen protocol is `tcr.evaluation.protocol`; nothing else may decide a sampling
parameter, a detector threshold or the evaluation set.
"""

__all__ = ["aggregate", "analyze", "episodes", "generate", "io_utils",
           "orchestrator", "prompts", "protocol", "quality", "registry",
           "structured"]
