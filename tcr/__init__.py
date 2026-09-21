# coding: utf-8
"""Library code for "From Hallucinated Targets to Runaway Repetition".

One package, no cross-project imports.  Sub-packages, in pipeline order:

    data            supervision construction (stages, raw arms, OBR, label noise)
    prompt_template the single prompt used for training AND evaluation
    evaluation      natural generation (vLLM) + per-response scoring
    events          relation-block parser, reuse events, continuous pattern
                    repetition, and the two-stage competing-risk process
    token_orbit     stable token orbit detector
    extraction      triple / entity-pair F1
    boundary        close-vs-continue decision readouts on a fixed prefix
    steering        contrast direction, single-pulse hook, SAE reading
    motif           artificial complete-motif gain probes and scoring
    support_probe   input-support probe (appendix)
    paper           model registry, paths, paired bootstrap, figure style
"""

__version__ = "1.0.0"
