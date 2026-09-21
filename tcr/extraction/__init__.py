# coding=utf-8
"""Reference-extraction scoring: strict (triple) and relaxed (entity-pair) micro-F1.

* ``protocol``            -- the frozen sampling protocol (per-mode params, K, seeds).
* ``evaluation.parsing``  -- response -> JSON relation list.
* ``evaluation.metrics``  -- normalisation, triple / pair sets, pooled P/R/F1.
"""
