# coding: utf-8
"""Which models the paper uses, what role each plays, and how they pair up.

One table, because the alternative is every entry hard-coding its own list of
tags and the tables silently disagreeing about which arm is the reference.
Tags are the E1 tags exactly as `e1_metrics.csv` writes them -- including the
two that carry no seed suffix (`qwen3-1.7b-cleanv2`, `qwen3-8b-cleanv2`), whose
training seed is 42 and is recorded here rather than parsed out of the name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    size: str            # 1.7B | 4B | 8B
    condition: str       # notrain | cleanv2 | keep4 | obr_p5 ... | isc_a ...
    train_seed: int | None
    role: str            # reference | C | K | O | control | type
    section: str         # main | appendix
    label: str           # short human label used in tables and figures


#: Every model this paper reads.  `role` is what it does in the argument:
#: C = entity-constraint-cleaned reference, K = raw keep4, O = one-for-one
#: target replacement, control/type = appendix conditions, reference = no SFT.
MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("qwen3-1.7b-notrain", "1.7B", "notrain", None, "reference", "main", "1.7B no SFT"),
    ModelSpec("qwen3-4b-notrain", "4B", "notrain", None, "reference", "main", "4B no SFT"),
    ModelSpec("qwen3-8b-notrain", "8B", "notrain", None, "reference", "main", "8B no SFT"),

    ModelSpec("qwen3-4b-cleanv2-s42", "4B", "cleanv2", 42, "C", "main", "4B cleaned s42"),
    ModelSpec("qwen3-4b-cleanv2-s123", "4B", "cleanv2", 123, "C", "main", "4B cleaned s123"),
    ModelSpec("qwen3-4b-keep4-s42", "4B", "keep4", 42, "K", "main", "4B raw s42"),
    ModelSpec("qwen3-4b-keep4-s123", "4B", "keep4", 123, "K", "main", "4B raw s123"),
    ModelSpec("qwen3-1.7b-cleanv2", "1.7B", "cleanv2", 42, "C", "main", "1.7B cleaned s42"),
    ModelSpec("qwen3-1.7b-keep4-s42", "1.7B", "keep4", 42, "K", "main", "1.7B raw s42"),
    ModelSpec("qwen3-8b-cleanv2", "8B", "cleanv2", 42, "C", "main", "8B cleaned s42"),
    ModelSpec("qwen3-8b-keep4-s42", "8B", "keep4", 42, "K", "main", "8B raw s42"),

    ModelSpec("qwen3-4b-obr-p5-s42", "4B", "obr_p5", 42, "O", "main", "OBR 5% s42"),
    ModelSpec("qwen3-4b-obr-p10-s42", "4B", "obr_p10", 42, "O", "main", "OBR 10% s42"),
    ModelSpec("qwen3-4b-obr-p15-s42", "4B", "obr_p15", 42, "O", "main", "OBR 15% s42"),
    ModelSpec("qwen3-4b-obr-p15-s123", "4B", "obr_p15", 123, "O", "main", "OBR 15% s123"),
    ModelSpec("qwen3-4b-obr-s42", "4B", "obr", 42, "O", "appendix", "OBR 24.3% s42"),

    ModelSpec("qwen3-4b-isc-a-s42", "4B", "isc_a", 42, "control", "appendix", "ISC-A"),
    ModelSpec("qwen3-4b-isc-e-s42", "4B", "isc_e", 42, "control", "appendix", "ISC-E"),
    ModelSpec("qwen3-4b-isc-ae-s42", "4B", "isc_ae", 42, "control", "appendix", "ISC-AE"),
    ModelSpec("qwen3-4b-benign-input-s42", "4B", "benign_input", 42, "control", "appendix", "BenignInput"),
    ModelSpec("qwen3-4b-generic-noise-s42", "4B", "generic_noise", 42, "control", "appendix", "GenericNoise"),
    ModelSpec("qwen3-4b-keep4a-s42", "4B", "keep4_a", 42, "type", "appendix", "keep4-A"),
    ModelSpec("qwen3-4b-keep4ae-s42", "4B", "keep4_ae", 42, "type", "appendix", "keep4-AE"),
)

BY_TAG: dict[str, ModelSpec] = {spec.tag: spec for spec in MODELS}


def spec(tag: str) -> ModelSpec:
    try:
        return BY_TAG[tag]
    except KeyError:
        raise KeyError(f"{tag!r} is not a model this paper uses; known tags: "
                       + ", ".join(sorted(BY_TAG))) from None


def label(tag: str) -> str:
    return BY_TAG[tag].label if tag in BY_TAG else tag


@dataclass(frozen=True)
class Pair:
    name: str            # directory-safe identifier
    m0: str              # baseline arm tag
    m1: str              # treated arm tag
    purpose: str
    with_p0c: bool = False


#: The nine R0 pairs, all computable in one pass now that every model is
#: evaluated.  M0 is always the arm the comparison treats as the reference, so
#: a positive M1-M0 difference always means "the treatment made it worse".
R0_PAIRS: tuple[Pair, ...] = (
    Pair("4b_CK_s42", "qwen3-4b-cleanv2-s42", "qwen3-4b-keep4-s42",
         "natural main contrast, 4B seed 42", with_p0c=True),
    Pair("4b_CO15_s42", "qwen3-4b-cleanv2-s42", "qwen3-4b-obr-p15-s42",
         "controlled target replacement at the main dose"),
    Pair("4b_notrainC_s42", "qwen3-4b-notrain", "qwen3-4b-cleanv2-s42",
         "what this task's SFT alone does; the reversed per-episode hazard"),
    Pair("4b_CGN_s42", "qwen3-4b-cleanv2-s42", "qwen3-4b-generic-noise-s42",
         "ordinary content noise that did not reproduce the OBR effect"),
    Pair("4b_CISCA_s42", "qwen3-4b-cleanv2-s42", "qwen3-4b-isc-a-s42",
         "input-side manipulation that did not raise repetition"),
    Pair("4b_CK_s123", "qwen3-4b-cleanv2-s123", "qwen3-4b-keep4-s123",
         "natural main contrast, second training seed"),
    Pair("4b_CO15_s123", "qwen3-4b-cleanv2-s123", "qwen3-4b-obr-p15-s123",
         "controlled replacement, second training seed"),
    Pair("1p7b_CK_s42", "qwen3-1.7b-cleanv2", "qwen3-1.7b-keep4-s42",
         "same-family scale check, 1.7B"),
    Pair("8b_CK_s42", "qwen3-8b-cleanv2", "qwen3-8b-keep4-s42",
         "same-family scale check, 8B", with_p0c=True),
)


#: R1 scores every model in one pool per scale.  The pool itself is built from
#: the seed-42 C and K responses of that scale only (§4.2); which models are
#: then SCORED on it is a separate decision, and this is it.
R1_SCORED: dict[str, tuple[str, ...]] = {
    "4B": (
        "qwen3-4b-cleanv2-s42", "qwen3-4b-keep4-s42", "qwen3-4b-obr-p15-s42",
        "qwen3-4b-cleanv2-s123", "qwen3-4b-keep4-s123", "qwen3-4b-obr-p15-s123",
    ),
    "8B": ("qwen3-8b-cleanv2", "qwen3-8b-keep4-s42"),
}

#: The prefix pool of each scale is cut from these two arms, one per source.
R1_POOL_SOURCES: dict[str, tuple[str, str]] = {
    "4B": ("qwen3-4b-cleanv2-s42", "qwen3-4b-keep4-s42"),
    "8B": ("qwen3-8b-cleanv2", "qwen3-8b-keep4-s42"),
}

#: R1's paired contrasts, as `(label, reference, treatment, kind)`.
#:
#: Which arm supplies the PREFIXES and which arm is the REFERENCE of a contrast
#: are two different decisions.  The pool is cut from the seed-42 arms because
#: those are the trajectories the paper's main comparison lives on; a contrast,
#: however, has to hold the training seed fixed, or the treatment effect and the
#: training randomness end up in the same number.  The seed-only contrast is
#: kept and reported separately, as the scale against which the others are read.
R1_CONTRASTS: dict[str, tuple[tuple[str, str, str, str], ...]] = {
    "4B": (
        ("raw - cleaned (s42)", "qwen3-4b-cleanv2-s42", "qwen3-4b-keep4-s42",
         "treatment"),
        ("OBR 15% - cleaned (s42)", "qwen3-4b-cleanv2-s42",
         "qwen3-4b-obr-p15-s42", "treatment"),
        ("raw - cleaned (s123)", "qwen3-4b-cleanv2-s123", "qwen3-4b-keep4-s123",
         "treatment"),
        ("OBR 15% - cleaned (s123)", "qwen3-4b-cleanv2-s123",
         "qwen3-4b-obr-p15-s123", "treatment"),
        ("training seed only (cleaned s123 - s42)", "qwen3-4b-cleanv2-s42",
         "qwen3-4b-cleanv2-s123", "seed"),
    ),
    "8B": (
        ("raw - cleaned (s42)", "qwen3-8b-cleanv2", "qwen3-8b-keep4-s42",
         "treatment"),
    ),
}

#: R2 fits the direction on C/K only and transfers it to the OBR arm, which
#: never participates in the fit.  Order matters: (C, K, transfer...).
R2_MODELS: dict[str, dict[str, Any]] = {
    "4B": {
        "clean": "qwen3-4b-cleanv2-s42",
        "raw": "qwen3-4b-keep4-s42",
        "transfer": ("qwen3-4b-obr-p15-s42",),
        "long_form": True,
        "sae": False,
    },
    "8B": {
        "clean": "qwen3-8b-cleanv2",
        "raw": "qwen3-8b-keep4-s42",
        "transfer": (),
        "long_form": False,
        "sae": True,
    },
}

#: X1 scores one fixed probe pool with these four 4B models.
X1_MODELS: tuple[str, ...] = (
    "qwen3-4b-notrain", "qwen3-4b-cleanv2-s42",
    "qwen3-4b-keep4-s42", "qwen3-4b-obr-p15-s42",
)

#: Table 1 rows, in reading order.
T1_ROWS: tuple[str, ...] = (
    "qwen3-1.7b-notrain", "qwen3-1.7b-cleanv2", "qwen3-1.7b-keep4-s42",
    "qwen3-4b-notrain", "qwen3-4b-cleanv2-s42", "qwen3-4b-keep4-s42",
    "qwen3-4b-cleanv2-s123", "qwen3-4b-keep4-s123",
    "qwen3-8b-notrain", "qwen3-8b-cleanv2", "qwen3-8b-keep4-s42",
)

#: Table 1's paired contrasts: same scale, same training seed, K minus C.
T1_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("1.7B s42", "qwen3-1.7b-cleanv2", "qwen3-1.7b-keep4-s42"),
    ("4B s42", "qwen3-4b-cleanv2-s42", "qwen3-4b-keep4-s42"),
    ("4B s123", "qwen3-4b-cleanv2-s123", "qwen3-4b-keep4-s123"),
    ("8B s42", "qwen3-8b-cleanv2", "qwen3-8b-keep4-s42"),
)

#: Table 2: the dose series, plus the second seed of the main dose.  0% is
#: cleanv2 itself -- no separate OBR-0 arm was ever built or trained.
T2_ROWS: tuple[tuple[str, str, float], ...] = (
    ("OBR 0% s42", "qwen3-4b-cleanv2-s42", 0.0),
    ("OBR 5% s42", "qwen3-4b-obr-p5-s42", 0.05),
    ("OBR 10% s42", "qwen3-4b-obr-p10-s42", 0.10),
    ("OBR 15% s42", "qwen3-4b-obr-p15-s42", 0.15),
    ("OBR 0% s123", "qwen3-4b-cleanv2-s123", 0.0),
    ("OBR 15% s123", "qwen3-4b-obr-p15-s123", 0.15),
)

T2_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("5% - 0% s42", "qwen3-4b-cleanv2-s42", "qwen3-4b-obr-p5-s42"),
    ("10% - 0% s42", "qwen3-4b-cleanv2-s42", "qwen3-4b-obr-p10-s42"),
    ("15% - 0% s42", "qwen3-4b-cleanv2-s42", "qwen3-4b-obr-p15-s42"),
    ("15% - 0% s123", "qwen3-4b-cleanv2-s123", "qwen3-4b-obr-p15-s123"),
)

#: Appendix A rows: everything that bounds the claim without carrying it.
APPENDIX_A_ROWS: tuple[str, ...] = (
    "qwen3-4b-cleanv2-s42", "qwen3-4b-obr-s42", "qwen3-4b-isc-a-s42",
    "qwen3-4b-isc-e-s42", "qwen3-4b-isc-ae-s42", "qwen3-4b-benign-input-s42",
    "qwen3-4b-generic-noise-s42", "qwen3-4b-keep4a-s42", "qwen3-4b-keep4ae-s42",
    "qwen3-4b-keep4-s42",
)


def all_tags() -> list[str]:
    return [spec.tag for spec in MODELS]


def tags_of(*roles: str) -> list[str]:
    wanted = set(roles)
    return [spec.tag for spec in MODELS if spec.role in wanted]


def required_tags(*groups: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for group in groups:
        for tag in group:
            if tag not in seen:
                seen.append(tag)
    return seen


def missing(tags: Sequence[str], available: Iterable[str]) -> list[str]:
    have = set(available)
    return [tag for tag in tags if tag not in have]
